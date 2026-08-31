"""In-graph audio chunking and prompt requeue for H3 continuation chains."""

import copy
from fractions import Fraction
import hashlib
import logging
import math
import os
import re
import subprocess
import threading
import time
import uuid
import wave

from comfy_api.latest import InputImpl, Types
import folder_paths
import imageio_ffmpeg
import numpy as np
import server
import torch
from PIL import Image
from comfy_extras.nodes_minimax_h3 import (
    adapt_canvas,
    align_frame_count,
    _resize as _h3_resize,
)

from .motion_context_core import (
    AutoChainMotionContextCore,
    MiniMaxH3MotionContextTrim,
    VIDEO_RUN_GRID,
)

try:
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save
except ImportError:
    _st_load = None
    _st_save = None


_LOG = logging.getLogger("h3_motion_context")
_CHAINS = {}
_LOCK = threading.Lock()
DEFAULT_FPS = 24.0
H3_AUDIO_SAMPLE_RATE = 32000
H3_FRAME_BLOCK = 17
H3_VALID_LENGTH_OFFSET = 5


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    return list(samples)


def _resolve_latent_path(path, clip_index=0):
    value = (path or "").strip().strip('"').strip("'") or "h3_context"
    candidates = [value, os.path.join(folder_paths.get_output_directory(), value)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        if not os.path.isdir(candidate):
            continue
        index = int(clip_index)
        if index > 0:
            endings = ("_%05d.safetensors" % index,
                       "_clip%03d.safetensors" % index)
            files = [os.path.join(candidate, name)
                     for name in os.listdir(candidate) if name.endswith(endings)]
            if not files:
                raise FileNotFoundError(
                    "h3_motion_context: no saved latent for clip %d in %s" %
                    (index, candidate))
        else:
            files = [os.path.join(candidate, name)
                     for name in os.listdir(candidate)
                     if name.endswith(".safetensors")]
            if not files:
                raise FileNotFoundError(
                    "h3_motion_context: no saved latents in %s" % candidate)
        return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a folder" % value)


def _chain_latent_path(chain_config, clip_index):
    return _output_path(
        chain_config["latent_prefix"],
        "_%05d.safetensors" % int(clip_index))


class MiniMaxH3AutoChainSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "chain_config": ("H3_CHAIN",),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"

    def save(self, latent, chain_config):
        if _st_save is None:
            raise RuntimeError("h3_motion_context: safetensors is unavailable")
        filename_prefix = chain_config["latent_prefix"]
        clip_index = chain_config["save_clip_index"]
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError("h3_motion_context: latent has no audio stream")
        video, audio = (part.cpu().contiguous() for part in parts[:2])
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            name = "%s_%05d.safetensors" % (filename, int(clip_index))
        else:
            name = "%s_%05d_.safetensors" % (filename, counter)
        path = os.path.join(folder, name)
        _st_save({"video": video, "audio": audio}, path,
                 metadata={"format": "h3_motion_context_av_v1"})
        _LOG.info("h3_motion_context: saved clip %d latent to %s",
                  int(clip_index), path)
        return (path,)


class MiniMaxH3AutoChainLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chain_config": ("H3_CHAIN",),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"

    @classmethod
    def IS_CHANGED(cls, chain_config):
        clip_index = chain_config["load_clip_index"]
        reset = chain_config["reset"]
        if reset:
            return float("NaN")
        try:
            path = _chain_latent_path(chain_config, clip_index)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            return "%s:%d" % (path, os.stat(path).st_mtime_ns)
        except FileNotFoundError:
            return float("NaN")

    def load(self, chain_config):
        clip_index = chain_config["load_clip_index"]
        reset = chain_config["reset"]
        if reset or int(clip_index) == 0:
            return (None,)
        try:
            path = _chain_latent_path(chain_config, clip_index)
        except FileNotFoundError:
            _LOG.warning("h3_motion_context: latent for clip %d is missing; "
                         "bypassing Motion Context for this clip",
                         int(clip_index) + 1)
            return (None,)
        if not os.path.isfile(path):
            _LOG.warning("h3_motion_context: latent for clip %d is missing at %s; "
                         "bypassing Motion Context for this clip",
                         int(clip_index) + 1, path)
            return (None,)
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is unavailable")
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError("h3_motion_context: invalid H3 AV latent: %s" % path)
        # Detach from safetensors' file mapping so Windows can remove the
        # completed chain files after the final stitch.
        video = data["video"].clone()
        audio = data["audio"].clone()
        _LOG.info("h3_motion_context: loaded clip %d latent from %s",
                  int(clip_index), path)
        return ({"samples": [video, audio]},)


class MiniMaxH3AutoChainMotionContext:
    """Run the standalone Motion Context core, allowing clip one to have no context."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = AutoChainMotionContextCore.INPUT_TYPES()
        inputs["required"].pop("context_length")
        inputs["required"].pop("audio_context_length")
        inputs["required"]["chain_config"] = ("H3_CHAIN", {
            "tooltip": "Auto Split configuration. Its context-frame count "
                       "sets the pinned video and audio context spans and "
                       "the matching trim."})
        inputs.setdefault("optional", {})["endless_continuation"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Append the previous clip's synchronized video/audio "
                       "tail as an H3 continuation reference, like HR Endless "
                       "Sampler."})
        return inputs

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"

    def apply(self, conditioning, vae, latent, chain_config,
              context_frames=None,
              context_latent=None, audio_vae=None, context_audio=None,
              endless_continuation=False):
        context_length = int(chain_config["effective_trim_frames"])
        if context_length == 0:
            return (conditioning, 0)
        if context_latent is None and context_frames is None:
            raise RuntimeError(
                "h3_motion_context: clip %d reserves %d context frames, "
                "but H3 Auto Chain Motion Context has no context input. "
                "Connect the previous latent or disable context for this "
                "chain."
                % (int(chain_config["clip_index"]), context_length))
        out, trim = AutoChainMotionContextCore().apply(
            conditioning, vae, latent, context_length,
            context_length, context_frames, context_latent,
            audio_vae, context_audio, endless_continuation)
        if int(trim) != context_length:
            raise RuntimeError(
                "h3_motion_context: clip %d produced a %d-frame context "
                "head, but Auto Split reserved %d. Refusing to desync its "
                "audio."
                % (int(chain_config["clip_index"]), int(trim), context_length))
        return (out, trim)


class MiniMaxH3AutoChainMotionContextTrim(MiniMaxH3MotionContextTrim):
    """Standalone trim node paired with Auto Chain Motion Context."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(MiniMaxH3MotionContextTrim.INPUT_TYPES())
        inputs["optional"].pop("fps")
        inputs["required"]["chain_config"] = ("H3_CHAIN", {
            "tooltip": "Connect Auto Split's chain_config. Its FPS and "
                       "reserved context span keep this trim on the same "
                       "timeline as the audio chunk."})
        return inputs

    def trim(self, images, trim_frames, chain_config, audio=None,
             match_tail=True):
        configured_trim = int(chain_config["effective_trim_frames"])
        actual_trim = int(trim_frames)
        if actual_trim != configured_trim:
            raise ValueError(
                "h3_motion_context: Motion Context returned %d trim frames "
                "for clip %d, but Auto Split reserved %d. Connect this "
                "node's trim_frames input directly to H3 Auto Chain Motion "
                "Context."
                % (actual_trim, int(chain_config["clip_index"]),
                   configured_trim))
        return super().trim(images, actual_trim, audio,
                            float(chain_config["fps"]), match_tail)


class MiniMaxH3AutoChainCreateVideo:
    """Create a chain clip at the FPS owned by Auto Split."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "chain_config": ("H3_CHAIN", {
                "tooltip": "Connect Auto Split's chain_config. This node "
                           "uses its FPS instead of a separate widget."}),
        }, "optional": {
            "audio": ("AUDIO",),
            "bit_depth": ("INT", {"default": 8, "min": 8, "max": 10}),
            "color_space": (["sRGB", "HDR", "HDR PQ"], {
                "default": "sRGB"}),
        }}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "create"
    CATEGORY = "conditioning/minimax"

    def create(self, images, chain_config, audio=None, bit_depth=8,
               color_space="sRGB"):
        frame_count = int(images.shape[0])
        expected = int(chain_config["clip_frames"])
        if frame_count != expected:
            raise ValueError(
                "h3_motion_context: H3 Auto Chain Create Video received %d "
                "frames but Auto Split planned %d. Check Motion Context "
                "Trim."
                % (frame_count, expected))
        if audio is not None:
            audio = _match_audio_duration(
                audio, frame_count, float(chain_config["fps"]))
        return (InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio,
                frame_rate=Fraction(str(float(chain_config["fps"]))),
            ),
            bit_depth=int(bit_depth), color_space=color_space),)


def _prompt_for_clip(style_prompt, clip_prompts, clip):
    text = str(clip_prompts)
    matches = list(re.finditer(
        r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)", text))
    if matches:
        prompts = {int(m.group(1)): m.group(2).strip() for m in matches}
        prompt = prompts.get(int(clip), "")
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        prompt = lines[min(int(clip) - 1, len(lines) - 1)] if lines else ""
    style = str(style_prompt).strip()
    return style + "\n" + prompt if style and prompt else style or prompt


def _run_name(chain_id):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(chain_id)).strip("._")
    if not name:
        raise ValueError("h3_motion_context: chain ID must contain a filename character")
    return name


def _output_path(prefix, suffix=""):
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    relative = os.path.normpath(prefix.replace("/", os.sep))
    path = os.path.abspath(os.path.join(output_dir, relative + suffix))
    if os.path.commonpath((output_dir, path)) != output_dir:
        raise ValueError("h3_motion_context: output prefix must stay inside "
                         "the ComfyUI output directory")
    return path


def _replace_output_file(source, target):
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 9:
                raise PermissionError(
                    "h3_motion_context: cannot replace %s because Windows "
                    "still has it open. Close its ComfyUI/browser/Explorer "
                    "preview, or use a new chain ID."
                    % target)
            time.sleep(0.2)


def _cleanup_chain_latents(run_name):
    folder = _output_path("h3_context")
    prefix = "%s_clip_" % run_name
    removed = 0
    for name in os.listdir(folder) if os.path.isdir(folder) else ():
        if not (name.startswith(prefix) and name.endswith(".safetensors")):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                _LOG.warning("h3_motion_context: could not remove latent %s: %s",
                             path, exc)
    _LOG.info("h3_motion_context: removed %d completed-chain latent file(s) for %s",
              removed, run_name)


def _write_audio_wav(audio, path):
    waveform = audio["waveform"].detach().cpu().float().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    samples = np.clip(waveform.T, -1.0, 1.0)
    pcm = np.rint(samples * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as output:
        output.setnchannels(int(pcm.shape[1]))
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm.tobytes())


def _chain_frame_range(clip, chunk_frames):
    """Return the delivered frame range for a token-aligned chain clip."""
    first_frames = int(chunk_frames) + H3_VALID_LENGTH_OFFSET
    if int(clip) == 1:
        return 0, first_frames
    start = first_frames + (int(clip) - 2) * int(chunk_frames)
    return start, start + int(chunk_frames)


def _chain_clip_count(total_frames, chunk_frames):
    first_frames = int(chunk_frames) + H3_VALID_LENGTH_OFFSET
    if total_frames <= first_frames:
        return 1
    return 1 + int(math.ceil((total_frames - first_frames) /
                             float(chunk_frames)))


def _h3_chain_chunk_frames(chunk_seconds, fps):
    requested = max(1, int(round(float(chunk_seconds) * float(fps))))
    aligned = max(H3_FRAME_BLOCK,
                  int(math.ceil(requested / float(H3_FRAME_BLOCK))) *
                  H3_FRAME_BLOCK)
    if aligned != requested:
        _LOG.warning(
            "h3_motion_context: adjusted %.3fs chunk from %d to %d frames "
            "to preserve H3 17-frame latent blocks",
            float(chunk_seconds), requested, aligned)
    return aligned


def _h3_context_frames(trim_frames):
    requested = max(0, int(trim_frames))
    if requested == 0:
        return 0
    return next(run for run in VIDEO_RUN_GRID if run <= requested)


def _available_chain_clips(run_name, total_frames, chunk_frames):
    """Return existing clips in timeline order, allowing sparse chains."""
    total_clips = _chain_clip_count(total_frames, chunk_frames)
    paths = []
    frame_counts = []
    clip_indices = []
    for clip in range(1, total_clips + 1):
        path = _output_path("video/%s" % run_name, "_clip_%03d.mp4" % clip)
        if not os.path.isfile(path):
            continue
        paths.append(path)
        start, end = _chain_frame_range(clip, chunk_frames)
        frame_counts.append(min(end - start, max(0, total_frames - start)))
        clip_indices.append(clip)
    return paths, frame_counts, clip_indices


def _available_chain_clips_sparse(run_name):
    folder = _output_path("video")
    if not os.path.isdir(folder):
        return [], [], []
    clips = []
    for name in os.listdir(folder):
        match = re.fullmatch(r"%s_clip_(\d+)\.mp4" % re.escape(run_name), name)
        if match:
            clips.append((int(match.group(1)), os.path.join(folder, name)))
    clips.sort(key=lambda item: item[0])
    return ([path for _, path in clips],
            [None] * len(clips),
            [index for index, _ in clips])


def _audio_for_clips(audio, clip_indices, chunk_frames, fps):
    """Concatenate source-audio ranges belonging to the available clips."""
    waveform = audio["waveform"].detach().cpu()
    sample_rate = int(audio["sample_rate"])
    pieces = []
    for clip in clip_indices:
        start_frame, end_frame = _chain_frame_range(clip, chunk_frames)
        start = _frame_sample_index(start_frame, sample_rate, fps)
        end = _frame_sample_index(end_frame, sample_rate, fps)
        pieces.append(waveform[..., start:min(end, waveform.shape[-1])])
    if not pieces:
        return {"waveform": waveform[..., :0], "sample_rate": sample_rate}
    return {"waveform": torch.cat(pieces, dim=-1), "sample_rate": sample_rate}


def _frame_sample_index(frame, sample_rate, fps):
    """Map a video frame boundary to the nearest source-audio sample."""
    rate = Fraction(str(float(fps))).limit_denominator(1000000)
    numerator = int(frame) * int(sample_rate) * rate.denominator
    denominator = int(rate.numerator)
    return (numerator + denominator // 2) // denominator


def _stitch_videos(paths, frame_counts, total_frames, source_audio,
                   final_path, fps):
    dimensions = []
    for path in paths:
        dimensions.append(InputImpl.VideoFromFile(path).get_dimensions())
    if len(set(dimensions)) > 1:
        _LOG.error(
            "h3_motion_context: refusing to stitch clips with mismatched "
            "dimensions: %s",
            ", ".join("%s=%sx%s" % (os.path.basename(path), width, height)
                      for path, (width, height) in zip(paths, dimensions)))
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        args.extend(["-i", path])
    filters = []
    concat_inputs = []
    for index, path in enumerate(paths):
        frames = frame_counts[index]
        if frames is None:
            filters.append("[%d:v:0]setpts=PTS-STARTPTS[v%d]" %
                           (index, index))
        else:
            filters.append(
                "[%d:v:0]trim=end_frame=%d,setpts=PTS-STARTPTS[v%d]" %
                (index, int(frames), index))
        concat_inputs.append("[v%d]" % index)
    filters.append("%sconcat=n=%d:v=1:a=0[v]" %
                   ("".join(concat_inputs), len(paths)))
    audio_path = None
    if source_audio is not None:
        audio_path = final_path + ".source.wav"
        _write_audio_wav(source_audio, audio_path)
        args.extend(["-i", audio_path])
        audio_index = len(paths)
        sample_rate = int(source_audio["sample_rate"])
        total_samples = _frame_sample_index(total_frames, sample_rate, fps)
        filters.append(
            "[%d:a:0]atrim=start_sample=0,apad,atrim=end_sample=%d,"
            "asetpts=PTS-STARTPTS[a]" % (audio_index, total_samples))
    else:
        audio_inputs = []
        for index, path in enumerate(paths):
            filters.append("[%d:a:0]asetpts=PTS-STARTPTS[a%d]" %
                           (index, index))
            audio_inputs.append("[a%d]" % index)
        filters.append("%sconcat=n=%d:v=0:a=1[a]" %
                       ("".join(audio_inputs), len(paths)))
    args.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-r", "%.6f" % fps, "-movflags", "+faststart", final_path,
    ])
    try:
        subprocess.run(args, check=True)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    return True


class MiniMaxH3SparseChainStitch:
    """Stitch whichever completed clips exist for a chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chain_id": ("STRING", {"default": "h3_auto_chain"}),
            "fps": ("FLOAT", {"default": DEFAULT_FPS, "min": 1.0,
                                "max": 240.0, "step": 0.001}),
            "chunk_seconds": ("FLOAT", {"default": 3.5, "min": 0.1,
                                          "max": 600.0, "step": 0.1}),
            "total_frames": ("INT", {"default": 0, "min": 0,
                                       "max": 10000000,
                                       "tooltip": "0 uses the complete "
                                                  "durations of the saved clips."}),
        }, "optional": {
            "chain_config": ("H3_CHAIN",),
            "audio": ("AUDIO",),
        }}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "stitch"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"

    def stitch(self, chain_id, fps=DEFAULT_FPS, chunk_seconds=3.5,
               total_frames=0, chain_config=None, audio=None):
        run_name = _run_name(chain_id)
        fps = float(fps)
        stitch_audio = None
        if chain_config is not None:
            run_name = chain_config["run_name"]
            fps = float(chain_config["fps"])
            paths, frame_counts, clip_indices = _available_chain_clips(
                run_name, int(chain_config["total_frames"]),
                int(chain_config["chunk_frames"]))
            if audio is not None:
                stitch_audio = _audio_for_clips(
                    audio, clip_indices, int(chain_config["chunk_frames"]), fps)
        else:
            with _LOCK:
                state = _CHAINS.get(str(chain_id))
            if state is not None:
                fps = float(state["fps"])
                paths, frame_counts, clip_indices = _available_chain_clips(
                    run_name, int(state["total_frames"]),
                    int(state["chunk_frames"]))
                if audio is not None:
                    stitch_audio = _audio_for_clips(
                        audio, clip_indices, int(state["chunk_frames"]), fps)
            elif int(total_frames) > 0:
                chunk_frames = _h3_chain_chunk_frames(chunk_seconds, fps)
                paths, frame_counts, clip_indices = _available_chain_clips(
                    run_name, int(total_frames), chunk_frames)
                if audio is not None:
                    stitch_audio = _audio_for_clips(
                        audio, clip_indices, chunk_frames, fps)
            else:
                paths, frame_counts, clip_indices = _available_chain_clips_sparse(
                    run_name)
        if not paths:
            raise FileNotFoundError(
                "h3_motion_context: no completed clips found for chain %s" %
                chain_id)
        final_path = _output_path("video/%s" % run_name, ".mp4")
        total_frames = sum(int(frames) for frames in frame_counts
                           if frames is not None)
        if not _stitch_videos(paths, frame_counts, total_frames, stitch_audio,
                              final_path, fps):
            return (None,)
        _LOG.info("h3_motion_context: sparsely stitched clips %s for chain %s",
                  ", ".join(str(index) for index in clip_indices), chain_id)
        return (InputImpl.VideoFromFile(final_path),)


def _current_prompt():
    running = server.PromptServer.instance.prompt_queue.currently_running
    if len(running) != 1:
        raise RuntimeError("h3_motion_context: automatic chaining requires "
                           "one active prompt")
    return next(iter(running.values()))


def _current_graph():
    value = _current_prompt()
    return value[2]


def _has_connected_input(node, name):
    value = node.get("inputs", {}).get(name)
    return isinstance(value, (list, tuple)) and len(value) >= 2


def _chain_context_enabled(current):
    """Whether the automatic Motion Context node will pin a prior clip."""
    for node in current.values():
        if node.get("class_type") != "MiniMaxH3AutoChainMotionContext":
            continue
        if (_has_connected_input(node, "context_latent") or
                _has_connected_input(node, "context_frames")):
            return True
    return False


def _source_node(current, connection):
    """Resolve a direct graph connection, following plain reroutes."""
    seen = set()
    while isinstance(connection, (list, tuple)) and len(connection) >= 2:
        node_id = connection[0]
        if node_id in seen:
            return None
        seen.add(node_id)
        node = current.get(str(node_id)) or current.get(node_id)
        if node is None or node.get("class_type") != "Reroute":
            return node
        connection = node.get("inputs", {}).get("")
    return None


def _chain_reference_fps(current):
    """Return the known rate of the IMAGE batch wired into Auto Split."""
    for node in current.values():
        if node.get("class_type") != "MiniMaxH3AutoChainAudio":
            continue
        source = _source_node(current, node.get("inputs", {}).get("ref_video"))
        if source is None or source.get("class_type") != "VHS_LoadVideo":
            continue
        rate = source.get("inputs", {}).get("force_rate", 0)
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return None
        return rate if rate > 0 else None
    return None


def _apply_chain_fps(current, fps):
    """Make the Create Video node consuming the chain trim use chain FPS."""
    for node in current.values():
        if node.get("class_type") != "CreateVideo":
            continue
        source = _source_node(current, node.get("inputs", {}).get("images"))
        if (source is not None and source.get("class_type") ==
                "MiniMaxH3AutoChainMotionContextTrim"):
            node.setdefault("inputs", {})["fps"] = float(fps)


def _describe_image_input(current, connection, reference_frame_path=None,
                          reference_mode="last_frame"):
    if not connection:
        return "not connected"
    source_id = str(connection[0])
    source = current.get(source_id) or current.get(connection[0])
    if source is None:
        return "node %s output %s" % (source_id, connection[1])
    class_type = source.get("class_type", "unknown node")
    if class_type == "MiniMaxH3AutoChainFrameReference":
        if reference_mode == "original":
            return "original image on every clip"
        if reference_frame_path:
            return "previous clip final frame: %s" % reference_frame_path
        return "initial image through H3 Auto Chain Frame Reference"
    image_name = source.get("inputs", {}).get("image")
    if image_name:
        return "%s (%s)" % (image_name, class_type)
    return "node %s output %s (%s)" % (source_id, connection[1], class_type)


def _apply_reference_mode(current, clip):
    mode = "last_frame"
    for node in current.values():
        if node.get("class_type") == "MiniMaxH3AutoChainFrameReference":
            mode = node.get("inputs", {}).get("reference_mode", mode)
            break
    if mode != "off" or int(clip) <= 1:
        return mode
    for node in current.values():
        if node.get("class_type") == "MiniMaxH3ReferenceToVideo":
            node["inputs"].pop("ref_images.ref_image_0", None)
    return mode


def _apply_reference_video(current, start_frame, end_frame, fps):
    """Keep a directly wired H3 reference video on the chain timeline."""
    for node_id, node in list(current.items()):
        if node.get("class_type") != "MiniMaxH3ReferenceToVideo":
            continue
        key = "ref_videos.ref_video_0"
        connection = node.get("inputs", {}).get(key)
        if not isinstance(connection, (list, tuple)) or len(connection) < 2:
            continue
        source = current.get(str(connection[0])) or current.get(connection[0])
        source_type = source.get("class_type") if source else None
        if source_type == "MiniMaxH3AutoChainAudio":
            continue
        source_fps = None
        if source_type == "VHS_LoadVideo":
            try:
                source_fps = float(source.get("inputs", {}).get(
                    "force_rate", 0))
            except (TypeError, ValueError):
                source_fps = None
        adapter_id = "__h3_auto_ref_video_%s" % node_id
        if source_type != "MiniMaxH3AutoChainReferenceVideo":
            current[adapter_id] = {
                "class_type": "MiniMaxH3AutoChainReferenceVideo",
                "inputs": {
                    "image": list(connection),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "fps": float(fps),
                    "source_fps": source_fps or float(fps),
                },
            }
            node["inputs"][key] = [adapter_id, 0]
        else:
            current[adapter_id]["inputs"]["start_frame"] = int(start_frame)
            current[adapter_id]["inputs"]["end_frame"] = int(end_frame)
            current[adapter_id]["inputs"]["fps"] = float(fps)
            if source_fps:
                current[adapter_id]["inputs"]["source_fps"] = source_fps
    return current


def _log_clip_inputs(current, clip, start, end, sample_rate, prompt,
                     reference_frame_path):
    reference_mode = next(
        (node.get("inputs", {}).get("reference_mode", "last_frame")
         for node in current.values()
         if node.get("class_type") == "MiniMaxH3AutoChainFrameReference"),
        "manual/static")
    for node_id, node in current.items():
        if node.get("class_type") != "MiniMaxH3ReferenceToVideo":
            continue
        ref0 = _describe_image_input(
            current, node.get("inputs", {}).get("ref_images.ref_image_0"),
            reference_frame_path, reference_mode)
        ref1 = _describe_image_input(
            current, node.get("inputs", {}).get("ref_images.ref_image_1"))
        _LOG.info(
            "h3_motion_context: clip %d audio samples %d..%d (%.3f..%.3f s, "
            "%.3f s) reference_mode=%s ref_image_0=%s ref_image_1=%s\n"
            "FULL PROMPT:\n%s",
            clip, start, end, start / sample_rate, end / sample_rate,
            (end - start) / sample_rate, reference_mode, ref0, ref1, prompt)
        return
    _LOG.info(
        "h3_motion_context: clip %d audio samples %d..%d (%.3f..%.3f s, "
        "%.3f s) H3 Reference To Video node not found\nFULL PROMPT:\n%s",
        clip, start, end, start / sample_rate, end / sample_rate,
        (end - start) / sample_rate, prompt)


def _requeue():
    value = _current_prompt()
    if len(value) == 6:
        _, _, current, extra_data, outputs_to_execute, sensitive = value
    else:
        _, _, current, extra_data, outputs_to_execute = value
        sensitive = {}
    current = copy.deepcopy(current)
    for node in current.values():
        if node.get("class_type") == "MiniMaxH3AutoChainAudio":
            node["inputs"]["reset"] = False
            node.pop("is_changed", None)
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(uuid.uuid4())
    server.PromptServer.instance.prompt_queue.put(
        (number, prompt_id, current, extra_data, outputs_to_execute, sensitive))


def _reference_video_seconds(ref_video, fps):
    if ref_video is None:
        return 0.0
    if isinstance(ref_video, torch.Tensor):
        return int(ref_video.shape[0]) / float(fps)
    return int(ref_video.get_frame_count()) / float(ref_video.get_frame_rate())


def _chain_state(chain_id, audio, ref_video, reference_fps, chunk_seconds,
                 fps, trim_frames, final_tail_mode, final_tail_frames, reset,
                 style_prompt, clip_prompts, start_clip, end_clip):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    audio_samples = int(waveform.shape[-1])
    audio_seconds = audio_samples / float(sample_rate)
    fps = float(fps)
    reference_seconds = _reference_video_seconds(
        ref_video, reference_fps or fps)
    total_frames = max(1, int(math.ceil(
        max(audio_seconds, reference_seconds) * fps)))
    total_samples = _frame_sample_index(total_frames, sample_rate, fps)
    total_seconds = total_frames / fps
    chunk_frames = _h3_chain_chunk_frames(chunk_seconds, fps)
    requested_context_frames = max(0, int(trim_frames))
    context_frames = _h3_context_frames(requested_context_frames)
    if context_frames != requested_context_frames:
        _LOG.info(
            "h3_motion_context: adjusted context frames from %d to %d "
            "for the H3 video VAE grid",
            requested_context_frames, context_frames)
    with _LOCK:
        state = _CHAINS.get(chain_id)
        if reset or state is None:
            if total_samples > audio_samples:
                _LOG.info(
                    "h3_motion_context: Auto Split timeline is %d frames "
                    "(%.6fs): reference video is longer than audio, so %d "
                    "silent source samples were added at the end",
                    total_frames, total_seconds, total_samples - audio_samples)
            else:
                _LOG.info(
                    "h3_motion_context: Auto Split timeline is %d frames "
                    "(%.6fs): audio %.6fs, reference video %.6fs",
                    total_frames, total_seconds, audio_seconds,
                    reference_seconds)
            first_clip = max(1, int(start_clip))
            previous_frame = None
            existing_videos = []
            if first_clip > 1:
                candidate = _output_path(
                    "video/%s" % _run_name(chain_id),
                    "_clip_%03d_last.png" % (first_clip - 1))
                if os.path.isfile(candidate):
                    previous_frame = candidate
                for previous_clip in range(1, first_clip):
                    previous_video = _output_path(
                        "video/%s" % _run_name(chain_id),
                        "_clip_%03d.mp4" % previous_clip)
                    if os.path.isfile(previous_video):
                        existing_videos.append(previous_video)
                    else:
                        _LOG.warning(
                            "h3_motion_context: clip %d is missing; "
                            "continuing with sparse-chain mode", previous_clip)
            state = {
                "clip": first_clip,
                "end_clip": max(0, int(end_clip)),
                "total_seconds": total_seconds,
                "audio_seconds": audio_seconds,
                "reference_seconds": reference_seconds,
                "sample_rate": sample_rate,
                "total_samples": total_samples,
                "fps": fps,
                "chunk_frames": chunk_frames,
                "trim_frames": context_frames,
                "final_tail_mode": final_tail_mode,
                "final_tail_frames": max(0, int(final_tail_frames)),
                "total_frames": total_frames,
                "videos": existing_videos,
                "frame_counts": [
                    min(end - start, max(0, total_frames - start))
                    for clip in range(1, first_clip)
                    for start, end in [_chain_frame_range(clip, chunk_frames)]
                ],
                "style_prompt": style_prompt,
                "clip_prompts": clip_prompts,
                "reset": bool(reset),
                "reference_frame_path": previous_frame,
            }
            _CHAINS[chain_id] = state
        return state


class MiniMaxH3AutoChainAudio:
    """Feed one sequential 20-second audio chunk into REF2VA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {
                "tooltip": "Complete audio for the entire automatic chain. "
                           "The node outputs one sequential chunk per run."}),
            "chain_id": ("STRING", {
                "default": "h3_auto_chain",
                "tooltip": "Unique name for this chain. It identifies the "
                           "saved latents, clips, and final video."}),
            "chunk_seconds": ("FLOAT", {"default": 20.0, "min": 1.0,
                                           "max": 600.0, "step": 0.1,
                                           "tooltip": "Target duration of each "
                                                      "clip. The final chunk "
                                                      "may be shorter."}),
            "fps": ("INT", {"default": int(DEFAULT_FPS), "min": 1,
                                "max": 240,
                                  "tooltip": "Frame rate of the chain "
                                             "timeline. Audio cuts, video "
                                             "windows, trimming, and stitched "
                                             "clips use this same frame grid."}),
            "trim_frames": ("INT", {"default": 22, "min": 0, "max": 4096,
                                      "tooltip": "H3 video-context frames. "
                                                 "Stored in chain_config and "
                                                 "used by Motion Context and "
                                                 "Motion Context Trim. Values "
                                                 "snap down to the H3 VAE grid "
                                                 "(22, 39, 56...)."}),
            "final_tail_mode": (["exact audio duration", "audio plus tail"], {
                "default": "exact audio duration",
                "tooltip": "Final partial clip output policy. Exact audio "
                           "duration removes the silent tail. Audio plus "
                           "tail keeps the configured silent tail after the "
                           "source audio ends."}),
            "final_tail_frames": ("INT", {"default": 24, "min": 0,
                                            "max": 4096,
                                            "tooltip": "Silent frames retained "
                                                       "after the source audio "
                                                       "ends when audio plus "
                                                       "tail is selected."}),
            "reset": ("BOOLEAN", {
                "default": True,
                "tooltip": "Start a new chain and discard the in-memory "
                           "position. Enable this for a new run or resume."}),
            "style_prompt": ("STRING", {
                "default": "", "multiline": True,
                "tooltip": "Shared style, character, lighting, camera, and "
                           "appearance text added to every clip prompt."}),
            "clip_prompts": ("STRING", {
                "default": "", "multiline": True,
                "tooltip": "Write one tagged prompt per clip: [1] prompt "
                           "for clip 1, [2] prompt for clip 2, and so on. "
                           "Tags must start a line."}),
            "start_clip": ("INT", {
                "default": 1, "min": 1, "max": 9999,
                "tooltip": "First clip to process. To resume after clips "
                           "1 and 2, set this to 3."}),
            "end_clip": ("INT", {"default": 0, "min": 0, "max": 9999,
                                   "tooltip": "Last clip to process. 0 means "
                                              "continue until the audio ends."}),
            "endless_continuation": ("BOOLEAN", {
                "default": False,
                "tooltip": "Use the previous clip's synchronized video/audio "
                           "tail as an H3 continuation reference, like HR "
                           "Endless Sampler. Off preserves the current chain."}),
        }, "optional": {
            "ref_video": ("IMAGE", {
                "tooltip": "Optional VHS reference video IMAGE batch. Its "
                           "known VHS force rate is resampled onto the chain "
                           "timeline; the longer of this video and audio sets "
                           "the complete, whole-frame chain length."}),
        }}

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "STRING", "STRING", "H3_CHAIN",
                    "AUDIO", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("audio", "chunk_seconds", "clip_index", "chain_id",
                    "prompt", "chain_config", "source_audio", "ref_video",
                    "frames", "fps")
    FUNCTION = "chunk"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Builds a whole-frame chain timeline from the longer of "
                   "the source audio and reference video, then outputs one "
                   "sequential H3 chunk per run.")

    @classmethod
    def IS_CHANGED(cls, audio=None, chain_id="h3_auto_chain",
                   chunk_seconds=20.0, fps=DEFAULT_FPS,
                   trim_frames=22, final_tail_mode="exact audio duration",
                   final_tail_frames=24, reset=False,
                   style_prompt="", clip_prompts="",
                   start_clip=1, end_clip=0, endless_continuation=False,
                   ref_video=None):
        if reset or audio is None:
            return float("NaN")
        with _LOCK:
            state = _CHAINS.get(chain_id)
            if state is None:
                return 1
            return (int(state["clip"]), float(chunk_seconds), float(fps),
                    int(trim_frames), str(final_tail_mode),
                    int(final_tail_frames), bool(endless_continuation))

    def chunk(self, audio, chain_id, chunk_seconds, fps=DEFAULT_FPS,
              trim_frames=22, final_tail_mode="exact audio duration",
              final_tail_frames=24, reset=False,
              style_prompt="", clip_prompts="",
              start_clip=1, end_clip=0, endless_continuation=False,
              ref_video=None):
        current = _current_graph()
        reference_fps = _chain_reference_fps(current) or float(fps)
        state = _chain_state(
            chain_id, audio, ref_video, reference_fps,
            float(chunk_seconds), float(fps), int(trim_frames),
            final_tail_mode, int(final_tail_frames), reset,
            style_prompt, clip_prompts, start_clip, end_clip)
        _apply_chain_fps(current, state["fps"])
        _apply_reference_mode(current, state["clip"])
        for node in current.values():
            if node.get("class_type") == "MiniMaxH3AutoChainMotionContext":
                node.setdefault("inputs", {})["endless_continuation"] = bool(
                    endless_continuation)
        clip = int(state["clip"])
        prompt = _prompt_for_clip(state["style_prompt"],
                                  state["clip_prompts"], clip)
        start_frame, requested_end_frame = _chain_frame_range(
            clip, state["chunk_frames"])
        end_frame = min(state["total_frames"], requested_end_frame)
        source_finished = end_frame >= state["total_frames"]
        partial_final = source_finished and end_frame < requested_end_frame
        load_clip_index = max(0, clip - 1)
        context_available = clip > 1 and _chain_context_enabled(current)
        if context_available and load_clip_index:
            expected_latent = _output_path(
                "h3_context/%s_clip" % _run_name(chain_id),
                "_%05d.safetensors" % load_clip_index)
            if not os.path.isfile(expected_latent):
                _LOG.warning(
                    "h3_motion_context: clip %d has no previous latent at %s; "
                    "Motion Context will reject the reserved audio pre-roll",
                    clip, expected_latent)
        effective_trim = state["trim_frames"] if context_available else 0
        output_tail = 0
        if partial_final and state["final_tail_mode"] == "audio plus tail":
            output_tail = state["final_tail_frames"]
        generation_end_frame = start_frame + (end_frame - start_frame)
        if clip > 1:
            generation_end_frame += effective_trim
        if partial_final:
            generation_end_frame += output_tail
        output_end_frame = end_frame + output_tail
        # The continuation head is pinned from the previous latent and then
        # removed by Motion Context Trim. Feed that same frame span as audio
        # pre-roll so the delivered clip still starts at source start_frame.
        audio_start_frame = max(0, start_frame - effective_trim)
        start = _frame_sample_index(audio_start_frame,
                                    state["sample_rate"], state["fps"])
        requested_audio_end_frame = end_frame + output_tail
        audio_frame_count = align_frame_count(
            max(5, requested_audio_end_frame - audio_start_frame))
        audio_end_frame = audio_start_frame + audio_frame_count
        _apply_reference_video(_current_graph(), audio_start_frame,
                               audio_end_frame, state["fps"])
        end = _frame_sample_index(audio_end_frame,
                                  state["sample_rate"], state["fps"])
        if start >= end:
            raise RuntimeError("h3_motion_context: chain has already finished; "
                               "enable reset for a new run")
        source_waveform = audio["waveform"].detach().cpu()
        input_samples = int(source_waveform.shape[-1])
        source_waveform = source_waveform[..., :state["total_samples"]]
        if input_samples < state["total_samples"]:
            source_waveform = torch.nn.functional.pad(
                source_waveform, (0, state["total_samples"] - input_samples))
        chunk_waveform = source_waveform[..., start:min(
            end, state["total_samples"])].contiguous()
        if end > state["total_samples"]:
            chunk_waveform = torch.nn.functional.pad(
                chunk_waveform, (0, end - state["total_samples"]))
        seconds = (end - start) / float(state["sample_rate"])
        source_seconds = (end_frame - start_frame) / state["fps"]
        silent_samples = max(0, min(end, state["total_samples"]) -
                             min(max(start, input_samples), state["total_samples"]))
        chain_config = {
            "chain_id": chain_id,
            "run_name": _run_name(chain_id),
            "latent_prefix": "h3_context/%s_clip" % _run_name(chain_id),
            "chunk_seconds": float(chunk_seconds),
            "fps": state["fps"],
            "chunk_frames": state["chunk_frames"],
            "context_frames": state["trim_frames"],
            "total_frames": state["total_frames"],
            "source_audio_frames": max(1, int(math.ceil(
                input_samples / state["sample_rate"] * state["fps"]))),
            "source_reference_frames": max(1, int(math.ceil(
                state["reference_seconds"] * state["fps"]))) if ref_video is not None else 0,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "clip_frames": output_end_frame - start_frame,
            "generation_frames": generation_end_frame - start_frame,
            "source_frames": end_frame - start_frame,
            "silent_padded_frames": max(0, generation_end_frame - end_frame),
            "final_tail_frames": output_tail,
            "final_tail_mode": state["final_tail_mode"],
            "effective_trim_frames": effective_trim,
            "context_latent_available": context_available,
            "audio_start_frame": audio_start_frame,
            "audio_end_frame": audio_end_frame,
            "audio_start_sample": start,
            "audio_end_sample": end,
            "source_seconds": source_seconds,
            "silent_padded_seconds": silent_samples / state["sample_rate"],
            "model_generation_seconds": seconds,
            "actual_seconds": seconds,
            "clip_index": clip,
            "load_clip_index": load_clip_index,
            "save_clip_index": clip,
            "start_clip": int(start_clip),
            "end_clip": int(end_clip),
            "reset": bool(state["reset"]),
            "reference_frame_path": state.get("reference_frame_path"),
            "endless_continuation": bool(endless_continuation),
        }
        state["reset"] = False
        _LOG.info(
            "h3_motion_context: clip %d source frames %d..%d (%d), "
            "model frames %d..%d (%d), source %.6fs, silent pad %.6fs, "
            "output %d frames, trim %d "
            "audio samples %d..%d (%d samples)",
            clip, start_frame, end_frame, end_frame - start_frame,
            start_frame, generation_end_frame,
            generation_end_frame - start_frame, source_seconds,
            silent_samples / state["sample_rate"], output_end_frame - start_frame,
            effective_trim, start, end, end - start)
        _log_clip_inputs(_current_graph(), clip, start, end,
                         state["sample_rate"], prompt,
                         state.get("reference_frame_path"))
        ref_video_chunk = _reference_video_window(
            ref_video, chain_config["audio_start_frame"],
            chain_config["audio_end_frame"], chain_config["fps"],
            chain_config["clip_index"], _chain_reference_fps(current))
        return ({"waveform": chunk_waveform, "sample_rate": state["sample_rate"]},
                seconds, clip, chain_id, prompt, chain_config,
                {"waveform": source_waveform,
                 "sample_rate": state["sample_rate"]},
                ref_video_chunk, audio_frame_count, state["fps"])


def _reference_image_window(images, start_frame, end_frame, fps,
                            source_fps=None):
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    source_fps = float(source_fps) if source_fps else float(fps)
    if math.isclose(source_fps, float(fps), rel_tol=0.0, abs_tol=1e-6):
        return images[start:min(end, int(images.shape[0]))]
    positions = torch.arange(start, end, device=images.device)
    positions = torch.floor(positions * source_fps / float(fps)).long()
    positions = positions[positions < int(images.shape[0])]
    if positions.numel() == 0:
        return images[:0]
    _LOG.info(
        "h3_motion_context: resampled reference frames from %.6f FPS to "
        "%.6f FPS for the chain timeline", source_fps, float(fps))
    return images[positions]


def _reference_video_window(video, start_frame, end_frame, fps, clip,
                            source_fps=None):
    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()
        frame_count = max(0, int(end_frame) - int(start_frame))
        if frame_count < 5:
            _LOG.warning(
                "h3_motion_context: reference video has fewer than 5 frames "
                "for clip %d; skipping its reference video", clip)
            return None
        frames = _reference_image_window(video, start_frame, end_frame, fps,
                                         source_fps)
        available_end = int(start_frame) + int(frames.shape[0])
        if int(frames.shape[0]) < frame_count:
            _LOG.warning(
                "h3_motion_context: reference video ends at frame %d; "
                "using frames %d..%d for clip %d",
                int(video.shape[0]), int(start_frame), available_end, clip)
        frame_count = int(frames.shape[0])
        if frame_count < 5:
            _LOG.warning(
                "h3_motion_context: reference video has only %d available "
                "frames for clip %d; skipping its reference video",
                frame_count, clip)
            return None
        _log_reference_video_window(frames, int(start_frame), available_end,
                                    clip)
        width, height = int(frames.shape[2]), int(frames.shape[1])
        target_width, target_height = adapt_canvas(width, height)
        if width * height < target_width * target_height:
            target_width = max(32, round(width / 32) * 32)
            target_height = max(32, round(height / 32) * 32)
        frames = _h3_resize(frames, target_width, target_height, "disabled")
        return frames.contiguous()
    source_fps = float(video.get_frame_rate())
    if not math.isclose(source_fps, float(fps), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "h3_motion_context: reference video FPS %.6f does not match "
            "chain FPS %.6f for clip %d" % (source_fps, fps, clip))
    frame_count = max(0, int(end_frame) - int(start_frame))
    if frame_count < 5:
        _LOG.warning(
            "h3_motion_context: reference video has fewer than 5 frames "
            "for clip %d; skipping its reference video", clip)
        return None
    trimmed = video.as_trimmed(
        float(start_frame) / float(fps),
        float(frame_count) / float(fps),
        strict_duration=False)
    if trimmed is None:
        _LOG.warning(
            "h3_motion_context: reference video has no frames for clip %d; "
            "skipping its reference video", clip)
        return None
    frames = trimmed.get_components().images.detach().cpu()
    if frames.shape[0] == 0:
        _LOG.warning(
            "h3_motion_context: reference video has no frames in the "
            "audio-matched window for clip %d; skipping its reference video",
            clip)
        return None
    if frames.shape[0] < 5:
        _LOG.warning(
            "h3_motion_context: reference video has only %d available "
            "frames for clip %d; skipping its reference video",
            frames.shape[0], clip)
        return None
    actual_end = int(start_frame) + int(frames.shape[0])
    if frames.shape[0] < frame_count:
        _LOG.warning(
            "h3_motion_context: reference video returned %d of %d requested "
            "frames for clip %d; using the available frames",
            frames.shape[0], frame_count, clip)
    frames = frames[:frame_count]
    _log_reference_video_window(
        frames, int(start_frame), actual_end, clip)
    width, height = int(frames.shape[2]), int(frames.shape[1])
    target_width, target_height = adapt_canvas(width, height)
    if width * height < target_width * target_height:
        target_width = max(32, round(width / 32) * 32)
        target_height = max(32, round(height / 32) * 32)
    frames = _h3_resize(frames, target_width, target_height, "disabled")
    return frames.contiguous()


def _log_reference_video_window(frames, start_frame, end_frame, clip):
    first = hashlib.sha1(
        frames[0].contiguous().numpy().tobytes()).hexdigest()[:12]
    last = hashlib.sha1(
        frames[-1].contiguous().numpy().tobytes()).hexdigest()[:12]
    _LOG.info(
        "h3_motion_context: clip %d reference frames %d..%d (%d) "
        "first=%s last=%s",
        clip, start_frame, end_frame, end_frame - start_frame, first, last)


def _with_chain_fps(video, fps):
    source_fps = float(video.get_frame_rate())
    if math.isclose(source_fps, float(fps), rel_tol=0.0, abs_tol=1e-6):
        return video
    components = video.get_components()
    _LOG.info(
        "h3_motion_context: using chain FPS %.6f for the supplied %d-frame "
        "video instead of Create Video's %.6f FPS",
        float(fps), int(components.images.shape[0]), source_fps)
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=components.images,
            alpha=components.alpha,
            audio=components.audio,
            frame_rate=Fraction(str(float(fps))),
            metadata=components.metadata,
        ),
        bit_depth=video.get_bit_depth(), color_space=video.get_color_space())


def _match_audio_duration(audio, clip_frames, fps):
    sample_rate = int(audio["sample_rate"])
    waveform = audio["waveform"]
    actual_samples = int(waveform.shape[-1])
    expected_samples = _frame_sample_index(clip_frames, sample_rate, fps)
    if actual_samples == expected_samples:
        return audio
    if actual_samples > expected_samples:
        waveform = waveform[..., :expected_samples]
        action = "trimmed"
    else:
        waveform = torch.nn.functional.pad(
            waveform, (0, expected_samples - actual_samples))
        action = "padded"
    _LOG.info(
        "h3_motion_context: %s generated-audio tail from %d to %d samples "
        "so it matches %d frames at %.6f FPS",
        action, actual_samples, expected_samples, int(clip_frames), float(fps))
    audio = dict(audio)
    audio["waveform"] = waveform
    return audio


def _with_chain_audio_duration(video, clip_frames, fps):
    components = video.get_components()
    if components.audio is None:
        raise ValueError(
            "h3_motion_context: generated video audio is selected, but the "
            "video has no embedded audio after Motion Context Trim")
    audio = _match_audio_duration(components.audio, clip_frames, fps)
    if audio is components.audio:
        return video
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=components.images,
            alpha=components.alpha,
            audio=audio,
            frame_rate=components.frame_rate,
            metadata=components.metadata,
        ),
        bit_depth=video.get_bit_depth(), color_space=video.get_color_space())


def _validate_chain_clip(video, clip_frames, require_audio):
    components = video.get_components()
    actual_frames = int(components.images.shape[0])
    if actual_frames != int(clip_frames):
        raise ValueError(
            "h3_motion_context: clip has %d decoded frames but Auto Split "
            "planned %d. Check the Motion Context Trim wiring; no frames "
            "were written."
            % (actual_frames, int(clip_frames)))
    if not require_audio:
        return
    audio = components.audio
    if audio is None:
        raise ValueError(
            "h3_motion_context: generated video audio is selected, but the "
            "video has no embedded audio after Motion Context Trim")


class MiniMaxH3AutoChain:
    """Requeue the current ComfyUI graph until all H3 audio chunks finish."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "chain_config": ("H3_CHAIN", {
                "tooltip": "Connect the chain_config output from H3 Auto Chain Audio. "
                           "This supplies the chain ID, clip timing, and latent slots."}),
            "audio_source": (["original audio input", "generated video audio"], {
                "default": "original audio input",
                "tooltip": "Original audio input replaces the audio in the "
                           "generated clips at frame-aligned boundaries. "
                           "Generated video audio keeps the audio already "
                           "embedded in each generated video clip."}),
            "save_mode": ([
                "save each clip separately + composite final video",
                "save each clip separately + partial and final composite videos",
                "save only each single clip",
                "save each clip separately + one partial composite (replace previous)",
                "save each clip separately + one final composite (replace previous)",
            ], {
                "default": "save each clip separately + composite final video",
                "tooltip": "Individual clip files are always saved. Choose "
                           "whether to also save the final composite, partial "
                           "composites, or no composite video."}),
            "delete_completed_latents": ("BOOLEAN", {
                "default": False,
                "tooltip": "Delete this chain's saved H3 latent files after "
                           "the final MP4 is stitched successfully. Disable "
                           "to keep them available for retry or resume."}),
        }, "optional": {
            "audio": ("AUDIO", {
                "tooltip": "The complete original audio. When connected, the "
                           "final MP4 uses this source at frame-aligned clip "
                           "boundaries instead of reusing per-clip encoded audio."}),
        }}

    RETURN_TYPES = ("VIDEO", "VIDEO", "VIDEO")
    RETURN_NAMES = ("current video", "single clip", "combined video")
    FUNCTION = "advance"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Automatically requeues the next H3 chunk, advances the "
                   "Motion Context latent slots, and stitches all clips into "
                   "one MP4 when the audio is complete.")

    def advance(self, video, chain_config, audio_source="original",
                save_mode="save each clip separately + composite final video",
                delete_completed_latents=False, audio=None):
        combined_path = None
        use_original_audio = audio_source in ("original", "original audio input")
        use_generated_audio = audio_source in ("generated", "generated video audio")
        if use_original_audio and audio is None:
            _LOG.warning(
                "h3_motion_context: original audio input is not connected; "
                "using generated video audio")
            use_original_audio = False
            use_generated_audio = True
        if not use_original_audio and not use_generated_audio:
            raise ValueError("h3_motion_context: unknown audio source %r" %
                             audio_source)
        chain_id = str(chain_config["chain_id"])
        fps = float(chain_config["fps"])
        clip_frames = int(chain_config["clip_frames"])
        video = _with_chain_fps(video, fps)
        if use_generated_audio:
            video = _with_chain_audio_duration(video, clip_frames, fps)
        _validate_chain_clip(video, clip_frames, use_generated_audio)
        output_prefix = "video/%s" % chain_config["run_name"]
        with _LOCK:
            state = _CHAINS.get(chain_id)
            if state is None:
                raise RuntimeError("h3_motion_context: connect the matching "
                                   "Auto Chain Audio node")
            clip = int(state["clip"])
            next_start_frame = _chain_frame_range(
                clip + 1, state["chunk_frames"])[0]
            finished = next_start_frame >= state["total_frames"]
            if state.get("end_clip", 0):
                finished = finished or clip >= int(state["end_clip"])
            if not finished:
                state["clip"] = clip + 1
            clip_path = _output_path(output_prefix, "_clip_%03d.mp4" % clip)
            state["videos"].append(clip_path)
            state["frame_counts"].append(int(chain_config["clip_frames"]))
        clip_path = os.path.abspath(clip_path)
        os.makedirs(os.path.dirname(clip_path), exist_ok=True)
        raw_path = clip_path + ".raw.mp4"
        temp_path = clip_path + ".tmp.mp4"
        for path in (raw_path, temp_path):
            if os.path.exists(path):
                os.remove(path)
        clip_duration = clip_frames / fps
        video_filter = (
            "fps=%.6f,trim=end_frame=%d,setpts=N/(%.6f*TB)"
            % (fps, clip_frames, fps))
        try:
            video.save_to(raw_path, format="mp4", codec="h264")
            args = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel",
                "error", "-y", "-i", raw_path,
                "-map", "0:v:0", "-map", "0:a:0?",
                "-frames:v", str(clip_frames),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-r", "%.6f" % fps,
            ]
            if use_generated_audio:
                args.extend([
                    "-vf", video_filter,
                    "-c:a", "copy",
                ])
            else:
                args.extend([
                    "-vf", video_filter,
                    "-af", "atrim=start_sample=0,apad,atrim=duration=%.9f,"
                           "asetpts=PTS-STARTPTS" % (clip_frames / fps),
                    "-c:a", "aac", "-b:a", "192k",
                ])
            args.append(temp_path)
            subprocess.run(args, check=True)
            _replace_output_file(temp_path, clip_path)
        except BaseException:
            for path in (raw_path, temp_path):
                if os.path.exists(path):
                    os.remove(path)
            raise
        if os.path.exists(raw_path):
            os.remove(raw_path)
        _LOG.info("h3_motion_context: clip %d trimmed to %.3f seconds",
                  clip, clip_duration)
        frame_path = _output_path(
            output_prefix, "_clip_%03d_last.png" % clip)
        frame_temp = frame_path + ".tmp.png"
        if os.path.exists(frame_temp):
            os.remove(frame_temp)
        try:
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel",
                "error", "-y", "-sseof", "-0.2", "-i", clip_path,
                "-frames:v", "1", frame_temp,
            ], check=True)
            _replace_output_file(frame_temp, frame_path)
        except BaseException:
            if os.path.exists(frame_temp):
                os.remove(frame_temp)
            raise
        with _LOCK:
            state["reference_frame_path"] = frame_path
        _LOG.info("h3_motion_context: saved last-frame reference %s", frame_path)
        save_partial = save_mode in (
            "save each clip separately + partial and final composite videos",
            "save partial clip on each step",
            "save each clip separately + one partial composite (replace previous)",
        )
        save_final = save_mode in (
            "save each clip separately + composite final video",
            "save each clip separately + partial and final composite videos",
            "save full clip if finished",
            "save partial clip on each step",
            "save each clip separately + one final composite (replace previous)",
        )
        replace_composite = save_mode in (
            "save each clip separately + one partial composite (replace previous)",
            "save each clip separately + one final composite (replace previous)",
        )
        if save_partial:
            partial_path = _output_path(output_prefix, "-partial.mp4")
            if replace_composite:
                for combined_path in (partial_path,
                                      _output_path(output_prefix, "-final.mp4")):
                    if os.path.isfile(combined_path):
                        os.remove(combined_path)
            paths, frame_counts, clip_indices = _available_chain_clips(
                chain_config["run_name"], state["total_frames"],
                state["chunk_frames"])
            stitch_frames = sum(frame_counts)
            stitch_audio = None
            if use_original_audio:
                stitch_audio = _audio_for_clips(
                    audio, clip_indices, state["chunk_frames"], fps)
            if _stitch_videos(paths, frame_counts, stitch_frames, stitch_audio,
                              partial_path, fps):
                combined_path = partial_path
                _LOG.info("h3_motion_context: saved partial stitch through clip %d",
                          clip)
        if not finished:
            current = _current_graph()
            load_node_id = next((node_id for node_id, node in current.items()
                                 if node.get("class_type") ==
                                 "MiniMaxH3AutoChainLoadLatent"), "")
            save_node_id = next((node_id for node_id, node in current.items()
                                 if node.get("class_type") ==
                                 "MiniMaxH3AutoChainSaveLatent"), "")
            if not load_node_id or not save_node_id:
                raise RuntimeError("h3_motion_context: automatic chain needs "
                                   "one Load Latent and one Save Latent node")
            _requeue()
            _LOG.info("h3_motion_context: queued automatic clip %d", clip + 1)
        elif save_final:
            final_path = _output_path(output_prefix, "-final.mp4")
            if replace_composite:
                for combined_path in (final_path,
                                      _output_path(output_prefix, "-partial.mp4")):
                    if os.path.isfile(combined_path):
                        os.remove(combined_path)
            paths, frame_counts, clip_indices = _available_chain_clips(
                chain_config["run_name"], state["total_frames"],
                state["chunk_frames"])
            if not paths:
                raise RuntimeError("h3_motion_context: no completed clips are "
                                   "available for stitching")
            stitch_frames = sum(frame_counts)
            stitch_audio = None
            if use_original_audio:
                stitch_audio = _audio_for_clips(
                    audio, clip_indices, state["chunk_frames"], fps)
            if _stitch_videos(paths, frame_counts, stitch_frames, stitch_audio,
                              final_path, fps):
                combined_path = final_path
            if len(paths) < _chain_clip_count(state["total_frames"],
                                              state["chunk_frames"]):
                _LOG.warning(
                    "h3_motion_context: stitched sparse chain with clips %s",
                    ", ".join(str(index) for index in clip_indices))
            if delete_completed_latents:
                _cleanup_chain_latents(chain_config["run_name"])
            _LOG.info("h3_motion_context: automatic chain complete at clip %d", clip)
        single_clip = InputImpl.VideoFromFile(clip_path)
        combined_video = (InputImpl.VideoFromFile(combined_path)
                          if combined_path is not None else None)
        return (video, single_clip, combined_video)


class MiniMaxH3AutoChainReferenceVideo:
    """Slice a reference frame batch to the current automatic-chain window."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "start_frame": ("INT", {"default": 0, "min": 0,
                                      "max": 1000000}),
            "end_frame": ("INT", {"default": 120, "min": 1,
                                    "max": 1000000}),
        }, "optional": {
            "fps": ("FLOAT", {"default": DEFAULT_FPS, "min": 1.0,
                                  "max": 240.0, "step": 0.001}),
            "source_fps": ("FLOAT", {"default": 0.0, "min": 0.0,
                                         "max": 240.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "slice"
    CATEGORY = "conditioning/minimax"

    def slice(self, image, start_frame, end_frame, fps=DEFAULT_FPS,
              source_fps=0.0):
        frames = _reference_image_window(image, start_frame, end_frame, fps,
                                         source_fps)
        if int(frames.shape[0]) < 5:
            return (None,)
        return (frames.contiguous(),)


class MiniMaxH3AutoChainFrameReference:
    """Use the original image for clip one and the prior clip tail after it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chain_config": ("H3_CHAIN", {
                "tooltip": "Selects the saved final frame from the previous "
                           "clip. Clip 1 uses the initial image."}),
            "initial_image": ("IMAGE", {
                "tooltip": "Reference image used for the first clip."}),
            "reference_mode": (["original", "last_frame", "off"], {
                "default": "last_frame",
                "tooltip": "original: use the initial image on every clip; "
                           "last_frame: use the previous clip final frame; "
                           "off: use the image only on clip 1, then rely on "
                           "Motion Context."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_image",)
    FUNCTION = "reference"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Keeps the H3 character reference by chaining final frames."

    @classmethod
    def IS_CHANGED(cls, chain_config=None, initial_image=None,
                   reference_mode="last_frame"):
        reference_frame_path = (
            chain_config.get("reference_frame_path")
            if isinstance(chain_config, dict) else None)
        return (reference_mode, reference_frame_path)

    def reference(self, chain_config, initial_image,
                  reference_mode="last_frame"):
        if reference_mode == "original":
            _LOG.info("h3_motion_context: using original reference image for "
                      "clip %s", chain_config.get("clip_index", "?"))
            return (initial_image,)
        if reference_mode == "off" and int(chain_config.get("clip_index", 1)) > 1:
            _LOG.info("h3_motion_context: reference image disabled for clip %s",
                      chain_config.get("clip_index", "?"))
            return (None,)
        path = chain_config.get("reference_frame_path")
        if not path:
            _LOG.info("h3_motion_context: using initial reference image for clip %s",
                      chain_config.get("clip_index", "?"))
            return (initial_image,)
        _LOG.info("h3_motion_context: using %s as ref_image_0 for clip %s",
                  path, chain_config.get("clip_index", "?"))
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AutoChainMotionContext": MiniMaxH3AutoChainMotionContext,
    "MiniMaxH3AutoChainMotionContextTrim": MiniMaxH3AutoChainMotionContextTrim,
    "MiniMaxH3AutoChainCreateVideo": MiniMaxH3AutoChainCreateVideo,
    "MiniMaxH3AutoChainSaveLatent": MiniMaxH3AutoChainSaveLatent,
    "MiniMaxH3AutoChainLoadLatent": MiniMaxH3AutoChainLoadLatent,
    "MiniMaxH3AutoChainAudio": MiniMaxH3AutoChainAudio,
    "MiniMaxH3AutoChain": MiniMaxH3AutoChain,
    "MiniMaxH3SparseChainStitch": MiniMaxH3SparseChainStitch,
    "MiniMaxH3AutoChainReferenceVideo": MiniMaxH3AutoChainReferenceVideo,
    "MiniMaxH3AutoChainFrameReference": MiniMaxH3AutoChainFrameReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AutoChainMotionContext": "H3 Auto Chain Motion Context",
    "MiniMaxH3AutoChainMotionContextTrim": "H3 Auto Chain Motion Context Trim",
    "MiniMaxH3AutoChainCreateVideo": "H3 Auto Chain Create Video",
    "MiniMaxH3AutoChainSaveLatent": "H3 Auto Chain Save Latent",
    "MiniMaxH3AutoChainLoadLatent": "H3 Auto Chain Load Latent",
    "MiniMaxH3AutoChainAudio": "H3 Auto Chain Audio",
    "MiniMaxH3AutoChain": "H3 Auto Chain + Stitch",
    "MiniMaxH3SparseChainStitch": "H3 Sparse Chain Stitch",
    "MiniMaxH3AutoChainReferenceVideo": "H3 Auto Chain Reference Video",
    "MiniMaxH3AutoChainFrameReference": "H3 Auto Chain Frame Reference",
}
