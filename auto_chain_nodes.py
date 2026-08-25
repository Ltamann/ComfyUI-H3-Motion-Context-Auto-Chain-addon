"""In-graph audio chunking and prompt requeue for H3 continuation chains."""

import copy
import logging
import math
import os
import re
import subprocess
import threading
import uuid
import wave

import folder_paths
import imageio_ffmpeg
import numpy as np
import server
import torch
from PIL import Image

from .motion_context_core import (
    AutoChainMotionContextCore,
    MiniMaxH3MotionContextTrim,
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
            "filename_prefix": ("STRING", {"default": "h3_context/clip"}),
            "clip_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
        }, "optional": {"chain_config": ("H3_CHAIN",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"

    def save(self, latent, filename_prefix, clip_index=0, chain_config=None):
        if _st_save is None:
            raise RuntimeError("h3_motion_context: safetensors is unavailable")
        if chain_config is not None:
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
            "latent_path": ("STRING", {"default": "h3_context"}),
            "clip_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            "reset": ("BOOLEAN", {"default": False}),
        }, "optional": {"chain_config": ("H3_CHAIN",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0, reset=False,
                   chain_config=None):
        if chain_config is not None:
            clip_index = chain_config["load_clip_index"]
            reset = chain_config["reset"]
        if reset:
            return float("NaN")
        try:
            if chain_config is not None and int(clip_index) > 0:
                path = _chain_latent_path(chain_config, clip_index)
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
            else:
                path = _resolve_latent_path(latent_path, clip_index)
            return "%s:%d" % (path, os.stat(path).st_mtime_ns)
        except FileNotFoundError:
            return float("NaN")

    def load(self, latent_path, clip_index=0, reset=False, chain_config=None):
        if chain_config is not None:
            clip_index = chain_config["load_clip_index"]
            reset = chain_config["reset"]
        if reset or int(clip_index) == 0:
            return (None,)
        try:
            if chain_config is not None:
                path = _chain_latent_path(chain_config, clip_index)
            else:
                path = _resolve_latent_path(latent_path, clip_index)
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
        return AutoChainMotionContextCore.INPUT_TYPES()

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"

    def apply(self, conditioning, vae, latent, context_length,
              audio_context_length=24, context_frames=None,
              context_latent=None, audio_vae=None, context_audio=None):
        if context_latent is None and context_frames is None:
            return (conditioning, 0)
        return AutoChainMotionContextCore().apply(
            conditioning, vae, latent, context_length,
            audio_context_length, context_frames, context_latent,
            audio_vae, context_audio)


class MiniMaxH3AutoChainMotionContextTrim(MiniMaxH3MotionContextTrim):
    """Standalone trim node paired with Auto Chain Motion Context."""


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


def _available_chain_clips(run_name, total_frames, chunk_frames):
    """Return existing clips in timeline order, allowing sparse chains."""
    total_clips = max(1, int(math.ceil(total_frames / float(chunk_frames))))
    paths = []
    frame_counts = []
    clip_indices = []
    for clip in range(1, total_clips + 1):
        path = _output_path("video/%s" % run_name, "_clip_%03d.mp4" % clip)
        if not os.path.isfile(path):
            continue
        paths.append(path)
        frame_counts.append(min(chunk_frames,
                                max(0, total_frames - (clip - 1) * chunk_frames)))
        clip_indices.append(clip)
    return paths, frame_counts, clip_indices


def _audio_for_clips(audio, clip_indices, chunk_frames, fps):
    """Concatenate source-audio ranges belonging to the available clips."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    pieces = []
    for clip in clip_indices:
        start = int(round((clip - 1) * chunk_frames * sample_rate / fps))
        end = int(round(clip * chunk_frames * sample_rate / fps))
        pieces.append(waveform[..., start:min(end, waveform.shape[-1])])
    if not pieces:
        return {"waveform": waveform[..., :0], "sample_rate": sample_rate}
    return {"waveform": torch.cat(pieces, dim=-1), "sample_rate": sample_rate}


def _stitch_videos(paths, frame_counts, total_frames, source_audio,
                   final_path, fps):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        args.extend(["-i", path])
    filters = []
    concat_inputs = []
    for index, path in enumerate(paths):
        frames = int(frame_counts[index])
        filters.append(
            "[%d:v:0]trim=end_frame=%d,setpts=PTS-STARTPTS[v%d]" %
            (index, frames, index))
        concat_inputs.append("[v%d]" % index)
    filters.append("%sconcat=n=%d:v=1:a=0[v]" %
                   ("".join(concat_inputs), len(paths)))
    audio_path = None
    if source_audio is not None:
        audio_path = final_path + ".source.wav"
        _write_audio_wav(source_audio, audio_path)
        args.extend(["-i", audio_path])
        audio_index = len(paths)
        total_seconds = float(total_frames) / fps
        filters.append(
            "[%d:a:0]atrim=start=0,apad,atrim=end=%.9f,"
            "asetpts=PTS-STARTPTS[a]" % (audio_index, total_seconds))
    else:
        audio_inputs = []
        for index, path in enumerate(paths):
            frames = int(frame_counts[index])
            duration = float(frames) / fps
            filters.append(
                "[%d:a:0]atrim=start=0:end=%.9f,asetpts=PTS-STARTPTS[a%d]" %
                (index, duration, index))
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


def _current_prompt():
    running = server.PromptServer.instance.prompt_queue.currently_running
    if len(running) != 1:
        raise RuntimeError("h3_motion_context: automatic chaining requires "
                           "one active prompt")
    return next(iter(running.values()))


def _current_graph():
    value = _current_prompt()
    return value[2]


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


def _requeue(load_node_id, save_node_id, previous_clip, next_clip):
    value = _current_prompt()
    if len(value) == 6:
        _, _, current, extra_data, outputs_to_execute, sensitive = value
    else:
        _, _, current, extra_data, outputs_to_execute = value
        sensitive = {}
    current = copy.deepcopy(current)
    if load_node_id in current:
        current[load_node_id]["inputs"]["clip_index"] = previous_clip
        current[load_node_id]["inputs"]["reset"] = False
    if save_node_id in current:
        current[save_node_id]["inputs"]["clip_index"] = next_clip
    for node in current.values():
        if node.get("class_type") == "MiniMaxH3AutoChainAudio":
            node["inputs"]["reset"] = False
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(uuid.uuid4())
    server.PromptServer.instance.prompt_queue.put(
        (number, prompt_id, current, extra_data, outputs_to_execute, sensitive))


def _chain_state(chain_id, audio, chunk_seconds, fps, trim_frames,
                 final_tail_mode, final_tail_frames, reset, style_prompt,
                 clip_prompts, start_clip, end_clip):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    total_samples = int(waveform.shape[-1])
    total_seconds = total_samples / float(sample_rate)
    fps = float(fps)
    chunk_frames = max(1, int(round(float(chunk_seconds) * fps)))
    total_frames = max(1, int(math.ceil(total_seconds * fps)))
    with _LOCK:
        state = _CHAINS.get(chain_id)
        if reset or state is None:
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
                "sample_rate": sample_rate,
                "total_samples": total_samples,
                "fps": fps,
                "chunk_frames": chunk_frames,
                "trim_frames": max(0, int(trim_frames)),
                "final_tail_mode": final_tail_mode,
                "final_tail_frames": max(0, int(final_tail_frames)),
                "total_frames": total_frames,
                "videos": existing_videos,
                "frame_counts": [
                    min(chunk_frames,
                        max(0, total_frames - (clip - 1) * chunk_frames))
                    for clip in range(1, first_clip)
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
            "fps": ("FLOAT", {"default": DEFAULT_FPS, "min": 1.0,
                                  "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the generated "
                                             "video. Clip starts and audio "
                                             "cuts are calculated on this "
                                             "frame grid."}),
            "trim_frames": ("INT", {"default": 22, "min": 0, "max": 4096,
                                      "tooltip": "Leading frames removed by "
                                                 "H3 Motion Context Trim from "
                                                 "continuation clips. Match "
                                                 "the Motion Context context "
                                                 "length; 22 is about one "
                                                 "second at 24 FPS."}),
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
        }}

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "STRING", "STRING", "H3_CHAIN",
                    "AUDIO")
    RETURN_NAMES = ("audio", "chunk_seconds", "clip_index", "chain_id",
                    "prompt", "chain_config", "source_audio")
    FUNCTION = "chunk"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Splits one AUDIO input into sequential chunks and reports "
                   "the chunk duration for H3 Reference To Video.")

    @classmethod
    def IS_CHANGED(cls, audio=None, chain_id="h3_auto_chain",
                   chunk_seconds=20.0, fps=DEFAULT_FPS,
                   trim_frames=22, final_tail_mode="exact audio duration",
                   final_tail_frames=24, reset=False,
                   style_prompt="", clip_prompts="",
                   start_clip=1, end_clip=0):
        if reset or audio is None:
            return float("NaN")
        with _LOCK:
            state = _CHAINS.get(chain_id)
            if state is None:
                return 1
            return (int(state["clip"]), float(chunk_seconds), float(fps),
                    int(trim_frames), str(final_tail_mode),
                    int(final_tail_frames))

    def chunk(self, audio, chain_id, chunk_seconds, fps=DEFAULT_FPS,
              trim_frames=22, final_tail_mode="exact audio duration",
              final_tail_frames=24, reset=False,
              style_prompt="", clip_prompts="",
              start_clip=1, end_clip=0):
        state = _chain_state(chain_id, audio, float(chunk_seconds), float(fps),
                             int(trim_frames), final_tail_mode,
                             int(final_tail_frames), reset,
                             style_prompt, clip_prompts,
                             start_clip, end_clip)
        _apply_reference_mode(_current_graph(), state["clip"])
        clip = int(state["clip"])
        prompt = _prompt_for_clip(state["style_prompt"],
                                  state["clip_prompts"], clip)
        start_frame = (clip - 1) * state["chunk_frames"]
        requested_end_frame = clip * state["chunk_frames"]
        end_frame = min(state["total_frames"], requested_end_frame)
        source_finished = end_frame >= state["total_frames"]
        partial_final = source_finished and end_frame < requested_end_frame
        load_clip_index = max(0, clip - 1)
        context_available = True
        if load_clip_index:
            expected_latent = _output_path(
                "h3_context/%s_clip" % _run_name(chain_id),
                "_%05d.safetensors" % load_clip_index)
            context_available = os.path.isfile(expected_latent)
            if not context_available:
                _LOG.warning(
                    "h3_motion_context: clip %d has no previous latent at %s; "
                    "using zero context trim", clip, expected_latent)
        effective_trim = state["trim_frames"] if context_available else 0
        output_tail = 0
        if partial_final and state["final_tail_mode"] == "audio plus tail":
            output_tail = state["final_tail_frames"]
        generation_end_frame = start_frame + state["chunk_frames"]
        if clip > 1:
            generation_end_frame += effective_trim
        if partial_final:
            generation_end_frame += output_tail
        output_end_frame = end_frame + output_tail
        start = int(round(start_frame / state["fps"] * state["sample_rate"]))
        end = int(round(generation_end_frame / state["fps"] *
                        state["sample_rate"]))
        if start >= end:
            raise RuntimeError("h3_motion_context: chain has already finished; "
                               "enable reset for a new run")
        chunk_waveform = audio["waveform"][..., start:min(end, state["total_samples"])].contiguous()
        if end > state["total_samples"]:
            chunk_waveform = torch.nn.functional.pad(
                chunk_waveform, (0, end - state["total_samples"]))
        seconds = (end - start) / float(state["sample_rate"])
        source_seconds = (end_frame - start_frame) / state["fps"]
        silent_samples = max(0, end - state["total_samples"])
        chain_config = {
            "chain_id": chain_id,
            "run_name": _run_name(chain_id),
            "latent_prefix": "h3_context/%s_clip" % _run_name(chain_id),
            "chunk_seconds": float(chunk_seconds),
            "fps": state["fps"],
            "chunk_frames": state["chunk_frames"],
            "total_frames": state["total_frames"],
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
        return ({"waveform": chunk_waveform, "sample_rate": state["sample_rate"]},
                seconds, clip, chain_id, prompt, chain_config, audio)


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

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "advance"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Automatically requeues the next H3 chunk, advances the "
                   "Motion Context latent slots, and stitches all clips into "
                   "one MP4 when the audio is complete.")

    def advance(self, video, chain_config, audio_source="original",
                delete_completed_latents=False, audio=None):
        use_original_audio = audio_source in ("original", "original audio input")
        use_generated_audio = audio_source in ("generated", "generated video audio")
        if use_original_audio and audio is None:
            raise ValueError(
                "h3_motion_context: original audio input requires the "
                "complete source audio connected to H3 Auto Chain + Stitch")
        if not use_original_audio and not use_generated_audio:
            raise ValueError("h3_motion_context: unknown audio source %r" %
                             audio_source)
        chain_id = str(chain_config["chain_id"])
        output_prefix = "video/%s" % chain_config["run_name"]
        with _LOCK:
            state = _CHAINS.get(chain_id)
            if state is None:
                raise RuntimeError("h3_motion_context: connect the matching "
                                   "Auto Chain Audio node")
            clip = int(state["clip"])
            next_start_frame = clip * state["chunk_frames"]
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
        clip_frames = int(chain_config["clip_frames"])
        fps = float(chain_config["fps"])
        clip_duration = clip_frames / fps
        try:
            video.save_to(raw_path, format="mp4", codec="h264")
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel",
                "error", "-y", "-i", raw_path,
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "fps=%.6f,tpad=stop_mode=clone:stop_duration=10" % fps,
                "-frames:v", str(clip_frames),
                "-af", "atrim=start=0:end=%.9f,apad,atrim=end=%.9f,asetpts=PTS-STARTPTS" %
                       (clip_duration, clip_duration),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", "%.6f" % fps,
                "-c:a", "aac", "-b:a", "192k", "-t", "%.9f" % clip_duration,
                "-avoid_negative_ts", "make_zero", temp_path,
            ], check=True)
            os.replace(temp_path, clip_path)
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
            os.replace(frame_temp, frame_path)
        except BaseException:
            if os.path.exists(frame_temp):
                os.remove(frame_temp)
            raise
        with _LOCK:
            state["reference_frame_path"] = frame_path
        _LOG.info("h3_motion_context: saved last-frame reference %s", frame_path)
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
            _requeue(str(load_node_id), str(save_node_id), clip, clip + 1)
            _LOG.info("h3_motion_context: queued automatic clip %d", clip + 1)
        else:
            final_path = _output_path(output_prefix, ".mp4")
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
            _stitch_videos(paths, frame_counts, stitch_frames, stitch_audio,
                           final_path, fps)
            if len(paths) < int(math.ceil(state["total_frames"] /
                                         float(state["chunk_frames"]))):
                _LOG.warning(
                    "h3_motion_context: stitched sparse chain with clips %s",
                    ", ".join(str(index) for index in clip_indices))
            if delete_completed_latents:
                _cleanup_chain_latents(chain_config["run_name"])
            _LOG.info("h3_motion_context: automatic chain complete at clip %d", clip)
        return (video,)


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
    def IS_CHANGED(cls, chain_config, initial_image,
                   reference_mode="last_frame"):
        return (reference_mode, chain_config.get("reference_frame_path"))

    def reference(self, chain_config, initial_image,
                  reference_mode="last_frame"):
        if reference_mode == "original":
            _LOG.info("h3_motion_context: using original reference image for "
                      "clip %s", chain_config.get("clip_index", "?"))
            return (initial_image,)
        if reference_mode == "off" and int(chain_config.get("clip_index", 1)) > 1:
            return (initial_image,)
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
    "MiniMaxH3AutoChainSaveLatent": MiniMaxH3AutoChainSaveLatent,
    "MiniMaxH3AutoChainLoadLatent": MiniMaxH3AutoChainLoadLatent,
    "MiniMaxH3AutoChainAudio": MiniMaxH3AutoChainAudio,
    "MiniMaxH3AutoChain": MiniMaxH3AutoChain,
    "MiniMaxH3AutoChainFrameReference": MiniMaxH3AutoChainFrameReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AutoChainMotionContext": "H3 Auto Chain Motion Context",
    "MiniMaxH3AutoChainMotionContextTrim": "H3 Auto Chain Motion Context Trim",
    "MiniMaxH3AutoChainSaveLatent": "H3 Auto Chain Save Latent",
    "MiniMaxH3AutoChainLoadLatent": "H3 Auto Chain Load Latent",
    "MiniMaxH3AutoChainAudio": "H3 Auto Chain Audio",
    "MiniMaxH3AutoChain": "H3 Auto Chain + Stitch",
    "MiniMaxH3AutoChainFrameReference": "H3 Auto Chain Frame Reference",
}
