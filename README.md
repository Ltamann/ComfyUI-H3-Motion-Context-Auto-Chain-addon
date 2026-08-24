# ComfyUI MiniMax H3 Motion Context Auto-Chain Addon

Automatic long-form video continuation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and MiniMax H3 video workflows.

This addon extends the original **ComfyUI H3 Motion Context** custom node with automatic audio chunking, latent-based clip continuation, per-clip prompts, final-frame references, automatic prompt requeueing, and MP4 video stitching.

It is designed for creators who want to generate a long MiniMax H3 video as a sequence of connected clips while preserving motion, audio, characters, and scene continuity.

## Features

- Automatically split long audio into sequential MiniMax H3 video clips.
- Calculate clip starts and audio cuts on a user-selected frame timeline (24 fps by default).
- Preserve the original full audio timeline for gap-free final stitching.
- Automatically queue the next clip without manual clip-number changes.
- Carry video and audio continuity through H3 latent context.
- Save and load retry-safe numbered H3 latent slots.
- Use a shared style prompt plus a different prompt for every clip.
- Reuse the previous clip's final frame as the next image reference.
- Select original-image, last-frame, or disabled reference behavior.
- Trim generated clips to their intended duration.
- Save individual MP4 clips and stitch them into one final MP4.
- Keep the original H3 Motion Context package unchanged.

## Requirements

This addon requires:

1. ComfyUI.
2. The original `ComfyUI-H3-Motion-Context` custom node package.
3. A working MiniMax H3 video workflow.
4. The Python packages used by the H3 workflow, including `safetensors`, `numpy`, `Pillow`, and `imageio-ffmpeg`.

## Installation

Install both folders inside `ComfyUI/custom_nodes/`:

```text
ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context-orig/
ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context-Auto-Chain-addon/
```

Restart ComfyUI after installing the addon.

Do not install the experimental modified H3 package alongside the original package and this addon. The modified fork registers duplicate original node IDs and can override the stock H3 nodes.

## Nodes included

### H3 Auto Chain Motion Context

Use this node in automatic-chain workflows instead of the original H3 Motion Context node. It delegates to the original node whenever previous context is available. On clip 1, when no previous latent or context frames exist, it passes the conditioning through and reports `trim_frames` as `0`.

This first-clip pass-through is required because an automatic chain has no previous clip to pin yet. The original H3 Motion Context node remains unchanged and should still be used for normal manual continuation workflows.

### H3 Auto Chain Audio

Splits the complete audio input into sequential clip-sized chunks and outputs the current clip's audio, prompt, clip number, and internal chain configuration.

Important inputs:

- `chain_id`: unique name for the project.
- `chunk_seconds`: duration of each generated clip; the default is 20 seconds.
- `fps`: generated video frame rate; default 24. Clip boundaries and audio
  sample cuts are calculated from this frame rate.
- `trim_frames`: leading frames removed by `H3 Motion Context Trim` from
  continuation clips. Match the Motion Context context length; the default
  `22` is about one second at 24 FPS. The addon requests this extra audio
  span so the trimmed clip still occupies its complete timeline slot.
- `reset`: starts a new chain.
- `style_prompt`: text shared by every clip.
- `clip_prompts`: optional numbered prompts such as `[1]`, `[2]`, and `[3]`.
- `start_clip`: clip number to begin with.
- `end_clip`: final clip number; `0` continues until the audio ends.

Connect its `chain_config` output to the other addon nodes.

The `source_audio` output is the complete original audio. Connect it to the
optional `audio` input of `H3 Auto Chain + Stitch` for frame-aligned final
audio instead of concatenating separately encoded clip audio.

### H3 Auto Chain Load Latent

Loads the previous H3 video/audio latent for the original H3 Motion Context node.

Connect its `LATENT` output only to:

```text
H3 Auto Chain Motion Context → context_latent
```

When `chain_config` is connected, the correct previous clip slot is selected automatically. The first clip correctly starts without a previous latent.

### H3 Auto Chain Save Latent

Saves the current H3 sampler output so the next clip can continue from it.

Connect the H3 sampler's output latent to this node. Connect `chain_config` from `H3 Auto Chain Audio` to automatically select the correct numbered save slot.

The saved latent is for H3 Motion Context only. Do not connect it to a VAE decode node.

### H3 Auto Chain Frame Reference

Provides an image reference for the current clip.

Connect:

- `chain_config` from `H3 Auto Chain Audio`.
- `initial_image` from the first image used by the workflow.
- `reference_image` to the H3 reference-image input.

Reference modes:

- `original`: use the initial image for every clip.
- `last_frame`: use the initial image for clip 1 and the previous clip's final frame afterward.
- `off`: use the image only for clip 1.

`last_frame` is the recommended mode for character and scene continuity.

### H3 Auto Chain + Stitch

Saves the current generated video, extracts its final frame, queues the next clip, and stitches all completed clips into one MP4 when the chain finishes.

Connect the final generated `VIDEO` and the `chain_config` output from `H3 Auto Chain Audio`.
Choose `audio_source`:

- `original audio input`: use the complete original audio. Connect `source_audio` from
  `H3 Auto Chain Audio` to the stitch node's optional `audio` input. This is
  the recommended mode when the final video must match the source timeline.
- `generated video audio`: use the audio carried by each generated video clip. No original
  audio connection is required.

Use `delete_completed_latents` to choose whether the chain's saved latent files
are removed after a successful final stitch. It is disabled by default, so the
files remain available for retry or resume.

## Recommended workflow

```text
Complete AUDIO
    ↓
H3 Auto Chain Audio
    ├── audio chunk → H3 video workflow
    ├── prompt → text-conditioning workflow
    └── chain_config → addon chain nodes

H3 Auto Chain Load Latent
    ↓
H3 Motion Context → context_latent

Initial IMAGE
    ↓
H3 Auto Chain Frame Reference
    ↓
H3 reference-image input

Sampler output LATENT
    ↓
H3 Auto Chain Save Latent

Decoded and trimmed VIDEO
    ↓
H3 Auto Chain + Stitch
```

Use `H3 Auto Chain Motion Context` for automatic-chain workflows. It uses the original H3 Motion Context implementation for continuation clips and bypasses context pinning only for clip 1. Connect its `trim_frames` output to the original H3 Motion Context Trim node before passing the final video to the stitch node.

## Starting a new video chain

Use these settings for a new project:

```text
chain_id:      a unique project name
chunk_seconds: 20
fps:           24
trim_frames:  22
reset:         True
start_clip:    1
end_clip:      0
reference_mode: last_frame
```

Queue the workflow once. The addon generates clip 1, saves its latent and final frame, automatically queues clip 2, and continues until the audio ends.

## Per-clip prompting

Use `style_prompt` for text that should remain consistent:

```text
cinematic lighting, realistic character, detailed face, consistent wardrobe
```

Use `clip_prompts` for changing actions or locations:

```text
[1] The character walks through a rainy city street.
[2] The character enters a warm cafe.
[3] The character looks out of the cafe window at the night traffic.
```

The addon combines the shared style prompt with the prompt for the current clip.

## Resume a chain

To continue after clips 1 and 2:

1. Keep the same `chain_id`.
2. Set `start_clip` to `3`.
3. Set `reset` to `True` to initialize the resumed chain state.
4. Confirm that the previous latent and video files still exist.
5. Queue the workflow.

Previous clip files are required for final video stitching.

## Retry a clip safely

The addon stores numbered latent slots, for example:

```text
h3_context/<chain>_clip_00001.safetensors
h3_context/<chain>_clip_00002.safetensors
h3_context/<chain>_clip_00003.safetensors
```

To retry a clip, use the same `chain_id`, set `start_clip` to the clip number, set `reset` to `True`, and queue again. The clip's own slot is overwritten, so a rejected render is not accidentally used as the previous context.

## Output files

Generated files are stored under the ComfyUI output directory, typically:

```text
output/video/<chain>_clip_001.mp4
output/video/<chain>_clip_001_last.png
output/video/<chain>_clip_002.mp4
output/video/<chain>_clip_002_last.png
output/video/<chain>.mp4
```

The final file without `_clip_###` is the stitched MP4.

When `delete_completed_latents` is enabled, a successful automatic-chain stitch
removes that chain's temporary `.safetensors` files from
`output/h3_context`. They remain available when the option is disabled or the
chain fails before final stitching. During a chain, clip `N` loads clip `N-1`
and saves clip `N`; clip 1 starts without a previous latent.

## Important limitations

### Keep the same resolution

When using `context_latent`, every clip in a chain must use the same resolution. Start a new chain with a new `chain_id` if the resolution changes.

### Use a unique chain ID

The chain ID controls the internal state, latent filenames, video filenames, and final-frame references. Do not reuse it for unrelated projects.

### Do not decode the saved context latent

The saved latent contains H3's paired video/audio representation. It is intended only for the original H3 Motion Context node's `context_latent` input.

### Keep only one base H3 package

Install the original H3 Motion Context package once, together with this addon. Do not run multiple forks that register the same original H3 node IDs.

## Troubleshooting

### The addon cannot find a previous latent

Check that:

- The `chain_id` is unchanged.
- `start_clip` is correct.
- The previous clip was saved successfully.
- The addon Load Latent node, not the stock ComfyUI Load Latent node, is connected.

### The chain starts from clip 1 again

Check that `H3 Auto Chain Audio` is the only chain-audio node in the graph and that the same `chain_id` is used throughout the workflow.

### The final video is missing clips

Make sure the final generated video is connected to `H3 Auto Chain + Stitch` and that previous clip MP4 files have not been moved or deleted.

### The character changes between clips

Try:

- Setting `reference_mode` to `last_frame`.
- Using a stronger shared style prompt.
- Keeping the same resolution and model settings.
- Connecting the previous latent through H3 Motion Context.

## Summary

The H3 Motion Context Auto-Chain addon turns a standard MiniMax H3 ComfyUI workflow into an automatic long-video generation workflow:

1. `H3 Auto Chain Audio` splits the audio and manages clip state.
2. `H3 Auto Chain Load Latent` loads the previous clip context.
3. The original `H3 Motion Context` node preserves motion and sound.
4. `H3 Auto Chain Frame Reference` preserves image and character identity.
5. `H3 Auto Chain Save Latent` stores the current clip for continuation.
6. `H3 Auto Chain + Stitch` queues the next clip and creates the final MP4.

Recommended starting settings are 20-second chunks, 22 context frames, 24 audio context frames, and `last_frame` reference mode.
