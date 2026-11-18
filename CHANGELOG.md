# Changelog

## v0.1.0 – 2026-11-18 · First release

The first complete version of SE Matte: a local AI background remover for After Effects.

### Panel
- New split layout: preview on the left, controls on the right (stacked on narrow panels); the whole
  frame always fits and **Remove background** is always visible.
- Premium dark theme with a cyan accent, glowing selected states, gradient-filled sliders and a layer thumbnail.
- Welcome screen with three steps; **Overlay / Cutout / Original** preview views; frame number on the image.
- Live **Edge look** sliders (choke, feather, contrast) for Fast mode.
- Render queue with Fast/Pro tags, progress with time left, errors shown inline; **Apply to layer**, **Show** and
  **Restore original**.
- Follows the After Effects playhead; Pro click markers stay a constant size on screen.

### Fast mode (people, no clicks)
- Robust Video Matting (ResNet-50) on Core ML / Apple Neural Engine in a native Swift helper.
- **~48 fps at 1080p (Standard), ~27 fps (Max)** on an M4 – 1.5× / 1.25× faster than the first build, with
  identical alpha: the model now runs only RVM's low-res pass and the full-res guided-filter step runs on the CPU
  in parallel.
- Production edges: blur-fusion edge colour (no background spill), tightened alpha (no fuzz), hair detail kept,
  no added flicker.
- Persistent preview helper: frames in ~17 ms, slider changes in 7–50 ms.

### Pro mode (any subject, click to select)
- SAM 2.1 click selection with correction keyframes → MatAnyone soft matte → ViTMatte edge refinement.
- Stages run concurrently (GPU, Neural Engine, CPU); re-runs with only look changes reuse the AI work.

### Engine
- Local FastAPI server on `127.0.0.1:47321` with a job queue; started automatically by the panel.
- Rotation-aware probing (phone portrait clips), exact frame-rate handling (23.976 / 29.97).
- Free-disk-space check before a render starts.
- In-place apply: the cutout replaces the layer's source; re-applying reuses the project item and cleans up
  unused older cutouts.

### Setup
- `engine/setup_mac.sh`: Python 3.11 environment, pinned third-party model code, model weights, panel install.
