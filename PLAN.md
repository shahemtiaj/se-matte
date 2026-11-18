# SE Matte — Build Plan

**SE Matte – AI Background Remover for After Effects**
Runs 100% locally (no internet after the one-time setup). Built Mac-first (Apple M4), with Windows support built in from day one and tuned later.

---

## 1. What we're building

An After Effects panel where you:

1. Select a footage layer in your comp
2. Click on the subject in a preview frame (green dot = keep, red dot = remove)
3. Pick a mode and press **Process**
4. SE Matte creates a clean alpha matte and drops it into your comp as a track matte automatically

Everything is processed by a local AI engine on your machine.

---

## 2. Architecture

```
┌───────────────────────────────────────────────┐
│  After Effects (2024 / 2027, Mac & Windows)   │
│  ┌─────────────────────────────────────────┐  │
│  │ SE Matte panel (CEP: HTML/JS)           │  │
│  │ preview + click points, modes, controls │  │
│  └──────────────┬──────────────────────────┘  │
│   ExtendScript: read layer, import result,    │
│   set track matte                             │
└─────────────────┼─────────────────────────────┘
                  │ localhost only (127.0.0.1)
┌─────────────────▼─────────────────────────────┐
│  SE Matte Engine (Python, local server)       │
│  • Device layer: Mac = Metal (MPS)            │
│                  Win = CUDA / CPU fallback    │
│  • Model registry (swappable models)          │
│  • Chunked processing + memory guard (16 GB)  │
│  • Alpha cache (re-tweak without re-running)  │
│  • ffmpeg in/out (ProRes 4444, PNG, matte)    │
└───────────────────────────────────────────────┘
```

**Why this design:** offline processing lets us use the best video models, which need to read frames in order and remember earlier ones. A live AE effect plugin can't do this reliably. One engine codebase serves both Mac and Windows.

---

## 3. AI modes

| Mode | Models | Best for | M4 speed (rough estimate, to benchmark) |
|---|---|---|---|
| **Pro** (default) | **SAM 2.1** (click → tracks the subject) → **MatAnyone** (soft, stable matte: hair, motion blur) | Anything: people, hair, objects | Slower than real time (~1 min per 10 s of 1080p) |
| **Fast** | **Robust Video Matting** | Talking heads, interviews; no clicks needed | Close to real time |
| **Detail** | **BiRefNet HR** | Products, logos, low-motion shots, very fine edges | Medium |

The model registry makes each model a plug-in, so models can be upgraded or swapped later (for example, for commercial licensing) without touching the panel.

### Quality pipeline (Pro mode)
1. SAM 2.1 builds a mask from your clicks and tracks it through the clip
2. If tracking drifts, add correction clicks on later frames and re-run from there
3. MatAnyone turns the hard mask into a soft, high-detail alpha with temporal memory (no flicker)
4. Refinement: edge-aware upscale to full resolution, plus optional temporal smoothing
5. Edge color cleanup: estimates the true foreground color so the old background doesn't bleed into the edges

---

## 4. Features

**Panel**
- Shows the selected layer: name, resolution, fps, trimmed in/out range
- Preview frame with a scrubber; click to add +/− points; live mask preview before processing
- Mode selector (Pro / Fast / Detail) and quality (Standard / High / Max resolution)
- Process only the layer's trimmed range, not the whole file
- Progress bar with time estimate; cancel button
- Auto-import: the result is placed above the source layer and set as its track matte (or applied as RGBA, your choice)

**Matte controls** (applied from cache, no AI re-run)
- Choke / expand, feather, contrast (crisp ↔ soft), temporal smoothing, edge color cleanup on/off

**Output**
- ProRes 4444 with alpha (default), matte-only (black/white), or PNG sequence
- Output saved next to the project in an `SE Matte Renders/` folder

**Workflow**
- Batch queue: add several layers, keep working while it processes
- Results cache per clip/range/settings
- Hardware check on start: detects GPU/memory and picks safe settings automatically

---

## 5. Mac (M4, 16 GB) optimisations
- SAM 2.1 *base-plus* size: near-max quality at much lower memory use than *large*
- Long clips processed in chunks so memory never runs out
- 4K: AI runs at ~1080p, then the matte is upscaled with edge-aware refinement; a full-res option is still available
- Unsupported Metal operations fall back to CPU automatically, so nothing crashes
- Later: Core ML / Neural Engine versions of Fast and Detail for extra speed

## 6. Windows readiness (kept in, tuned later)
- No Mac-only code in the core engine; all paths are OS-neutral
- Device auto-select: NVIDIA CUDA → CPU fallback (DirectML evaluated later)
- ffmpeg bundled per OS; the CEP panel is the same code on both
- Optional later: **matte server** mode, where weak laptops send clips to a stronger machine over the office network (still no internet)

---

## 7. Build phases

Each phase ends with a checkpoint: you test it and say OK before I continue.

| Phase | What | You get |
|---|---|---|
| **0. Setup** | Install Python 3.11 env (via `uv`), PyTorch, ffmpeg check, download models once | Working AI environment |
| **1. Engine prototype** | Pro mode from the command line on your test clips; benchmark on the M4 | ProRes 4444 results to judge quality and speed |
| **2. Engine complete** | Fast + Detail modes, refinement controls, cache, all output formats, local server | Full engine |
| **3. AE panel** | CEP panel: layer info, click-to-select preview, process, auto-import + track matte | Usable in AE 2024 & 2027 |
| **4. Pro workflow** | Batch queue, correction keyframes, presets, error handling, hardware check | Team-ready tool |
| **5. Team install** | One-click Mac installer (engine + models bundled, works offline); Windows build scripted, testing later | Install on all Mac minis |

---

## 8. Folder structure

```
SE Matte/
├── PLAN.md
├── engine/            # Python AI engine (server + CLI)
│   ├── semat/         #   core: devices, models, pipelines, io, cache
│   └── models/        #   downloaded model weights (git-ignored)
├── panel/             # CEP extension (HTML/JS + ExtendScript)
├── installer/         # mac (.pkg) and windows build scripts
├── tests/clips/       # your test footage
└── docs/              # team user guide
```

---

## 9. Licensing (current use: internal team, which is fine)

| Model | License | Commercial sale later |
|---|---|---|
| SAM 2.1 | Apache 2.0 | ✅ OK |
| BiRefNet | MIT | ✅ OK |
| MatAnyone | Non-commercial (S-Lab) | ❌ Must swap or license |
| Robust Video Matting | GPL-3.0 | ⚠️ Restrictive if distributed |

The swappable model registry keeps a future commercial version possible.

---

## 10. Known limits in v1
- Works on **footage layers** (video and image sequences). Precomps and layers with effects or time-remap will be supported later by rendering them first.
- Pro mode is not real time; it's a "process, then use" workflow.
- Very heavy AE projects plus Pro mode at 4K may be tight on 16 GB; the panel will warn and use safe settings.

---

## 11. ⚠️ Blockers I found on your Mac

| Item | Status |
|---|---|
| Apple M4, 16 GB, macOS 26.3 | ✅ Good |
| After Effects 2024 and 2027 | ✅ Both installed, will support both |
| CEP debug mode | ✅ Already enabled (unsigned panels load) |
| ffmpeg, Homebrew, git | ✅ Installed |
| Python | ⚠️ Only system 3.9. I'll install 3.11 via `uv` (isolated, no system changes) |
| **Free disk space** | ❌ **Only 2.2 GB free (99% full).** Need **at least 15 GB** free: ~4 GB for the AI environment and models, plus room for ProRes 4444 renders |

## 12. What I need from you
1. **Free up ~15 GB+ of disk space** before Phase 0
2. **2–3 typical test clips** from your real work (for example a talking head, hair against a busy background, a fast-motion or object shot). Put them in `tests/clips/`
3. Your OK on this plan
