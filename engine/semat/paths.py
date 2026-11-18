"""Engine paths + vendor imports. Import this before any model code.

All model weights live inside engine/models so the engine can be bundled and run fully offline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ENGINE_DIR / "models"
VENDOR_DIR = ENGINE_DIR / "vendor"
CACHE_DIR = Path(os.environ.get("SEMATTE_CACHE", Path.home() / ".se_matte" / "cache"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face downloads go into engine/models/hf (bundle-able); offline once weights exist.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")  # model libs print progress bars; we report our own
if os.environ.get("SEMATTE_OFFLINE", "1") == "1" and (MODELS_DIR / "hf" / "hub").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

for sub in ("sam2", "MatAnyone", "RobustVideoMatting", "BiRefNet"):
    p = str(VENDOR_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
