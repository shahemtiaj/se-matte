"""Fast mode: people only, no clicks, ~27–48 fps at 1080p on an M4 (a 10-minute 30 fps clip in ~6 min at Standard).

Runs the native helper (engine/native/semfast → engine/bin/mac/semfast): hardware decode →
Robust Video Matting (Core ML; true soft alpha + spill-free edge colour, stable over time) →
hardware ProRes 4444 with straight alpha. Apple Vision person segmentation is the fallback when the
RVM model can't be prepared. Mac only for now; Windows will get its own fast backend later.
"""
from __future__ import annotations

import json
import os
import platform
import select
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import paths
from .refine import RefineSettings

Progress = Callable[[str, float, str], None]
NATIVE_DIR = paths.ENGINE_DIR / "native"
BIN = paths.ENGINE_DIR / "bin" / "mac" / "semfast"
# quality → (RVM variant, detail of RVM's internal pass, Core ML compute units, Vision fallback level)
QUALITY = {"standard": ("resnet50", 1.5, "ane", "balanced"), "max": ("resnet50", 2.0, "ane", "accurate")}
TIGHTEN = 1.0   # squeeze RVM's soft edge band (levels 0.2–0.8): crisp outline without faint fuzz


@dataclass
class FastJob:
    source: str
    start: int
    end: int                      # exclusive
    quality: str = "standard"     # standard = RVM ResNet-50 at 1.5× detail (~48 fps at 1080p), max = 2× detail (~27 fps)
    refine: RefineSettings = field(default_factory=RefineSettings)
    out_dir: str | None = None


def available() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "Fast mode is Mac-only for now. Use Pro mode on Windows."
    try:
        binary()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def binary() -> Path:
    """The helper binary; (re)built from source when missing or older than the source."""
    src = NATIVE_DIR / "semfast" / "main.swift"
    if BIN.exists() and (not src.exists() or BIN.stat().st_mtime >= src.stat().st_mtime):
        return BIN
    if not src.exists():
        raise RuntimeError("Fast-mode helper is missing. Re-run the installer.")
    r = subprocess.run(["/bin/zsh", str(NATIVE_DIR / "build_mac.sh")], capture_output=True, text=True)
    if r.returncode != 0 or not BIN.exists():
        raise RuntimeError("Could not build the Fast-mode helper (needs Xcode Command Line Tools): "
                           + (r.stderr or r.stdout)[-400:])
    return BIN


_model_lock = threading.Lock()


def _model(source: str, quality: str, progress: Progress | None = None) -> dict:
    """RVM model + compute units for this video's (upright) size, prepared on first use;
    {"quality": …} for the Apple Vision fallback when RVM can't be prepared."""
    from .models import rvm_coreml
    from .video_io import probe

    variant, detail, units, vision = QUALITY.get(quality, QUALITY["standard"])
    info = probe(source)
    try:
        with _model_lock:   # a preview and a job may ask for the same new size at once
            if progress and not rvm_coreml.compiled_path(variant, info.width, info.height, detail).exists():
                progress("fast", 0.0, "Preparing the AI model for this video size (first time only)…")
            return {"model": str(rvm_coreml.ensure(variant, info.width, info.height, detail)), "units": units,
                    "tighten": TIGHTEN}
    except Exception as e:  # noqa: BLE001
        print(f"RVM unavailable ({e}); using Apple Vision.", flush=True)
        return {"quality": vision}


def _model_args(source: str, quality: str, progress: Progress | None = None) -> list[str]:
    return [a for k, v in _model(source, quality, progress).items() for a in (f"--{k}", str(v))]


def _look_args(s: RefineSettings) -> list[str]:
    a = []
    if s.choke:
        a += ["--choke", str(s.choke)]
    if s.feather:
        a += ["--feather", str(s.feather)]
    if s.contrast:
        a += ["--contrast", str(s.contrast)]
    return a


class _Helper:
    """One long-running `semfast serve` process for panel previews: the video stays open and the model
    stays loaded, so a preview takes ~10–100 ms instead of a process start + model load (~300 ms+)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.p: subprocess.Popen | None = None
        self.built = 0.0

    def _start(self):
        b = binary()
        self.built = b.stat().st_mtime
        self.p = subprocess.Popen([str(b), "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.kill()
        self.p = None

    def ask(self, req: dict, timeout: float = 60) -> dict:
        with self.lock:
            if self.p is None or self.p.poll() is not None or binary().stat().st_mtime != self.built:
                self.stop()
                self._start()
            try:
                self.p.stdin.write(json.dumps(req) + "\n")
                self.p.stdin.flush()
                ready, _, _ = select.select([self.p.stdout], [], [], timeout)
                line = self.p.stdout.readline() if ready else ""
            except (BrokenPipeError, OSError):
                line = ""
            if not line:
                self.stop()
                raise RuntimeError("The Fast-mode preview helper stopped. Try again.")
            r = json.loads(line)
            if not r.get("ok"):
                raise RuntimeError(r.get("error") or "preview failed")
            return r


HELPER = _Helper()


def preview(source: str, frame: int, quality: str = "standard", refine: RefineSettings | None = None,
            max_side: int = 960) -> dict:
    """What Fast mode keeps on this frame: {"data": base64 gray PNG, "coverage": 0..1, "ms": …}."""
    s = refine or RefineSettings()
    return HELPER.ask({"cmd": "preview", "src": source, "frame": frame, "max": max_side,
                       "choke": s.choke, "feather": s.feather, "contrast": s.contrast, **_model(source, quality)})


def frame_jpeg(source: str, frame: int, max_side: int = 960) -> bytes:
    """One upright source frame as JPEG via the hardware decoder (faster than starting ffmpeg)."""
    import base64

    return base64.b64decode(HELPER.ask({"cmd": "frame", "src": source, "frame": frame, "max": max_side}, 30)["data"])


def _out_path(job: FastJob) -> Path:
    out_dir = Path(job.out_dir or Path(job.source).parent / "SE Matte Renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{Path(job.source).stem[:60]}_SEMatte_{job.start}-{job.end - 1}"
    stem, v = base, 1
    while any((out_dir / f"{stem}{ext}").exists() for ext in (".mov", "_matte.mov", "")):
        v += 1
        stem = f"{base}_v{v}"
    return out_dir / f"{stem}.mov"


def check_disk_space(out: Path, source: str, frames: int) -> None:
    """Fail before starting when the render can't fit (ProRes 4444 with alpha is ~0.35 bytes per pixel per
    frame on real footage, e.g. ~0.7 MB per 1080×1920 frame), instead of dying near the end with Disk Full."""
    import shutil

    from .video_io import probe

    try:
        info = probe(source)
        need = frames * info.width * info.height * 0.36 * 1.1 + 300e6
        free = shutil.disk_usage(out.parent).free
    except Exception:
        return
    if free < need:
        raise RuntimeError(f"Not enough disk space: this render needs about {need / 1e9:.1f} GB, "
                           f"only {free / 1e9:.1f} GB free on that drive. Free some space (old renders are in "
                           f"\"{out.parent.name}\") and try again.")


def run(job: FastJob, progress: Progress = lambda *a: None) -> dict:
    t0 = time.time()
    out = _out_path(job)
    check_disk_space(out, job.source, job.end - job.start)
    progress("fast", 0.0, "Starting…")
    cmd = [str(binary()), "render", "--src", job.source, "--out", str(out), "--start", str(job.start),
           "--end", str(job.end), *_model_args(job.source, job.quality, progress), *_look_args(job.refine)]
    errf = tempfile.TemporaryFile(mode="w+")   # not a pipe: a full stderr pipe would block the helper
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True)
    t1 = time.time()   # speed is measured from here (model preparation can take a while the first time)
    ok = done_seen = False
    try:
        total = job.end - job.start
        for line in p.stdout:
            line = line.strip()
            if line.startswith("P "):
                done, total = (int(x) for x in line[2:].split())
                el = max(time.time() - t1, 1e-6)
                fps = done / el
                eta = (total - done) / max(fps, 1e-6)
                progress("fast", done / max(total, 1), f"Removing background {done}/{total} · {fps:.0f} fps · ~{eta:.0f}s left")
            elif line.startswith("DONE"):
                done_seen = True
        p.wait()
        if p.returncode != 0 or not done_seen:
            errf.seek(0)
            err = errf.read().strip().splitlines()
            raise RuntimeError((err[-1] if err else f"Fast mode failed (code {p.returncode})").removeprefix("ERR ").strip())
        ok = True
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        errf.close()
        if not ok and out.exists():
            os.remove(out)   # never leave a half-written render
    return {"prores4444": str(out), "seconds": round(time.time() - t0, 1), "mode": "fast"}
