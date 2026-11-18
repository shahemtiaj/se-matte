"""Pro mode pipeline (v3, concurrent).

  1. SAM 2.1 image     – your clicks → subject mask on each keyframe
  2. MatAnyone         – soft, temporally stable matte from the keyframe masks (GPU)
  3. Fusion + ViTMatte – specks removed / pinholes filled, every edge re-solved at full resolution
                         (Neural Engine on Macs, GPU elsewhere)
  4. Export            – look controls + ProRes 4444 / matte / PNG (CPU + ffmpeg)

Stages 2–4 run at the same time as a frame pipeline: while the GPU mattes frame N, the Neural
Engine refines frame N-k and the CPU writes frame N-2k. Total time ≈ the slowest stage instead of
the sum of all three. Each stage caches its frames, so a re-run (or a change in later stages)
skips finished work.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import paths
from .device import DeviceInfo, empty_cache, pick_device
from .fusion import Fuser, FusionSettings
from .refine import RefineSettings, TemporalSmoother, compose_rgba, guided_upscale, shape
from .video_io import FrameWriter, VideoInfo, probe, read_frame, read_frames

Progress = Callable[[str, float, str], None]  # (stage, 0..1, message)
PIPELINE_VERSION = 3

QUALITY = {
    # name: (MatAnyone short side, ViTMatte variant)
    "standard": (720, "small"),
    "max": (1080, "base"),
}


@dataclass
class Keyframe:
    frame: int                         # absolute source frame index
    points: list[tuple[float, float]]  # source pixel coords
    labels: list[int]                  # 1 = keep, 0 = remove

    def to_dict(self):
        return {"frame": self.frame, "points": [list(p) for p in self.points], "labels": list(self.labels)}


@dataclass
class ProJob:
    source: str
    start: int                        # first source frame (inclusive)
    end: int                          # last source frame (exclusive)
    keyframes: list[Keyframe]
    quality: str = "standard"
    refine: RefineSettings = field(default_factory=RefineSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    outputs: tuple[str, ...] = ("prores4444",)  # any of prores4444, matte, png
    out_dir: str | None = None

    def key(self, *parts) -> str:
        st = os.stat(self.source)
        blob = json.dumps({
            "v": PIPELINE_VERSION, "src": os.path.abspath(self.source), "mtime": st.st_mtime,
            "size": st.st_size, "start": self.start, "end": self.end,
            "kf": [k.to_dict() for k in sorted(self.keyframes, key=lambda k: k.frame)], "parts": parts,
        }, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _noop(stage, p, msg):
    pass


def _png(path: Path, a: np.ndarray) -> None:
    """Write atomically (tmp + rename) so another process never sees a half-written frame."""
    tmp = path.with_name(path.stem + ".tmp.png")
    cv2.imwrite(str(tmp), np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8) if a.dtype != np.uint8 else a)
    os.replace(tmp, path)


def _load(path: Path) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise RuntimeError(f"Missing cached frame {path.name}; re-run processing.")
    return x


def _reverse_frames(info: VideoInfo, first: int, last_excl: int, chunk: int = 48) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (idx, frame) from last_excl-1 down to `first`, decoding in chunks to bound memory."""
    hi = last_excl
    while hi > first:
        lo = max(first, hi - chunk)
        buf = list(read_frames(info, lo, hi - lo))
        for i in range(len(buf) - 1, -1, -1):
            yield lo + i, buf[i]
        del buf
        hi = lo


# ------------------------------------------------------------------ pipeline plumbing
class _Abort(Exception):
    """Another stage failed or the job was cancelled."""


class _Ready:
    """Which frames a stage has finished; consumers block until their frame is there."""

    def __init__(self, board: "_Board"):
        self.board = board
        self.done: set[int] = set()

    def mark(self, i: int):
        with self.board.cv:
            self.done.add(i)
            self.board.cv.notify_all()

    def mark_all(self, idxs):
        with self.board.cv:
            self.done.update(idxs)
            self.board.cv.notify_all()

    def wait(self, i: int):
        with self.board.cv:
            while i not in self.done:
                if self.board.error is not None:
                    raise _Abort()
                self.board.cv.wait(0.5)


class _Board:
    """Shared state of one run: error flag + combined progress of the concurrent stages."""

    STAGES = ("matte", "edges", "export")
    LABEL = {"matte": "Soft matte", "edges": "Edges", "export": "Writing"}

    def __init__(self, total: int, progress: Progress):
        self.cv = threading.Condition()
        self.error: BaseException | None = None
        self.total = max(total, 1)
        self.progress = progress
        self.count = {s: 0 for s in self.STAGES}
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.on_fail: list[Callable[[], None]] = []   # e.g. stop worker processes

    def fail(self, e: BaseException):
        with self.cv:
            first = self.error is None
            if first:
                self.error = e
            self.cv.notify_all()
        if first:
            for h in list(self.on_fail):
                try:
                    h()
                except Exception:  # noqa: BLE001
                    pass

    def skip(self, stage: str):
        self.count[stage] = self.total

    def tick(self, stage: str):
        if self.error is not None:
            raise _Abort()
        with self.lock:
            self.count[stage] += 1
            c = self.count
            el = max(time.time() - self.t0, 1e-6)
            # the slowest stage decides; export can't finish before edges, edges not before matte
            p = (0.25 * c["matte"] + 0.6 * c["edges"] + 0.15 * c["export"]) / self.total
            done = c["export"]
            fps = done / el if done else 0.0
            # ETA from the stage that is furthest behind relative to its own speed
            eta = 0.0
            for s in self.STAGES:
                if 0 < c[s] < self.total:
                    eta = max(eta, (self.total - c[s]) / (c[s] / el))
            parts = [f"{self.LABEL[s]} {c[s]}/{self.total}" for s in self.STAGES if c[s] < self.total]
            msg = " · ".join(parts or ["Finishing"]) + (f" · {fps:.2f} fps" if fps else "") + \
                (f" · ~{eta:.0f}s left" if eta else "")
            self.progress("all", min(p, 0.999), msg)


def _run_threads(board: _Board, fns: list[Callable[[], None]]):
    errors: list[BaseException] = []

    def wrap(fn):
        def go():
            try:
                fn()
            except _Abort:
                pass
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                board.fail(e)
        return go

    threads = [threading.Thread(target=wrap(f), daemon=True) for f in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if board.error is not None:
        raise board.error


# ------------------------------------------------------------------ stages
def _stage_keymasks(job, info, dev, d: Path, progress) -> dict[int, np.ndarray]:
    kfs = sorted([k for k in job.keyframes if job.start <= k.frame < job.end], key=lambda k: k.frame)
    if not kfs:
        raise ValueError("Add at least one click point inside the processed range.")
    d.mkdir(parents=True, exist_ok=True)
    masks, todo = {}, []
    for k in kfs:
        p = d / f"key_{k.frame:06d}.png"
        if p.exists():
            masks[k.frame] = _load(p) > 127
        else:
            todo.append(k)
    if todo:
        progress("ai", 0.0, "Finding subject from your clicks")
        from .models.sam2_seg import Sam2Segmenter

        seg = Sam2Segmenter(dev)
        for k in todo:
            masks[k.frame] = seg.segment(read_frame(info, k.frame), k.points, k.labels)
            _png(d / f"key_{k.frame:06d}.png", masks[k.frame].astype(np.uint8) * 255)
        del seg
        empty_cache(dev)
    return masks


def _stage_matte(job, info, dev, kfs, masks, size: int, d: Path, board: _Board, out: _Ready) -> None:
    frames = range(job.start, job.end)
    if (d / "done").exists():
        out.mark_all(frames)
        board.skip("matte")
        return
    d.mkdir(parents=True, exist_ok=True)
    from .models.matanyone_matte import MatAnyoneMatter

    matter = MatAnyoneMatter(dev, internal_size=min(size, min(info.width, info.height)))

    def put(i, a):
        _png(d / f"{i:06d}.png", a)
        out.mark(i)
        board.tick("matte")

    k0 = kfs[0]
    if k0.frame > job.start:  # backwards: first keyframe → range start
        idxs = []

        def rev():
            for i, f in _reverse_frames(info, job.start, k0.frame + 1):
                idxs.append(i)
                yield f

        for n, a in enumerate(matter.run(rev(), masks[k0.frame])):
            if n:
                put(idxs[n], a)
    for i, k in enumerate(kfs):  # forwards: each keyframe → next keyframe / range end
        seg_end = kfs[i + 1].frame if i + 1 < len(kfs) else job.end
        for n, a in enumerate(matter.run(read_frames(info, k.frame, seg_end - k.frame), masks[k.frame])):
            put(k.frame + n, a)
        empty_cache(dev)
    del matter
    empty_cache(dev)
    (d / "done").write_text("1")


def _stage_matte_proc(job, info, dev, kfs, masks, size: int, d: Path, board: _Board, out: _Ready) -> None:
    """_stage_matte in a worker process (see matte_worker.py for why)."""
    import pickle
    import subprocess
    import sys
    import tempfile

    frames = range(job.start, job.end)
    if (d / "done").exists():
        out.mark_all(frames)
        board.skip("matte")
        return
    fd, arg = tempfile.mkstemp(prefix="sematte_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump({"job": job, "info": info, "device": dev.kind, "kfs": kfs, "masks": masks,
                     "size": size, "dir": d}, f)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "-m", "semat.matte_worker", arg], cwd=str(paths.ENGINE_DIR),
                         stdout=subprocess.PIPE, stderr=None, text=True, env=env)
    board.on_fail.append(lambda: p.poll() is None and p.terminate())
    finished = False
    try:
        for line in p.stdout:
            line = line.strip()
            if line.startswith("M "):
                out.mark(int(line[2:]))
                board.tick("matte")
            elif line == "ALL":
                out.mark_all(frames)
                board.skip("matte")
            elif line == "DONE":
                finished = True
                break
            elif line.startswith("ERR "):
                raise RuntimeError("Soft matte failed: " + line[4:])
            if board.error is not None:
                raise _Abort()
        if not finished:
            p.wait()
            if board.error is not None:
                raise _Abort()
            raise RuntimeError(f"Soft matte process stopped unexpectedly (code {p.returncode}).")
        p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        try:
            os.remove(arg)
        except OSError:
            pass


def _stage_refine(job, info, dev, vit: str, mdir: Path, d: Path, board: _Board,
                  inp: _Ready, out: _Ready) -> None:
    frames = range(job.start, job.end)
    if (d / "done").exists():
        out.mark_all(frames)
        board.skip("edges")
        return
    d.mkdir(parents=True, exist_ok=True)
    from .models.vitmatte_refine import VitMatteRefiner, make_trimap

    ref = VitMatteRefiner(dev, variant=vit)
    fuser = Fuser(job.fusion)
    H, W = info.height, info.width
    band = max(4, int(round(job.fusion.band * H / 1080)))
    for n, img in enumerate(read_frames(info, job.start, job.end - job.start)):
        i = job.start + n
        inp.wait(i)
        mat = _load(mdir / f"{i:06d}.png").astype(np.float32) / 255.0
        if mat.shape != (H, W):
            mat = guided_upscale(mat, img)
        coarse = fuser.coarse(mat)
        tri = make_trimap(coarse, band)
        alpha = fuser.finish(ref.refine(img, tri), tri)
        _png(d / f"{i:06d}.png", alpha)
        out.mark(i)
        board.tick("edges")
        if ref.backend == "torch" and n % 16 == 0:
            empty_cache(dev)
    del ref
    empty_cache(dev)
    (d / "done").write_text("1")


def _stage_export(job: ProJob, info: VideoInfo, alpha_dir: Path, board: _Board, inp: _Ready) -> dict[str, str]:
    out_dir = Path(job.out_dir or Path(job.source).parent / "SE Matte Renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{Path(job.source).stem[:60]}_SEMatte_{job.start}-{job.end - 1}"
    stem, v = base, 1
    # never overwrite a render that may already be imported in an AE project
    while any((out_dir / f"{stem}{ext}").exists() for ext in (".mov", "_matte.mov", "")):
        v += 1
        stem = f"{base}_v{v}"

    writers, results = {}, {}
    W, H = info.width, info.height
    if "prores4444" in job.outputs:
        results["prores4444"] = str(out_dir / f"{stem}.mov")
        writers["prores4444"] = FrameWriter(results["prores4444"], "prores4444", W, H, info.fps)
    if "matte" in job.outputs:
        results["matte"] = str(out_dir / f"{stem}_matte.mov")
        writers["matte"] = FrameWriter(results["matte"], "matte", W, H, info.fps)
    if "png" in job.outputs:
        d = out_dir / stem
        results["png"] = str(d)
        writers["png"] = FrameWriter(str(d / f"{stem}_%05d.png"), "png", W, H, info.fps)

    smooth = TemporalSmoother(job.refine.temporal)
    ok = False
    try:
        for n, img in enumerate(read_frames(info, job.start, job.end - job.start)):
            i = job.start + n
            inp.wait(i)
            a = _load(alpha_dir / f"{i:06d}.png").astype(np.float32) / 255.0
            if a.shape != (H, W):
                a = guided_upscale(a, img)
            a = smooth(shape(a, job.refine))
            if "prores4444" in writers or "png" in writers:
                rgba = compose_rgba(img, a, job.refine.decontaminate)
                for k in ("prores4444", "png"):
                    if k in writers:
                        writers[k].write(rgba)
            if "matte" in writers:
                writers["matte"].write(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
            board.tick("export")
        ok = True
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        if not ok:  # don't leave half-written renders behind
            for k, p in results.items():
                if k != "png" and os.path.exists(p):
                    os.remove(p)
    return results


def _stage_export_proc(job: ProJob, info: VideoInfo, alpha_dir: Path, board: _Board) -> dict:
    """_stage_export in a worker process (see export_worker.py)."""
    import pickle
    import subprocess
    import sys
    import tempfile

    fd, arg = tempfile.mkstemp(prefix="sematte_x_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump({"job": job, "info": info, "alpha_dir": alpha_dir, "parent": os.getpid()}, f)
    p = subprocess.Popen([sys.executable, "-m", "semat.export_worker", arg], cwd=str(paths.ENGINE_DIR),
                         stdout=subprocess.PIPE, stderr=None, text=True,
                         env=dict(os.environ, PYTHONUNBUFFERED="1"))
    board.on_fail.append(lambda: p.poll() is None and p.terminate())
    result = None
    try:
        for line in p.stdout:
            line = line.strip()
            if line == "X":
                board.tick("export")
            elif line.startswith("RES "):
                result = json.loads(line[4:])
                break
            elif line.startswith("ERR "):
                raise RuntimeError("Export failed: " + line[4:])
        if result is None:
            p.wait()
            if board.error is not None:
                raise _Abort()
            raise RuntimeError(f"Export process stopped unexpectedly (code {p.returncode}).")
        p.wait()
        return result
    finally:
        if p.poll() is None:
            p.terminate()  # its own cleanup removes the half-written render
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        try:
            os.remove(arg)
        except OSError:
            pass


# ------------------------------------------------------------------ entry points
def _dirs(job: ProJob, info: VideoInfo, dev: DeviceInfo):
    size, vit = QUALITY[job.quality]
    root = paths.CACHE_DIR / job.key()
    from .models import vitmatte_coreml

    backend = "coreml" if (dev.kind == "mps" and vitmatte_coreml.available()) else "torch"
    final = root / f"final_{job.key(job.quality, asdict(job.fusion), backend)}"
    return size, vit, root, root / f"matte_{size}", final


def run(job: ProJob, dev: DeviceInfo | None = None, progress: Progress = _noop) -> dict:
    """AI stages + export, overlapped. Returns output paths, seconds and the alpha cache dir."""
    t0 = time.time()
    info = probe(job.source)
    dev = dev or pick_device()
    size, vit, root, mdir, final = _dirs(job, info, dev)
    board = _Board(job.end - job.start, progress)
    matte_ready, final_ready = _Ready(board), _Ready(board)
    results: dict = {}

    if (final / "done").exists():
        progress("ai", 0.02, "Using cached AI result")
        final_ready.mark_all(range(job.start, job.end))
        board.skip("matte")
        board.skip("edges")
        stages = []
    else:
        masks = _stage_keymasks(job, info, dev, root / "keys", progress)
        kfs = sorted([k for k in job.keyframes if job.start <= k.frame < job.end], key=lambda k: k.frame)
        stages = [
            lambda: _stage_matte_proc(job, info, dev, kfs, masks, size, mdir, board, matte_ready),
            lambda: _stage_refine(job, info, dev, vit, mdir, final, board, matte_ready, final_ready),
        ]

    def exp():
        results.update(_stage_export_proc(job, info, final, board))

    _run_threads(board, stages + [exp])
    results["seconds"] = round(time.time() - t0, 1)
    results["cache"] = str(final)
    return results


def run_ai(job: ProJob, dev: DeviceInfo | None = None, progress: Progress = _noop) -> Path:
    """AI stages only (no export)."""
    info = probe(job.source)
    dev = dev or pick_device()
    size, vit, root, mdir, final = _dirs(job, info, dev)
    if (final / "done").exists():
        return final
    board = _Board(job.end - job.start, progress)
    matte_ready, final_ready = _Ready(board), _Ready(board)
    masks = _stage_keymasks(job, info, dev, root / "keys", progress)
    kfs = sorted([k for k in job.keyframes if job.start <= k.frame < job.end], key=lambda k: k.frame)
    _run_threads(board, [
        lambda: _stage_matte_proc(job, info, dev, kfs, masks, size, mdir, board, matte_ready),
        lambda: _stage_refine(job, info, dev, vit, mdir, final, board, matte_ready, final_ready),
    ])
    return final


def export(job: ProJob, alpha_dir: Path, progress: Progress = _noop) -> dict[str, str]:
    info = probe(job.source)
    board = _Board(job.end - job.start, progress)
    ready = _Ready(board)
    ready.mark_all(range(job.start, job.end))
    board.skip("matte")
    board.skip("edges")
    return _stage_export(job, info, alpha_dir, board, ready)
