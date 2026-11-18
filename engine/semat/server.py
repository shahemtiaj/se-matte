"""SE Matte local engine server (talks to the After Effects panel over http://127.0.0.1 only).

  python -m semat.server [--port 47321]

Endpoints
  GET  /health                       engine + device status
  GET  /probe?path=                  video size / fps / frame count
  GET  /frame?path=&frame=&max=      preview JPEG of one source frame
  POST /segment                      clicks → subject mask preview (PNG, base64)
  POST /jobs                         queue a processing job
  GET  /jobs, /jobs/{id}             job status / progress / result
  POST /jobs/{id}/cancel             cancel a queued or running job
  POST /shutdown                     stop the server

Nothing leaves this computer: the server binds to 127.0.0.1 and all models run locally.
The server stops itself after IDLE_MINUTES with no requests and no jobs.
"""
from __future__ import annotations

import argparse
import base64
import gc
import os
import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import __version__, paths  # noqa: F401  (paths sets offline/HF env)
from .device import empty_cache, pick_device
from .fusion import FusionSettings
from .pipeline_pro import QUALITY, Keyframe, ProJob, run
from .refine import RefineSettings
from .video_io import probe, read_frame

DEFAULT_PORT = 47321
IDLE_MINUTES = 20

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="SE Matte engine", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEV = pick_device()
_last_request = time.time()


@app.middleware("http")
async def _touch(request, call_next):
    global _last_request
    _last_request = time.time()
    return await call_next(request)


# ------------------------------------------------------------------ helpers
def _probe(path: str):
    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"File not found: {path}")
    try:
        return probe(path)   # cached per file version
    except Exception as e:  # noqa: BLE001
        raise HTTPException(415, f"Can't read this video: {e}")


def _fit(w: int, h: int, mx: int) -> tuple[int, int]:
    s = min(1.0, mx / max(w, h))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


# ------------------------------------------------------------------ preview
@app.get("/health")
def health():
    return {"ok": True, "app": "SE Matte", "version": __version__, "device": DEV.to_dict(),
            "busy": JOBS.running is not None, "queued": sum(j.state == "queued" for j in JOBS.jobs.values()),
            "qualities": list(QUALITY),
            "fast": _fast_ok()}


_fast_state = {"t": 0.0, "ok": False}


def _fast_ok() -> bool:
    """Fast mode usable here (checked at most every 30 s: /health is polled every few seconds)."""
    if time.time() - _fast_state["t"] > 30:
        from . import pipeline_fast

        _fast_state.update(ok=pipeline_fast.available()[0], t=time.time())
    return _fast_state["ok"]


@app.get("/probe")
def probe_ep(path: str):
    info = _probe(path)
    return {"path": path, "width": info.width, "height": info.height, "fps": float(info.fps),
            "fps_str": info.fps_str, "frames": info.nb_frames}


_frame_cache: "OrderedDict[tuple, bytes]" = OrderedDict()   # recent preview JPEGs (scrubbing back and forth)
_frame_lock = threading.Lock()


@app.get("/frame")
def frame_ep(path: str, frame: int = 0, max: int = 960):  # noqa: A002
    info = _probe(path)
    frame = int(np.clip(frame, 0, info.nb_frames - 1))
    key = (path, os.path.getmtime(path), frame, max)
    with _frame_lock:
        data = _frame_cache.get(key)
        if data is not None:
            _frame_cache.move_to_end(key)
    if data is None:
        data = None
        if _fast_ok():
            try:   # hardware decode in the warm Fast-mode helper (no ffmpeg start per frame)
                from . import pipeline_fast

                data = pipeline_fast.frame_jpeg(path, frame, max)
            except Exception as e:  # noqa: BLE001
                print(f"helper frame failed ({e}); using ffmpeg", flush=True)
        if data is None:
            w, h = _fit(info.width, info.height, max)
            img = read_frame(info, frame, scale=(w, h))
            data = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        with _frame_lock:
            _frame_cache[key] = data
            while len(_frame_cache) > 48:
                _frame_cache.popitem(last=False)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class SegmentReq(BaseModel):
    path: str
    frame: int
    points: list[list[float]]   # source pixel coords
    labels: list[int]
    max: int = 960


class _Seg:
    """One SAM 2 image model kept warm for click previews (CPU while a job owns the GPU)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.models: dict[str, Any] = {}
        self.frames: dict[str, np.ndarray] = {}

    def get(self, on_gpu: bool):
        from .models.sam2_seg import Sam2Segmenter

        kind = "gpu" if on_gpu else "cpu"
        if kind not in self.models:
            dev = DEV if on_gpu else pick_device("cpu")
            self.models[kind] = Sam2Segmenter(dev)
        return self.models[kind]

    def free(self, kind: str | None = None):
        with self.lock:
            for k in [kind] if kind else list(self.models):
                self.models.pop(k, None)
            self.frames.clear()
        gc.collect()
        empty_cache(DEV)


SEG = _Seg()


@app.post("/segment")
def segment_ep(r: SegmentReq):
    info = _probe(r.path)
    if not r.points:
        raise HTTPException(400, "No click points.")
    key = f"{r.path}|{os.path.getmtime(r.path)}|{r.frame}"
    with SEG.lock:
        on_gpu = JOBS.running is None
        if on_gpu and "cpu" in SEG.models:
            SEG.models.pop("cpu")
        seg = SEG.get(on_gpu)
        if key not in SEG.frames:
            SEG.frames.clear()
            SEG.frames[key] = read_frame(info, r.frame)
        t0 = time.time()
        m = seg.segment(SEG.frames[key], [tuple(p) for p in r.points], r.labels, image_key=key)
    w, h = _fit(info.width, info.height, r.max)
    small = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", small)
    return {"mask": base64.b64encode(buf.tobytes()).decode(), "width": w, "height": h,
            "coverage": float(m.mean()), "ms": int((time.time() - t0) * 1000)}


# ------------------------------------------------------------------ jobs
class KeyframeReq(BaseModel):
    frame: int
    points: list[list[float]]
    labels: list[int]


class RefineReq(BaseModel):
    choke: float = 0.0
    feather: float = 0.0
    contrast: float = 0.0
    temporal: float = 0.0
    decontaminate: bool = True


class JobReq(BaseModel):
    source: str
    start: int = 0
    end: int | None = None            # exclusive
    keyframes: list[KeyframeReq] = []
    mode: str = "pro"                 # pro = clicks + MatAnyone/ViTMatte · fast = people, no clicks
    quality: str = "standard"
    refine: RefineReq = RefineReq()
    outputs: list[str] = ["prores4444"]
    out_dir: str | None = None
    meta: dict = {}                   # opaque data the panel wants back (comp / layer ids)


class Cancelled(Exception):
    pass


# rough share of total time per stage on an M4 (matte 2.4 fps, edges 1.2 fps, export 8 fps)
STAGE_SPAN = {"ai": (0.0, 0.02), "all": (0.02, 1.0), "fast": (0.0, 1.0), "matte": (0.02, 0.32),
              "edges": (0.32, 0.92), "export": (0.92, 1.0)}


@dataclass
class Job:
    id: str
    req: JobReq
    state: str = "queued"            # queued | running | done | error | cancelled
    stage: str = ""
    progress: float = 0.0
    message: str = "Waiting…"
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel: bool = False

    def view(self) -> dict:
        return {"id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 4),
                "message": self.message, "result": self.result, "error": self.error, "meta": self.req.meta,
                "source": self.req.source, "start": self.req.start, "end": self.req.end,
                "created": self.created, "started": self.started, "finished": self.finished}


class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.running: Job | None = None
        threading.Thread(target=self._worker, daemon=True, name="sematte-worker").start()

    def add(self, req: JobReq) -> Job:
        j = Job(uuid.uuid4().hex[:12], req)
        self.jobs[j.id] = j
        self.q.put(j.id)
        return j

    def _worker(self):
        while True:
            j = self.jobs.get(self.q.get())
            if j is None or j.cancel:
                continue
            self.running = j
            j.state, j.started = "running", time.time()
            SEG.free("gpu")  # give the GPU memory to the job
            try:
                j.result = self._run(j)
                j.state, j.progress, j.message = "done", 1.0, "Done"
            except Cancelled:
                j.state, j.message = "cancelled", "Cancelled"
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                j.state, j.error, j.message = "error", f"{type(e).__name__}: {e}", "Failed"
            finally:
                j.finished = time.time()
                self.running = None
                SEG.free("cpu")
                gc.collect()
                empty_cache(DEV)

    def _run(self, j: Job) -> dict:
        r = j.req
        info = _probe(r.source)
        end = min(r.end or info.nb_frames, info.nb_frames)
        start = max(0, min(r.start, end - 1))

        def progress(stage, p, msg):
            if j.cancel:
                raise Cancelled()
            a, b = STAGE_SPAN.get(stage, (j.progress, j.progress))
            j.stage, j.message = stage, msg
            j.progress = max(j.progress, a + (b - a) * float(p))

        if r.mode == "fast":
            from . import pipeline_fast

            res = pipeline_fast.run(pipeline_fast.FastJob(
                source=r.source, start=start, end=end, quality=r.quality,
                refine=RefineSettings(**r.refine.model_dump()), out_dir=r.out_dir or None), progress)
            res.update({"start": start, "end": end, "fps": float(info.fps)})
            return res

        job = ProJob(
            source=r.source, start=start, end=end,
            keyframes=[Keyframe(k.frame, [tuple(p) for p in k.points], list(k.labels)) for k in r.keyframes],
            quality=r.quality if r.quality in QUALITY else "standard",
            refine=RefineSettings(**r.refine.model_dump()), fusion=FusionSettings(),
            outputs=tuple(o for o in r.outputs if o in ("prores4444", "matte", "png")) or ("prores4444",),
            out_dir=r.out_dir or None,
        )
        res = run(job, DEV, progress)
        res.update({"start": start, "end": end, "fps": float(info.fps)})
        return res


JOBS = JobQueue()


@app.post("/jobs")
def add_job(r: JobReq):
    _probe(r.source)
    if r.mode == "fast":
        from . import pipeline_fast

        ok, why = pipeline_fast.available()
        if not ok:
            raise HTTPException(501, why)
    elif not any(k.points for k in r.keyframes):
        raise HTTPException(400, "Click on the subject first.")
    return JOBS.add(r).view()


class FastPreviewReq(BaseModel):
    path: str
    frame: int
    quality: str = "standard"
    refine: RefineReq = RefineReq()
    max: int = 960


@app.post("/fastpreview")
def fast_preview(r: FastPreviewReq):
    """What Fast mode will keep on this frame (people), as a mask PNG for the panel overlay."""
    from . import pipeline_fast

    info = _probe(r.path)
    ok, why = pipeline_fast.available()
    if not ok:
        raise HTTPException(501, why)
    frame = int(np.clip(r.frame, 0, info.nb_frames - 1))
    try:
        res = pipeline_fast.preview(r.path, frame, r.quality, RefineSettings(**r.refine.model_dump()), r.max)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    return {"mask": res["data"], "coverage": res.get("coverage", 1.0), "ms": res.get("ms", 0)}


@app.get("/jobs")
def list_jobs():
    return [j.view() for j in sorted(JOBS.jobs.values(), key=lambda j: j.created)]


@app.get("/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS.jobs:
        raise HTTPException(404, "Unknown job")
    return JOBS.jobs[jid].view()


@app.post("/jobs/{jid}/cancel")
def cancel_job(jid: str):
    j = JOBS.jobs.get(jid)
    if j is None:
        raise HTTPException(404, "Unknown job")
    j.cancel = True
    if j.state == "queued":
        j.state, j.message, j.finished = "cancelled", "Cancelled", time.time()
    return j.view()


@app.post("/shutdown")
def shutdown():
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"ok": True}


# dev only: serve the panel so its UI can be tried in a normal browser
_panel = paths.ENGINE_DIR.parent / "panel"
if _panel.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/dev/panel", StaticFiles(directory=str(_panel), html=True), name="panel")


def _idle_watch():
    while True:
        time.sleep(30)
        busy = JOBS.running is not None or not JOBS.q.empty()
        if not busy and time.time() - _last_request > IDLE_MINUTES * 60:
            print("SE Matte engine idle – stopping.", flush=True)
            os._exit(0)


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(prog="semat.server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SEMATTE_PORT", DEFAULT_PORT)))
    args = ap.parse_args(argv)
    threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"SE Matte engine {__version__} on http://127.0.0.1:{args.port}  device={DEV.to_dict()}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
