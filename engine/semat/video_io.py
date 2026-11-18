"""Video decode/encode through ffmpeg pipes (no temp frames on disk).

- Reads any format ffmpeg understands, optionally a trimmed range.
- Writes ProRes 4444 with alpha, matte-only, PNG sequences, and a lossless alpha cache (FFV1).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    """Bundled ffmpeg (engine/bin/<os>/) first, then PATH, then Homebrew."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    here = Path(__file__).resolve().parent.parent / "bin" / sys.platform / exe
    if here.exists():
        return str(here)
    found = shutil.which(exe)
    if found:
        return found
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        cand = Path(p) / exe
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"{name} not found. Install ffmpeg or place it in engine/bin/{sys.platform}/")


@dataclass
class VideoInfo:
    path: str
    width: int          # as displayed (rotation applied, like After Effects and ffmpeg show it)
    height: int
    fps: Fraction
    duration: float
    nb_frames: int
    rotation: int = 0   # 0 / 90 / 180 / 270 (phone videos are often stored sideways + a rotation flag)

    @property
    def fps_str(self) -> str:
        return f"{self.fps.numerator}/{self.fps.denominator}"


def _rotation(s: dict) -> int:
    rot = 0
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and str((s.get("tags") or {}).get("rotate", "")).lstrip("-").isdigit():
        rot = int(s["tags"]["rotate"])
    return rot % 360


_probe_cache: dict[tuple, VideoInfo] = {}


def probe(path: str) -> VideoInfo:
    """Video size / fps / frame count (cached per file version)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _probe_cache:
        return _probe_cache[key]
    out = subprocess.check_output([
        ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:"
        "stream_side_data=rotation:stream_tags=rotate:format=duration",
        "-of", "json", path,
    ], text=True)
    j = json.loads(out)
    if not j.get("streams"):
        raise ValueError("no video stream")
    s = j["streams"][0]
    fps = Fraction(s.get("r_frame_rate") or s.get("avg_frame_rate") or "25/1")
    if fps <= 0:
        fps = Fraction(25)
    duration = float(s.get("duration") or j.get("format", {}).get("duration") or 0)
    nb = int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else int(round(duration * float(fps)))
    w, h, rot = int(s["width"]), int(s["height"]), _rotation(s)
    if rot in (90, 270):
        w, h = h, w
    info = VideoInfo(path, w, h, fps, duration, nb, rot)
    _probe_cache[key] = info
    return info


def _range_args(info: VideoInfo, start_frame: int, num_frames: int | None) -> tuple[list[str], list[str]]:
    pre, post = [], []
    if start_frame > 0:
        # accurate seek (-ss before -i decodes from the previous keyframe and drops frames before the
        # target). Aim half a frame early so a timestamp rounded down by the container still counts.
        pre = ["-ss", f"{(start_frame - 0.5) / float(info.fps):.6f}"]
    if num_frames is not None:
        post = ["-frames:v", str(num_frames)]
    return pre, post


def read_frames(info: VideoInfo, start_frame: int = 0, num_frames: int | None = None,
                scale: tuple[int, int] | None = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (H, W, 3). Optional resize to `scale` = (w, h)."""
    w, h = scale or (info.width, info.height)
    pre, post = _range_args(info, start_frame, num_frames)
    vf = ["-vf", f"scale={w}:{h}:flags=area"] if scale else []
    cmd = [ffmpeg_bin(), "-v", "error", *pre, "-i", info.path, *vf, *post,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_frame(info: VideoInfo, frame: int, scale: tuple[int, int] | None = None) -> np.ndarray:
    for f in read_frames(info, frame, 1, scale):
        return f
    raise RuntimeError(f"Can't decode frame {frame} of {os.path.basename(info.path)}")


class FrameWriter:
    """Pipe frames into ffmpeg.

    kinds:
      - "prores4444": RGBA → ProRes 4444 with alpha (.mov)
      - "matte":      gray → ProRes 422 HQ black/white matte (.mov)
      - "png":        RGBA → PNG sequence (path should contain %05d)
      - "cache":      gray → lossless FFV1 (.mkv), used for the alpha cache
    """

    def __init__(self, path: str, kind: str, width: int, height: int, fps: Fraction):
        self.kind, self.w, self.h = kind, width, height
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        in_fmt = "rgba" if kind in ("prores4444", "png") else "gray"
        base = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_fmt,
                "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        if kind == "prores4444":
            enc = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                   "-alpha_bits", "16", "-vendor", "apl0", "-qscale:v", "9"]
        elif kind == "matte":
            enc = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
        elif kind == "png":
            enc = ["-c:v", "png", "-pix_fmt", "rgba"]
        elif kind == "cache":
            enc = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray"]
        else:
            raise ValueError(kind)
        self.bpp = 4 if in_fmt == "rgba" else 1
        self.proc = subprocess.Popen(base + enc + [path], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert frame.dtype == np.uint8 and frame.shape[0] == self.h and frame.shape[1] == self.w
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="ignore")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-800:]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_cache(path: str, width: int, height: int) -> Iterator[np.ndarray]:
    """Read an alpha cache back as gray uint8 (H, W)."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
