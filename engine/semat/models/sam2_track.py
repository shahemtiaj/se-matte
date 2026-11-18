"""SAM 2.1 video tracking → robust per-frame "guard" mask of the subject.

The guard decides WHAT is the subject in every frame (no background chunks, no lost limbs);
MatAnyone + ViTMatte decide the soft edge. Frames are loaded lazily from small JPEGs so memory stays
low even for long clips on 16 GB machines. License: Apache-2.0.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from typing import Callable, Iterator

import cv2
import numpy as np
import torch

from .. import paths  # noqa: F401
from ..device import DeviceInfo
from ..video_io import VideoInfo, ffmpeg_bin
from .sam2_seg import VARIANTS


class _LazyFrames:
    """Stand-in for SAM 2's preloaded frame tensor: decodes JPEGs on demand, keeps a small LRU."""

    def __init__(self, folder: str, n: int, size: int, keep: int = 8):
        self.folder, self.n, self.size, self.keep = folder, n, size, keep
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        img = cv2.imread(os.path.join(self.folder, f"{i:06d}.jpg"))[..., ::-1]
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        self.cache[i] = t
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return t


class Sam2Tracker:
    def __init__(self, dev: DeviceInfo, variant: str = "base-plus", half: bool = True):
        os.environ.setdefault("TQDM_DISABLE", "1")
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2_video_predictor

        repo, fname, cfg = VARIANTS[variant]
        self.dev = dev
        self.dtype = torch.bfloat16 if (half and dev.kind in ("mps", "cuda")) else torch.float32
        self.pred = build_sam2_video_predictor(cfg, hf_hub_download(repo, fname), device=str(dev.device))
        self.size = self.pred.image_size

    def _extract(self, info: VideoInfo, start: int, n: int, folder: str) -> None:
        cmd = [ffmpeg_bin(), "-v", "error", "-y"]
        if start > 0:
            cmd += ["-ss", f"{start / float(info.fps):.6f}"]
        cmd += ["-i", info.path, "-frames:v", str(n), "-vf", f"scale={self.size}:{self.size}:flags=area",
                "-q:v", "2", "-start_number", "0", os.path.join(folder, "%06d.jpg")]
        subprocess.run(cmd, check=True)

    def track(self, *a, **kw):
        ctx = (torch.autocast(self.dev.kind, dtype=self.dtype) if self.dtype != torch.float32
               else torch.autocast("cpu", enabled=False))
        with torch.inference_mode(), ctx:
            yield from self._track(*a, **kw)

    def _track(self, info: VideoInfo, start: int, end: int, keyframes, masks: dict | None = None,
               progress: Callable[[float], None] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (absolute_frame, bool mask HxW) for every frame in [start, end), in any order.

        keyframes: objects with .frame, .points, .labels (absolute frame numbers, source pixels).
        """
        import sam2.sam2_video_predictor as svp

        svp.tqdm = lambda it, *a, **k: it  # we report our own progress
        n = end - start
        tmp = tempfile.mkdtemp(prefix="sematte_sam2_")
        try:
            self._extract(info, start, n, tmp)
            n = len([f for f in os.listdir(tmp) if f.endswith(".jpg")])
            frames = _LazyFrames(tmp, n, self.size)
            orig = svp.load_video_frames
            svp.load_video_frames = lambda **kw: (frames, info.height, info.width)
            try:
                state = self.pred.init_state(video_path=tmp, offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
            finally:
                svp.load_video_frames = orig

            kfs = sorted(keyframes, key=lambda k: k.frame)
            for k in kfs:
                if masks is not None and k.frame in masks:
                    # full keyframe mask defines the whole subject (incl. held objects) far better than clicks
                    self.pred.add_new_mask(state, frame_idx=k.frame - start, obj_id=1,
                                           mask=torch.from_numpy(masks[k.frame].astype(bool)))
                else:
                    self.pred.add_new_points_or_box(
                        state, frame_idx=k.frame - start, obj_id=1,
                        points=np.asarray(k.points, np.float32), labels=np.asarray(k.labels, np.int32))

            first = kfs[0].frame - start
            done = 0
            total = n

            def emit(fi, logits):
                nonlocal done
                done += 1
                if progress:
                    progress(done / total)
                return fi + start, (logits[0, 0] > 0).cpu().numpy()

            for fi, _, logits in self.pred.propagate_in_video(state, start_frame_idx=first):
                yield emit(fi, logits)
                if self.dev.kind == "mps" and done % 8 == 0:
                    torch.mps.empty_cache()
            if first > 0:
                for fi, _, logits in self.pred.propagate_in_video(state, start_frame_idx=first, reverse=True):
                    if fi == first:
                        continue
                    yield emit(fi, logits)
            self.pred.reset_state(state)
            del state
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
