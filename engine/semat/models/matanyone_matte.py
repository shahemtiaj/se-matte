"""MatAnyone wrapper: first-frame mask → soft, temporally stable alpha, streamed frame by frame.

License: MatAnyone is S-Lab License 1.0 (non-commercial). Internal use only; swap for a commercial build.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import cv2
import numpy as np
import torch

from .. import paths  # noqa: F401  (sets HF_HOME + vendor sys.path)
from ..device import DeviceInfo

REPO_ID = "PeiqingYang/MatAnyone"


class MatAnyoneMatter:
    def __init__(self, dev: DeviceInfo, internal_size: int = 1080, warmup: int = 10):
        from matanyone.model.matanyone import MatAnyone

        self.dev = dev
        self.internal_size = internal_size
        self.warmup = warmup
        # releasing Metal's buffer cache every frame is only needed at high res (it blew up to 17 GB
        # at 1080p); at 720p every 4th frame is safe (≈3.4 GB) and 12% faster
        self.flush_every = 1 if internal_size > 800 else 4
        self.network = MatAnyone.from_pretrained(REPO_ID).to(dev.device).eval()

    def _new_core(self):
        from matanyone.inference.inference_core import InferenceCore

        core = InferenceCore(self.network, cfg=self.network.cfg, device=self.dev.device)
        core.max_internal_size = -1  # we resize ourselves (its built-in resize breaks single-object masks)
        return core

    def work_size(self, w: int, h: int) -> tuple[int, int]:
        """AI resolution: short side = internal_size, rounded to multiples of 16."""
        s = self.internal_size / min(w, h)
        if s >= 1:
            return w, h
        return max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)

    @staticmethod
    def prepare_mask(mask: np.ndarray, r: int = 10) -> np.ndarray:
        """Same preprocessing MatAnyone was trained with: dilate then erode the binary mask."""
        m = (mask > 0).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        m = cv2.dilate(m, k)
        m = cv2.erode((m >= 1).astype(np.float32), k)
        return m * 255.0

    @torch.inference_mode()
    def run(self, frames: Iterable[np.ndarray], first_mask: np.ndarray) -> Iterator[np.ndarray]:
        """frames: RGB uint8 (H,W,3), first one matches first_mask.

        Yields alpha float32 in [0,1] at the AI work size (see work_size); the caller upscales.
        """
        core = self._new_core()
        dev = self.dev.device
        H, W = first_mask.shape[:2]
        ww, wh = self.work_size(W, H)
        m = first_mask.astype(np.uint8)
        if (ww, wh) != (W, H):
            m = cv2.resize(m, (ww, wh), interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(self.prepare_mask(m, r=max(3, int(10 * ww / W)))).to(dev)
        objects = [1]

        def to_t(f):
            if f.shape[1] != ww or f.shape[0] != wh:
                f = cv2.resize(f, (ww, wh), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(np.ascontiguousarray(f)).to(dev).permute(2, 0, 1).float() / 255.0

        it = iter(frames)
        first = next(it)
        img = to_t(first)
        core.step(img, mask_t, objects=objects)
        prob = core.step(img, first_frame_pred=True)
        for _ in range(self.warmup):  # repeat first frame so the matte settles before output
            prob = core.step(img, first_frame_pred=True)
        yield core.output_prob_to_mask(prob).float().cpu().numpy().clip(0, 1)

        for n, f in enumerate(it):
            prob = core.step(to_t(f))
            out = core.output_prob_to_mask(prob).float().cpu().numpy().clip(0, 1)
            # Metal keeps freed buffers cached and fragments at high res → release them regularly
            if self.dev.kind == "mps" and n % self.flush_every == 0:
                torch.mps.empty_cache()
            yield out

        del core
