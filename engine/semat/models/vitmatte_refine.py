"""ViTMatte edge refinement at full output resolution.

Takes a coarse alpha, builds a trimap (sure-fg / sure-bg / unknown edge band) and re-solves only
the unknown band. Runs tile-by-tile on tiles that actually contain edges, so cost scales with the
edge length, not the frame size.

License: MIT (hustvl/ViTMatte). Commercially safe.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .. import paths  # noqa: F401
from ..device import DeviceInfo

REPOS = {"small": "hustvl/vitmatte-small-composition-1k", "base": "hustvl/vitmatte-base-composition-1k"}


def make_trimap(alpha: np.ndarray, band: int, guard: np.ndarray | None = None) -> np.ndarray:
    """uint8 trimap: 255 fg, 0 bg, 128 unknown. `band` = half-width of the unknown zone in px."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    fg = cv2.erode((alpha > 0.97).astype(np.uint8), k)
    bg = cv2.erode((alpha < 0.03).astype(np.uint8), k)
    tri = np.full(alpha.shape, 128, np.uint8)
    tri[fg > 0] = 255
    tri[bg > 0] = 0
    if guard is not None:
        # far outside the tracked subject is always background
        far = cv2.dilate(guard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * band + 1, 4 * band + 1)))
        tri[far == 0] = 0
    return tri


class VitMatteRefiner:
    """Batched, tiled ViTMatte with temporal tile reuse.

    * 512 px tiles keep enough context for clean edges (smaller tiles were measurably worse).
    * Tiles are solved in batches (same result, less GPU overhead).
    * reuse=True: a tile whose pixels and trimap have not changed since it was last solved keeps
      that result instead of being solved again. Compared against the frame where the tile was
      solved (not the previous frame), so slow drift can't accumulate; refreshed every `refresh`
      frames anyway. Typical talking-head / interview shots skip most tiles.
    """

    def __init__(self, dev: DeviceInfo, variant: str = "small", tile: int = 512, overlap: int = 64,
                 half: bool = True, batch: int = 4, reuse: bool = False, img_tol: float = 1.5,
                 tri_tol: float = 0.002, refresh: int = 12, backend: str = "auto"):
        from . import vitmatte_coreml

        self.dev, self.variant = dev, variant
        self.tile, self.overlap, self.batch = tile, overlap, batch
        self.dtype = torch.float16 if (half and dev.kind in ("mps", "cuda")) else torch.float32
        self._torch = None
        self.coreml = None
        if backend in ("auto", "coreml") and vitmatte_coreml.available():
            try:
                self.coreml = vitmatte_coreml.CoreMLVitMatte(REPOS[variant], tile)
            except Exception as e:  # noqa: BLE001  (fall back to the GPU path)
                print(f"Core ML ViTMatte unavailable ({e}); using PyTorch.", flush=True)
        self.backend = "coreml" if self.coreml else "torch"
        if self.coreml is None:
            self._load_torch()
        T, O = tile, overlap
        ramp = np.minimum(np.arange(T) + 1, np.arange(T)[::-1] + 1).astype(np.float32)
        self.ramp = np.clip(ramp / O, 0, 1)
        self.reuse, self.img_tol, self.tri_tol, self.refresh = reuse, img_tol, tri_tol, refresh
        self.memo: dict = {}      # (y, x) → (img tile, tri tile, alpha tile, age)
        self.stats = [0, 0]       # solved, reused

    def reset(self):
        """Call between unrelated frame sequences."""
        self.memo.clear()

    def _same(self, key, im, tr) -> bool:
        m = self.memo.get(key)
        if m is None or m[3] >= self.refresh or m[0].shape != im.shape:
            return False
        band = (tr == 128) | (m[1] == 128)
        if (m[1] != tr).mean() > self.tri_tol:
            return False
        d = np.abs(im.astype(np.int16) - m[0].astype(np.int16)).max(2)[band]
        return d.size == 0 or float(d.mean()) < self.img_tol

    def _load_torch(self):
        if self._torch is None:
            from transformers import VitMatteForImageMatting

            self._torch = VitMatteForImageMatting.from_pretrained(REPOS[self.variant]).to(
                self.dev.device, self.dtype).eval()
        return self._torch

    def _infer_batch(self, imgs: list[np.ndarray], tris: list[np.ndarray]) -> np.ndarray:
        """imgs (h,w,3) uint8 / tris (h,w) uint8, all the same size → alphas (B,h,w) float32."""
        h, w = tris[0].shape
        if self.coreml is not None and h == self.tile and w == self.tile:
            x = np.stack(imgs).astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
            t = np.stack(tris).astype(np.float32)[:, None] / 255.0
            return self.coreml(np.ascontiguousarray(np.concatenate([x, t], 1))).astype(np.float32)
        return self._infer_torch(imgs, tris)

    @torch.inference_mode()
    def _infer_torch(self, imgs: list[np.ndarray], tris: list[np.ndarray]) -> np.ndarray:
        h, w = tris[0].shape
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        x = torch.from_numpy(np.stack(imgs)).to(self.dev.device)          # B,h,w,3 uint8
        t = torch.from_numpy(np.stack(tris)).to(self.dev.device)          # B,h,w
        x = x.permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1.0
        t = (t.to(self.dtype) / 255.0)[:, None]
        inp = torch.cat([x, t], 1)
        if ph or pw:
            inp = torch.nn.functional.pad(inp.float(), (0, pw, 0, ph), mode="reflect").to(self.dtype)
        out = self._load_torch()(pixel_values=inp).alphas
        return out[:, 0, :h, :w].float().cpu().numpy()

    def _infer(self, img: np.ndarray, tri: np.ndarray) -> np.ndarray:
        return self._infer_batch([img], [tri])[0]

    def refine(self, img: np.ndarray, tri: np.ndarray) -> np.ndarray:
        """img RGB uint8 (H,W,3), tri uint8 (H,W). Returns alpha float32 (H,W)."""
        H, W = tri.shape
        out = (tri == 255).astype(np.float32)
        unknown = tri == 128
        if not unknown.any():
            return out
        T, O = min(self.tile, H, W), self.overlap
        step = T - O
        boxes = []
        for y0 in range(0, max(H - O, 1), step):
            for x0 in range(0, max(W - O, 1), step):
                y1, x1 = min(y0 + T, H), min(x0 + T, W)
                ys, xs = max(0, y1 - T), max(0, x1 - T)  # keep full tile size at borders
                if unknown[ys:y1, xs:x1].any() and (ys, xs) not in boxes:
                    boxes.append((ys, xs))
        acc = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        wt = np.outer(self.ramp[:T], self.ramp[:T]) if T == self.tile else np.ones((T, T), np.float32)
        results, todo = {}, []
        for key in boxes:
            y, x = key
            im, tr = img[y:y + T, x:x + T], tri[y:y + T, x:x + T]
            if self.reuse and self._same(key, im, tr):
                m = self.memo[key]
                self.memo[key] = (m[0], m[1], m[2], m[3] + 1)
                results[key] = m[2]
                self.stats[1] += 1
            else:
                todo.append(key)
        for b in range(0, len(todo), self.batch):
            chunk = todo[b:b + self.batch]
            ims = [img[y:y + T, x:x + T] for y, x in chunk]
            trs = [tri[y:y + T, x:x + T] for y, x in chunk]
            for key, im, tr, a in zip(chunk, ims, trs, self._infer_batch(ims, trs)):
                results[key] = a
                if self.reuse:
                    self.memo[key] = (im.copy(), tr.copy(), a, 0)
            self.stats[0] += len(chunk)
        for (y, x), a in results.items():
            acc[y:y + T, x:x + T] += a * wt
            wsum[y:y + T, x:x + T] += wt
        solved = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), out)
        out[unknown] = np.clip(solved[unknown], 0, 1)
        return out
