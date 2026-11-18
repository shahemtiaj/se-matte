"""Clean the MatAnyone matte, then let ViTMatte re-solve the edge band at full resolution.

MatAnyone defines WHAT the subject is (it follows the keyframe mask, incl. held objects).
Here we only:
  * remove stray islands  – small blobs not connected to the subject (background specks)
  * fill pinholes         – tiny fully-enclosed holes inside the subject
  * smooth edge shimmer   – light temporal smoothing inside the ViTMatte edge band only

(v2 tried using SAM 2 tracking to veto "leaks"; it removed real held objects during fast motion,
so it is no longer used here. Attached leaks are fixed with correction keyframes instead.)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FusionSettings:
    island_keep: float = 0.08   # keep blobs at least this share of the main subject's area
    island_min_px: int = 2500   # ...or at least this many px (at 1080p)
    hole_max_px: int = 600      # fill enclosed holes up to this size (at 1080p)
    band: int = 10              # half-width of ViTMatte unknown band (px at 1080p)
    edge_smooth: float = 0.35   # temporal smoothing inside the edge band (0..1)


class Fuser:
    def __init__(self, s: FusionSettings | None = None):
        self.s = s or FusionSettings()
        self.prev_final: np.ndarray | None = None

    def coarse(self, mat: np.ndarray, guard: np.ndarray | None = None) -> np.ndarray:
        s = self.s
        H, W = mat.shape
        area_scale = (H / 1080.0) ** 2
        a = mat.copy()

        fg = (a > 0.04).astype(np.uint8)  # include faint specks, not just solid blobs
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 2:
            areas = st[1:, cv2.CC_STAT_AREA]
            main = areas.max()
            thr = min(main * s.island_keep, s.island_min_px * area_scale)
            drop = [j + 1 for j, ar in enumerate(areas) if ar < thr]
            if drop:
                rm = np.isin(lab, drop)
                rm = cv2.dilate(rm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (a < 0.98)
                rm |= np.isin(lab, drop)
                a[rm] = 0.0

        bg = (a < 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
        if n > 2:
            for j in range(1, n):
                x, y, w, h, ar = st[j]
                touches = x == 0 or y == 0 or x + w >= W or y + h >= H
                if not touches and ar <= s.hole_max_px * area_scale:
                    a[lab == j] = 1.0
        return a

    def finish(self, refined: np.ndarray, tri: np.ndarray) -> np.ndarray:
        """Temporal smoothing only inside the unknown band (kills per-frame edge shimmer)."""
        k = self.s.edge_smooth
        if k > 0 and self.prev_final is not None and self.prev_final.shape == refined.shape:
            band = tri == 128
            d = np.abs(refined - self.prev_final)
            w = k * np.exp(-d / 0.1)  # follow real motion, smooth tiny jitter
            refined = np.where(band, w * self.prev_final + (1 - w) * refined, refined)
        self.prev_final = refined.astype(np.float32)
        return self.prev_final
