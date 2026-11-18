"""Matte refinement — runs on cached alpha, no AI re-run needed.

All functions take/return float32 arrays in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class RefineSettings:
    choke: float = 0.0        # px; >0 shrinks the matte, <0 expands it
    feather: float = 0.0      # px gaussian softening of the edge
    contrast: float = 0.0     # 0 = as-is, 1 = very crisp edges
    temporal: float = 0.0     # 0..1 flicker smoothing strength
    decontaminate: bool = True  # estimate true foreground colour at the edges

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- edge-aware upscale
def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_upscale(alpha_lr: np.ndarray, guide_rgb: np.ndarray, r: int = 8, eps: float = 1e-4) -> np.ndarray:
    """Fast guided filter: upscale a low-res alpha using the full-res image as edge guide."""
    H, W = guide_rgb.shape[:2]
    h, w = alpha_lr.shape[:2]
    if (h, w) == (H, W):
        return alpha_lr
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g_lr = cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA)
    rr = max(1, int(round(r * w / W)))
    mean_I, mean_p = _box(g_lr, rr), _box(alpha_lr, rr)
    cov = _box(g_lr * alpha_lr, rr) - mean_I * mean_p
    var = _box(g_lr * g_lr, rr) - mean_I * mean_I
    a = cov / (var + eps)
    b = mean_p - a * mean_I
    a = cv2.resize(_box(a, rr), (W, H), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(_box(b, rr), (W, H), interpolation=cv2.INTER_LINEAR)
    # keep confidently solid/empty regions from the plain upscale, use guided result at edges
    up = cv2.resize(alpha_lr, (W, H), interpolation=cv2.INTER_LINEAR)
    gf = np.clip(a * guide + b, 0, 1)
    edge = ((up > 0.02) & (up < 0.98)).astype(np.float32)
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8))
    return up * (1 - edge) + gf * edge


# ---------------------------------------------------------------- per-frame shaping
def shape(alpha: np.ndarray, s: RefineSettings) -> np.ndarray:
    a = alpha
    if s.choke:
        k = int(round(abs(s.choke)))
        if k > 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            a = cv2.erode(a, ker) if s.choke > 0 else cv2.dilate(a, ker)
    if s.feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), s.feather)
    if s.contrast > 0:
        # levels around 0.5: contrast 1 → window of 0.1
        width = 1.0 - 0.9 * min(s.contrast, 1.0)
        a = np.clip((a - 0.5) / width + 0.5, 0, 1)
    return a.astype(np.float32)


class TemporalSmoother:
    """Motion-aware EMA: smooths flicker where the matte barely changes, follows real motion."""

    def __init__(self, strength: float):
        self.k = float(np.clip(strength, 0, 1)) * 0.8
        self.prev: np.ndarray | None = None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        if self.k <= 0 or self.prev is None:
            self.prev = a
            return a
        diff = np.abs(a - self.prev)
        w = self.k * np.exp(-diff / 0.08)  # big change → trust the new frame
        out = w * self.prev + (1 - w) * a
        self.prev = out
        return out.astype(np.float32)


# ---------------------------------------------------------------- foreground colour
def estimate_foreground(img_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blur-fusion foreground estimation (Forte & Pitié, 2021).

    Removes background colour spill from semi-transparent edges (hair, motion blur).
    Returns uint8 RGB.
    """
    I = img_rgb.astype(np.float32) / 255.0
    a = alpha[..., None]
    F, B = I.copy(), I.copy()
    for r in (90, 6):
        F, B = _blur_fusion_step(I, F, B, a, r)
    # only replace where alpha is partial; keep original pixels in solid areas
    mix = ((alpha > 0.01) & (alpha < 0.99))[..., None]
    out = np.where(mix, F, I)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _blur_fusion_step(I, F, B, a, r):
    k = (r, r)
    blur_a = cv2.blur(a[..., 0], k)[..., None]
    blur_Fa = cv2.blur(F * a, k)
    blur_F = blur_Fa / (blur_a + 1e-5)
    blur_B1a = cv2.blur(B * (1 - a), k)
    blur_B = blur_B1a / (cv2.blur(1 - a[..., 0], k)[..., None] + 1e-5)
    F = blur_F + a * (I - a * blur_F - (1 - a) * blur_B)
    return np.clip(F, 0, 1), blur_B


def compose_rgba(img_rgb: np.ndarray, alpha: np.ndarray, decontaminate: bool) -> np.ndarray:
    fg = estimate_foreground(img_rgb, alpha) if decontaminate else img_rgb
    a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([fg, a8])
