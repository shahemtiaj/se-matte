"""SAM 2.1 image segmenter: click points on one frame → binary mask.

Only runs on keyframes (the frames you click), so memory use stays low on 16 GB machines.
License: Apache-2.0.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import paths  # noqa: F401
from ..device import DeviceInfo

VARIANTS = {
    # name: (hf repo, checkpoint file, config)
    "base-plus": ("facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


class Sam2Segmenter:
    def __init__(self, dev: DeviceInfo, variant: str = "base-plus"):
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        repo, fname, cfg = VARIANTS[variant]
        ckpt = hf_hub_download(repo, fname)
        model = build_sam2(cfg, ckpt, device=str(dev.device))
        self.pred = SAM2ImagePredictor(model)

    @torch.inference_mode()
    def segment(self, frame_rgb: np.ndarray, points, labels, image_key=None) -> np.ndarray:
        """Return a bool mask (H, W) for the clicked subject.

        image_key: pass a stable id (e.g. path+frame) to reuse the image embedding between clicks
        on the same frame (interactive preview). Without it the embedding is freed after each call.
        """
        if image_key is None or image_key != getattr(self, "_key", None):
            self.pred.set_image(frame_rgb)
            self._key = image_key
        pc = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        pl = np.asarray(labels, dtype=np.int32).reshape(-1)
        # one click is ambiguous (shirt vs whole person): ask for 3 candidates, keep the best/largest
        multi = len(pc) == 1
        masks, scores, logits = self.pred.predict(point_coords=pc, point_labels=pl, multimask_output=multi)
        if multi:
            # prefer the whole subject: weight score by area so "full person" beats "just the shirt"
            areas = masks.reshape(len(masks), -1).mean(1)
            best = int(np.argmax(scores * (0.5 + areas / (areas.max() + 1e-6))))
        else:
            best = int(np.argmax(scores))
        # second pass seeded with the previous logits sharpens the boundary
        masks2, _, _ = self.pred.predict(point_coords=pc, point_labels=pl,
                                         mask_input=logits[best][None], multimask_output=False)
        if image_key is None:
            self.pred.reset_predictor()
            self._key = None
        return masks2[0] > 0

    def release(self) -> None:
        self.pred.reset_predictor()
        self._key = None
