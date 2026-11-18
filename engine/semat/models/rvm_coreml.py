"""Robust Video Matting (RVM) as Core ML, for Fast mode on Macs.

RVM is a real-time, recurrent video matting network for people: true soft alpha (hair), and frame-to-frame
stability. (Edge colour comes from blur-fusion in the native helper: RVM's own is too soft.) Converted per video size
(the recurrent state shapes depend on it), compiled once and cached in engine/models/coreml/.

  resnet50, detail 1.5: ~48 fps at 1080p on the M4 Neural Engine; detail 2: ~27 fps, finest hair
  Lower detail and mobilenetv3 are faster but keep background objects (e.g. a doorway next to the
  head) and give mushy edges – not production quality, so neither is used.

Format 3 ("low-res" model): the model only does RVM's low-res pass. Input `sm` is the frame already
downsampled (PyTorch bilinear, done natively), output `ab` is the deep guided filter's A and b for alpha at
low res. The full-res step (alpha = A·mean(RGB) + b, A/b upsampled bilinearly) runs in the native helper on
the CPU in parallel with the next prediction. RVM's foreground colour is not computed at all (blur-fusion is
used instead). Same alpha as the full model (mean difference 0.003/255), ~1.6× less Neural Engine time.
I/O is plain fp16 tensors; the native helper fills and reads them with vImage / its own loops.

License: RVM is GPL-3.0 – fine for internal use; swap before selling (see PLAN.md §9).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .. import paths

COREML_DIR = paths.MODELS_DIR / "coreml"
WEIGHTS = paths.MODELS_DIR / "rvm"
VARIANTS = ("mobilenetv3", "resnet50")


def downsample_ratio(w: int, h: int, detail: float = 1.0) -> float:
    """RVM's internal low-res pass works best at ~256–512 px on the long side; `detail` 2 doubles it
    (finer hair and fewer stray background bits, about half the speed)."""
    m = max(w, h)
    base = 0.5 if m <= 1024 else 0.25 if m <= 2048 else 0.125 if m <= 4096 else 0.0625
    return min(1.0, base * detail)


FORMAT = 3  # bump when the model I/O changes


def compiled_path(variant: str, w: int, h: int, detail: float = 1.0) -> Path:
    tag = "" if detail == 1.0 else f"-d{int(round(detail * 100))}"
    return COREML_DIR / f"rvm{FORMAT}-{variant}-{w}x{h}{tag}.mlmodelc"


def ensure(variant: str, w: int, h: int, detail: float = 1.0) -> Path:
    """Compiled Core ML model for this variant and frame size (converted on first use, ~5 s)."""
    out = compiled_path(variant, w, h, detail)
    if out.exists():
        return out
    pkg = out.with_suffix(".mlpackage")
    if not pkg.exists():
        convert(variant, w, h, pkg, detail)
    import coremltools as ct

    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)  # just to compile
    tmp = out.with_suffix(".tmp.mlmodelc")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(m.get_compiled_model_path(), tmp)
    tmp.rename(out)
    shutil.rmtree(pkg, ignore_errors=True)   # only the compiled model is needed from here on
    return out


def convert(variant: str, w: int, h: int, out: Path, detail: float = 1.0) -> Path:
    import sys

    import coremltools as ct
    import torch
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as cops

    sys.path.insert(0, str(paths.VENDOR_DIR / "RobustVideoMatting"))
    from model import MattingNetwork  # RVM's own code (vendor)

    weights = WEIGHTS / f"rvm_{variant}.pth"
    if not weights.exists():
        raise RuntimeError(f"RVM weights missing: {weights}")

    orig_cast = cops._cast

    def _cast(context, node, dtype, dtype_name):  # numpy 2 + coremltools
        x = cops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.reshape(-1)[0].item()), name=node.name), node.name)
            return
        return orig_cast(context, node, dtype, dtype_name)

    ds = downsample_ratio(w, h, detail)
    net = MattingNetwork(variant).eval()
    net.load_state_dict(torch.load(weights, map_location="cpu"))

    class Low(torch.nn.Module):
        """RVM's forward up to the guided filter's low-res A, b (alpha channel only)."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sm, r1, r2, r3, r4):
            m = self.m
            f1, f2, f3, f4 = m.backbone(sm)
            f4 = m.aspp(f4)
            hid, r1, r2, r3, r4 = m.decoder(sm, f1, f2, f3, f4, r1, r2, r3, r4)
            fres, pha = m.project_mat(hid).split([3, 1], dim=-3)
            rf = m.refiner
            bx = torch.cat([sm, sm.mean(1, keepdim=True)], 1)
            by = torch.cat([fres, pha], 1)
            mx, my = rf.box_filter(bx), rf.box_filter(by)
            cov = rf.box_filter(bx * by) - mx * my
            var = rf.box_filter(bx * bx) - mx * mx
            A = rf.conv(torch.cat([cov, var, hid], 1))
            b = my - A * mx
            return torch.cat([A[:, 3:4], b[:, 3:4]], 1), r1, r2, r3, r4

    src = torch.rand(1, 3, h, w)
    sm = torch.nn.functional.interpolate(src, scale_factor=ds, mode="bilinear", align_corners=False,
                                         recompute_scale_factor=False)
    with torch.no_grad():
        _, _, *rec = net(src, None, None, None, None, downsample_ratio=ds)
        rec = [torch.zeros_like(r) for r in rec]
        traced = torch.jit.trace(Low(net).eval(), (sm, *rec), check_trace=False)
    cops._cast = _cast
    try:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="sm", shape=sm.shape, dtype=np.float16)]
            + [ct.TensorType(name=f"r{i + 1}i", shape=r.shape, dtype=np.float16) for i, r in enumerate(rec)],
            outputs=[ct.TensorType(name="ab", dtype=np.float16)]
            + [ct.TensorType(name=f"r{i + 1}o", dtype=np.float16) for i in range(4)],
            compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram")
    finally:
        cops._cast = orig_cast
    ml.user_defined_metadata["ds"] = repr(ds)   # the native helper rebuilds PyTorch's exact sampling from it
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    ml.save(str(tmp))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out
