"""ViTMatte on Apple's Neural Engine (Core ML). Mac only; other systems use the PyTorch path.

The Neural Engine is a separate chip from the GPU: ViTMatte runs ~2× faster there than on the M4 GPU
(38 vs 79 ms per 512 px tile) and leaves the GPU free for the soft-matte model.

The Core ML model is converted once from the same Hugging Face weights (≈10 s) and cached in
engine/models/coreml/. A fixed tile size makes ViTDet's interpolated position embeddings constants,
which is what makes the conversion possible.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np

from .. import paths

COREML_DIR = paths.MODELS_DIR / "coreml"


def available() -> bool:
    if platform.system() != "Darwin" or os.environ.get("SEMATTE_NO_COREML") == "1":
        return False
    try:
        import coremltools  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def model_path(repo: str, tile: int) -> Path:
    return COREML_DIR / f"{repo.split('/')[-1]}-{tile}.mlpackage"


def convert(repo: str, tile: int) -> Path:
    import coremltools as ct
    import torch
    import transformers.models.vitdet.modeling_vitdet as vd
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as cops
    from transformers import VitMatteForImageMatting

    out = model_path(repo, tile)
    out.parent.mkdir(parents=True, exist_ok=True)

    # numpy 2 + coremltools: casting 1-element arrays needs .item()
    orig_cast = cops._cast

    def _cast(context, node, dtype, dtype_name):
        x = cops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.reshape(-1)[0].item()), name=node.name), node.name)
            return
        return orig_cast(context, node, dtype, dtype_name)

    # fixed tile size → position embeddings are constants: compute eagerly, replay while tracing
    abs_cache, rel_cache = {}, {}
    orig_abs = vd.VitDetEmbeddings.get_absolute_positions
    orig_rel = getattr(vd.get_rel_pos, "__wrapped__", vd.get_rel_pos)

    def abs_pos(self, emb, cls, h, w):
        k = (id(self), int(h), int(w))
        if k not in abs_cache:
            with torch.no_grad():
                abs_cache[k] = orig_abs(self, emb, cls, int(h), int(w)).detach().clone()
        return abs_cache[k]

    def rel_pos(q, k, rp):
        key = (id(rp), int(q), int(k))
        if key not in rel_cache:
            with torch.no_grad():
                rel_cache[key] = orig_rel(int(q), int(k), rp).detach().clone()
        return rel_cache[key]

    saved = (cops._cast, vd.VitDetEmbeddings.get_absolute_positions, vd.get_rel_pos)
    cops._cast, vd.VitDetEmbeddings.get_absolute_positions, vd.get_rel_pos = _cast, abs_pos, rel_pos
    try:
        m = VitMatteForImageMatting.from_pretrained(repo).eval()

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).alphas

        w = Wrap(m).eval()
        x = torch.randn(1, 4, tile, tile)
        with torch.no_grad():
            w(x)  # eager pass fills the constant caches
            traced = torch.jit.trace(w, x, strict=False, check_trace=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)],
                        outputs=[ct.TensorType(name="alpha")], compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS14, convert_to="mlprogram")
        tmp = out.with_suffix(".tmp.mlpackage")
        ml.save(str(tmp))
        if out.exists():
            import shutil
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        cops._cast, vd.VitDetEmbeddings.get_absolute_positions, vd.get_rel_pos = saved
    return out


class CoreMLVitMatte:
    """Same contract as the torch model: (B,4,T,T) normalised float32 → (B,T,T) alpha."""

    def __init__(self, repo: str, tile: int):
        import coremltools as ct

        p = model_path(repo, tile)
        compiled = p.with_suffix(".mlmodelc")
        self.tile = tile
        if compiled.exists():  # fast path: skip the ~10 s compile on every load
            try:
                self.model = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
                return
            except Exception:  # noqa: BLE001  (stale / OS update → recompile)
                import shutil
                shutil.rmtree(compiled, ignore_errors=True)
        if not p.exists():
            convert(repo, tile)
        m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE)
        try:
            import shutil
            shutil.copytree(m.get_compiled_model_path(), compiled, dirs_exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        self.model = m

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return np.stack([self.model.predict({"x": inp[i:i + 1]})["alpha"][0, 0] for i in range(len(inp))])
