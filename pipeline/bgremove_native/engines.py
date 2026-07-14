"""Factory for native background-removal matting engines."""

from __future__ import annotations

from pipeline.bgremove_native import SUPPORTED_MODELS


def create_engine(
    model_name: str,
    *,
    fp16: bool = True,
    infer_long_edge: int | None = None,
    alpha_shrink: int = 0,
    alpha_feather: int = 0,
):
    """Create a matting engine without importing its heavyweight dependencies."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {SUPPORTED_MODELS}")

    if model_name == "RMBG-2.0 HQ":
        from pipeline.bgremove_native.rmbg_engine import RMBGEngine

        return RMBGEngine(
            fp16=fp16,
            infer_long_edge=infer_long_edge,
            alpha_shrink=alpha_shrink,
            alpha_feather=alpha_feather,
        )

    from pipeline.bgremove_native.rvm_engine import RVMEngine

    variant = "mobilenetv3" if model_name == "RVM MobileNetV3" else "resnet50"
    return RVMEngine(
        variant=variant,
        fp16=fp16,
        infer_long_edge=infer_long_edge,
        alpha_shrink=alpha_shrink,
        alpha_feather=alpha_feather,
    )
