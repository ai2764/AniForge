"""Robust Video Matting native engine."""

from __future__ import annotations

import numpy as np

from pipeline.bgremove_native.base import MattingEngine


class RVMEngine(MattingEngine):
    """Robust Video Matting engine with temporal recurrence."""

    def __init__(
        self,
        *,
        variant: str = "mobilenetv3",
        fp16: bool = True,
        infer_long_edge: int | None = None,
        alpha_shrink: int = 0,
        alpha_feather: int = 0,
    ) -> None:
        super().__init__(alpha_shrink=alpha_shrink, alpha_feather=alpha_feather)
        if variant not in ("mobilenetv3", "resnet50"):
            raise ValueError(f"Unknown RVM variant: {variant}")
        self.variant = variant
        self.fp16 = fp16
        self.infer_long_edge = infer_long_edge
        self.model = None
        self.device: str | None = None
        self._rec = [None] * 4

    def load(self, device: str = "cuda") -> None:
        import torch

        self.device = device
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting", self.variant, trust_repo=True
        ).to(device).eval()
        if self.fp16 and "cuda" in device:
            self.model = self.model.half()

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        import torch.nn.functional as functional

        height, width = rgb_frame.shape[:2]
        with torch.no_grad():
            source = (
                torch.from_numpy(rgb_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(self.device)
            )
            if self.fp16 and "cuda" in self.device:
                source = source.half()

            if self.infer_long_edge is not None and max(height, width) > self.infer_long_edge:
                scale = self.infer_long_edge / max(height, width)
                resized_height = max(16, int(round(height * scale / 16) * 16))
                resized_width = max(16, int(round(width * scale / 16) * 16))
                source = functional.interpolate(
                    source,
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                )

            _, alpha, *self._rec = self.model(source, *self._rec)
            if alpha.shape[-2:] != (height, width):
                alpha = functional.interpolate(
                    alpha, size=(height, width), mode="bilinear", align_corners=False
                )

        return self._postprocess(alpha[0, 0].float().cpu().numpy())

    def reset(self) -> None:
        super().reset()
        self._rec = [None] * 4

    def unload(self) -> None:
        if self.model is not None:
            import torch

            del self.model
            self.model = None
            self.reset()
            torch.cuda.empty_cache()
