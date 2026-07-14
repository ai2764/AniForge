"""RMBG-2.0 native matting engine."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.bgremove_native.base import MattingEngine

_BUNDLED_MODEL_DIR = Path(__file__).with_name("models") / "RMBG-2.0"
_HF_MODEL_ID = "briaai/RMBG-2.0"


def resolve_model_source() -> str:
    """Return the configured, bundled, or Hugging Face RMBG model source."""
    configured_model_dir = os.environ.get("RMBG_MODEL_DIR")
    if configured_model_dir:
        return configured_model_dir
    if _BUNDLED_MODEL_DIR.is_dir():
        return str(_BUNDLED_MODEL_DIR)
    return _HF_MODEL_ID


class RMBGEngine(MattingEngine):
    """RMBG-2.0 (BiRefNet) per-frame matting with temporal smoothing."""

    def __init__(
        self,
        *,
        fp16: bool = True,
        infer_long_edge: int | None = None,
        alpha_shrink: int = 0,
        alpha_feather: int = 0,
    ) -> None:
        super().__init__(alpha_shrink=alpha_shrink, alpha_feather=alpha_feather)
        self.temporal_smoothing = 0.2
        self.fp16 = fp16
        self.infer_long_edge = infer_long_edge
        self.model = None
        self.device: str | None = None
        self._model_size = (1024, 1024)

    def load(self, device: str = "cuda") -> None:
        from transformers import AutoModelForImageSegmentation

        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            resolve_model_source(), trust_remote_code=True
        ).to(device).eval()
        if self.fp16 and "cuda" in device:
            self.model = self.model.half()

    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from torchvision.transforms.functional import normalize

        height, width = rgb_frame.shape[:2]
        with torch.no_grad():
            image = Image.fromarray(rgb_frame).resize(self._model_size, Image.BILINEAR)
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            tensor = tensor.unsqueeze(0).to(self.device)
            if self.fp16 and "cuda" in self.device:
                tensor = tensor.half()
            alpha = self.model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()

        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
        return self._postprocess(np.array(alpha_image).astype(np.float32) / 255.0)

    def unload(self) -> None:
        if self.model is not None:
            import torch

            del self.model
            self.model = None
            self.reset()
            torch.cuda.empty_cache()
