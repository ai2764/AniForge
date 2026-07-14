"""Shared base class for native matting engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class MattingEngine(ABC):
    """Abstract interface and common alpha post-processing."""

    def __init__(self, *, alpha_shrink: int = 0, alpha_feather: int = 0) -> None:
        self.alpha_shrink = alpha_shrink
        self.alpha_feather = alpha_feather
        self.temporal_smoothing = 0.0
        self._prev_alpha: np.ndarray | None = None

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load the model onto the requested device."""

    @abstractmethod
    def predict(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Return an alpha matte for an RGB uint8 frame."""

    def reset(self) -> None:
        self._prev_alpha = None

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    def _postprocess(self, alpha: np.ndarray) -> np.ndarray:
        if self.alpha_shrink != 0:
            radius = abs(int(self.alpha_shrink))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
            operation = cv2.erode if self.alpha_shrink > 0 else cv2.dilate
            alpha = operation(alpha_u8, kernel).astype(np.float32) / 255.0

        if self.alpha_feather > 0:
            radius = int(self.alpha_feather)
            alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), 0)

        if self.temporal_smoothing > 0.0 and self._prev_alpha is not None:
            alpha = (1.0 - self.temporal_smoothing) * alpha + (
                self.temporal_smoothing * self._prev_alpha
            )

        self._prev_alpha = alpha.copy() if self.temporal_smoothing > 0.0 else None
        return alpha
