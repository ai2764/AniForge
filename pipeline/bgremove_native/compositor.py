from __future__ import annotations

import numpy as np


def _alpha3(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    return np.clip(a, 0.0, 1.0)


def composite_frame(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = _alpha3(alpha)
    fg = np.asarray(rgb, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_rgba(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    a8 = (a * 255.0).astype(np.uint8)
    return np.dstack([np.asarray(fg, dtype=np.uint8), a8])
