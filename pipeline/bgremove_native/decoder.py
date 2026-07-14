from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoDecoder:
    """Extract RGB frames from a video file using OpenCV."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {self.path}")
        try:
            self.info = VideoInfo(
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
        finally:
            cap.release()

        self.width = self.info.width
        self.height = self.info.height
        self.fps = self.info.fps
        self.frame_count = self.info.frame_count

    def frames(self) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
