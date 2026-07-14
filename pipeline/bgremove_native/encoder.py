from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.bgremove_native.ffmpeg import resolve_ffmpeg


class VideoEncoder:
    """FFmpeg pipe-based video encoder supporting alpha-channel formats."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        fmt: str = "prores",
        ffmpeg_path: str | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = ffmpeg_path or resolve_ffmpeg()
        pix_in, codec_args = _encoder_args(fmt)
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_in,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            *codec_args,
            str(self.output_path),
        ]
        self._expected_channels = 4 if pix_in == "rgba" else 3
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != self._expected_channels:
            raise ValueError(
                f"Expected {self._expected_channels} channels, got {frame.shape[-1] if frame.ndim else 0}"
            )
        if self._proc.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> Path:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr_bytes = self._proc.stderr.read() if self._proc.stderr is not None else b""
        returncode = self._proc.wait()
        if returncode != 0:
            err = stderr_bytes.decode(errors="replace")
            raise RuntimeError(f"FFmpeg failed (rc={returncode}): {err}")
        return self.output_path


def _encoder_args(fmt: str) -> tuple[str, list[str]]:
    if fmt == "prores":
        return "rgba", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    if fmt == "webm":
        return "rgba", [
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "20", "-b:v", "0",
            "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
        ]
    if fmt == "webp":
        return "rgba", [
            "-c:v", "libwebp_anim", "-pix_fmt", "yuva420p", "-lossless", "0", "-quality", "80", "-loop", "0",
        ]
    if fmt == "h264":
        return "rgb24", ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unknown format: {fmt}")


def _write_frames(path: str | Path, frames: Iterable[np.ndarray], fps: float, fmt: str) -> Path:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Cannot encode an empty frame sequence") from None

    height, width = first.shape[:2]
    encoder = VideoEncoder(path, fps, width, height, fmt=fmt)
    try:
        encoder.write_frame(first)
        for frame in iterator:
            encoder.write_frame(frame)
    finally:
        encoder.close()
    return Path(path)


def write_h264_preview(path: str | Path, frames: Iterable[np.ndarray], fps: float) -> Path:
    return _write_frames(path, frames, fps, "h264")


def write_vp9_alpha(path: str | Path, rgba_frames: Iterable[np.ndarray], fps: float) -> Path:
    return _write_frames(path, rgba_frames, fps, "webm")


def extract_audio(input_path: str, output_audio: str, *, ffmpeg_path: str | None = None) -> bool:
    ffmpeg_path = ffmpeg_path or resolve_ffmpeg()
    result = subprocess.run(
        [ffmpeg_path, "-y", "-i", input_path, "-vn", "-acodec", "copy", output_audio],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.isfile(output_audio) and os.path.getsize(output_audio) > 0


def mux_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    ffmpeg_path: str | None = None,
) -> None:
    ffmpeg_path = ffmpeg_path or resolve_ffmpeg()
    result = subprocess.run(
        [ffmpeg_path, "-y", "-i", video_path, "-i", audio_path, "-c", "copy", "-shortest", output_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio mux failed: {result.stderr.decode(errors='replace')}")
