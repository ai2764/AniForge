"""Spring time-remap: warp a motion video's playback timeline with a damped
spring (overshoot-and-settle), without touching any joints or pixels.

Pipeline context: HY-Motion produces an idle motion; scail2 drives a neutral
body (mannequin) with that motion to make the "body motion" clip; this script
adds the spring bounce to that body-motion clip; scail2 then maps the character
art onto the sprung body motion for the final live2d-style portrait.

The remap only resamples frames in time. For the first T seconds the output
progress is warped by a damped sinusoid so the video races ahead of the target,
overshoots, and springs back; after T it plays at normal (identity) time.

    K    = T * fps                       # spring window, in frames
    p    = i / K                         # normalized progress in the window
    warp = p + B * exp(-D * p) * sin(2*pi*F * p)
    src  = warp * K       if i < K       # sample this (fractional) source frame
    src  = i              otherwise      # normal time

Output frame i is a linear blend of source frames floor(src) and floor(src)+1.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def remap_indices(n_frames: int, fps: float, b: float, d: float, f: float, t: float):
    """Return, for each output frame, the fractional source-frame index to sample.

    `n_frames` output frames are produced (same count as the source). Frames
    inside the spring window (i < T*fps) follow the damped-spring warp; the rest
    map identically (normal time).
    """
    k = t * fps  # spring window length in frames
    src = []
    for i in range(n_frames):
        if i < k and k > 0:
            p = i / k
            warp = p + b * math.exp(-d * p) * math.sin(2.0 * math.pi * f * p)
            s = warp * k
        else:
            s = float(i)
        src.append(max(0.0, min(n_frames - 1, s)))
    return src


def remap_indices_monotonic(n_frames: int, snap: float):
    """Strictly increasing time warp: every source frame is played once, in order,
    with a fast approach and a decelerating settle. No frame is ever replayed
    (no positional overshoot) and the duration is not padded.

    `snap` (>1) biases speed toward the start: higher = quicker reach, longer
    settle tail. snap=1 is linear (identity).
    """
    src = []
    for i in range(n_frames):
        u = i / (n_frames - 1) if n_frames > 1 else 0.0
        s = 1.0 - (1.0 - u) ** snap        # ease-out: fast start, slow settle
        src.append(s * (n_frames - 1))
    return src


def sample_frame(frames, src_float: float, sampling: str = "blend"):
    """Pick the source frame for `src_float`: linear blend, or nearest frame.

    `nearest` avoids double-exposure ghosts on hard-edged content such as
    skeleton guide videos; `blend` is smoother for natural video.
    """
    n = len(frames)
    lo = int(math.floor(src_float))
    hi = min(lo + 1, n - 1)
    blend = src_float - lo
    if sampling == "nearest":
        return frames[hi] if blend >= 0.5 else frames[lo]
    if blend <= 0.0:
        return frames[lo]
    a = frames[lo].astype(np.float32)
    c = frames[hi].astype(np.float32)
    return ((1.0 - blend) * a + blend * c).round().astype(np.uint8)


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open input video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from: {path}")
    return frames, float(fps)


def write_video(path: Path, frames, fps: float) -> None:
    """Write frames as browser-playable H.264/yuv420p mp4 via ffmpeg.

    OpenCV's default ``mp4v`` (MPEG-4 Part 2) is widely rejected by HTML5
    video elements, which broke click-to-play of the timed action clip.
    """
    import os
    import subprocess

    h, w = frames[0].shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")

    # Intermediate mp4v is reliable for OpenCV writers; re-encode for browsers.
    tmp = path.with_suffix(path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open video writer for: {tmp}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(tmp),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                "-movflags", "+faststart",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def time_remap_file(inp: Path, out: Path, b=0.42, d=4.2, f=2.4, t=1.15, sampling="blend"):
    """Read input mp4, compute spring time-remap, sample each output frame, write output mp4.

    Args:
        inp: Path to input video file
        out: Path to output video file
        b: Spring amplitude (overshoot magnitude), default 0.42
        d: Damping decay (larger = settles faster), default 4.2
        f: Oscillation frequency (bounces), default 2.4
        t: Spring window in seconds, default 1.15
        sampling: Frame sampling method - 'blend' (smooth) or 'nearest' (no ghosting), default 'blend'
    """
    frames, fps = read_video(inp)
    n = len(frames)
    src_indices = remap_indices(n, fps, b, d, f, t)
    remapped = [sample_frame(frames, s, sampling) for s in src_indices]

    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(out, remapped, fps)
