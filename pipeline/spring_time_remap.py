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


def _ffmpeg() -> str:
    import os
    import shutil

    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).is_file():
        return env
    which = shutil.which("ffmpeg")
    return which or "ffmpeg"


def write_video(path: Path, frames, fps: float) -> None:
    """Write frames as browser-playable H.264/yuv420p mp4 via ffmpeg.

    OpenCV's default ``mp4v`` (MPEG-4 Part 2) is widely rejected by HTML5
    video elements, which broke click-to-play of the timed action clip.

    Accepts BGR (3ch) or BGRA (4ch). Alpha is composited onto neutral gray
    (RGB 200) for H.264 browser previews — same as videoBGremoval preview.mp4
    — so idle_nobg and action_timed look consistent in the combined player.
    """
    import os
    import subprocess

    h, w = frames[0].shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    # Flatten alpha → gray (200) for browser H.264 (match bgremove preview).
    PREVIEW_BG = 200.0
    bgr_frames = []
    for frame in frames:
        if frame.ndim == 3 and frame.shape[2] == 4:
            bgr = frame[:, :, :3].astype(np.float32)
            a = frame[:, :, 3:4].astype(np.float32) / 255.0
            flat = bgr * a + PREVIEW_BG * (1.0 - a)
            bgr_frames.append(flat.round().astype(np.uint8))
        else:
            bgr_frames.append(frame[:, :, :3] if frame.ndim == 3 else frame)

    # Intermediate mp4v is reliable for OpenCV writers; re-encode for browsers.
    tmp = path.with_suffix(path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open video writer for: {tmp}")
    try:
        for frame in bgr_frames:
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


def probe_has_alpha(path: Path) -> bool:
    """True if ffprobe reports alpha_mode or yuva / rgba pixel format."""
    import json
    import subprocess

    path = Path(path)
    if not path.is_file():
        return False
    # Extension hint first
    if path.suffix.lower() in (".webm", ".mov"):
        try:
            r = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=pix_fmt:stream_tags=alpha_mode",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if r.returncode == 0 and r.stdout:
                data = json.loads(r.stdout)
                for st in data.get("streams") or []:
                    pix = (st.get("pix_fmt") or "").lower()
                    tags = st.get("tags") or {}
                    if "yuva" in pix or "rgba" in pix or "argb" in pix or "bgra" in pix:
                        return True
                    if str(tags.get("alpha_mode", "")).strip() in ("1", "true", "yes"):
                        return True
        except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
            # Fall through to extension-only for webm
            if path.suffix.lower() == ".webm":
                return True
    return False


def read_video_rgba(path: Path):
    """Decode video to BGRA frames, preserving alpha when present.

    VP9+alpha WebM **must** use libvpx-vp9 (native decoder drops alpha).
    """
    import json
    import subprocess

    path = Path(path)
    ffmpeg = _ffmpeg()
    # Probe size / fps
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    st = json.loads(probe.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    cmd = [ffmpeg, "-y"]
    if path.suffix.lower() == ".webm":
        cmd.extend(["-c:v", "libvpx-vp9"])
    cmd.extend(
        [
            "-i",
            str(path.resolve()),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ]
    )
    r = subprocess.run(cmd, capture_output=True, check=False)
    if r.returncode != 0 or not r.stdout:
        # Fallback: OpenCV BGR (no alpha)
        frames, fps2 = read_video(path)
        return frames, fps2, False

    raw = r.stdout
    frame_bytes = w * h * 4
    n = len(raw) // frame_bytes
    if n < 1:
        frames, fps2 = read_video(path)
        return frames, fps2, False

    frames = []
    for i in range(n):
        chunk = raw[i * frame_bytes : (i + 1) * frame_bytes]
        rgba = np.frombuffer(chunk, dtype=np.uint8).reshape((h, w, 4))
        # RGBA → BGRA for OpenCV-style sample_frame blending
        bgra = rgba[:, :, [2, 1, 0, 3]].copy()
        frames.append(bgra)
    return frames, float(fps), True


def write_video_vp9_alpha(
    path: Path,
    frames_bgra,
    fps: float,
    *,
    lossless: bool = True,
    crf: int = 12,
) -> None:
    """Write BGRA frames as high-quality VP9 WebM with alpha (libvpx-vp9).

    Uses a temporary RGBA PNG sequence (reliable + no pipe under-run), then
    encodes with either VP9 lossless or low-CRF CQ. Default is **lossless** so
    time-remap / CapCut intermediates do not accumulate generation loss.
    """
    import shutil
    import subprocess
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgra:
        raise ValueError("no frames to write")
    h, w = frames_bgra[0].shape[:2]
    ffmpeg = _ffmpeg()

    tmp_root = Path(tempfile.mkdtemp(prefix="vp9_alpha_"))
    try:
        # 1) Lossless RGBA PNG frames (no generation loss before encode)
        for i, fr in enumerate(frames_bgra):
            if fr.ndim == 2:
                bgr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                a = np.full((h, w), 255, dtype=np.uint8)
            elif fr.shape[2] == 4:
                bgr = fr[:, :, :3]
                a = fr[:, :, 3]
            else:
                bgr = fr[:, :, :3]
                a = np.full((h, w), 255, dtype=np.uint8)
            # BGR + A → BGRA for imwrite
            bgra = np.dstack([bgr, a])
            png = tmp_root / f"f_{i:05d}.png"
            if not cv2.imwrite(str(png), bgra):
                raise RuntimeError(f"failed to write {png}")

        # 2) Encode from PNG sequence — prefer lossless alpha for intermediates
        seq = str(tmp_root / "f_%05d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            seq,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-row-mt",
            "1",
            "-threads",
            "4",
            "-an",
        ]
        if lossless:
            cmd.extend(["-lossless", "1"])
        else:
            # Constant-quality mode (b:v 0 + crf). Lower crf = better.
            cmd.extend(
                [
                    "-b:v",
                    "0",
                    "-crf",
                    str(max(0, min(63, int(crf)))),
                    "-quality",
                    "best",
                    "-cpu-used",
                    "1",
                ]
            )
        cmd.append(str(path.resolve()))
        print(f"[time_remap] vp9 alpha: {' '.join(cmd)}", flush=True)
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if r.returncode != 0 or not path.is_file() or path.stat().st_size < 1000:
            tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
            # Fallback: try CQ if lossless failed (some ffmpeg builds)
            if lossless:
                print("[time_remap] lossless failed, retry CQ crf=10", flush=True)
                write_video_vp9_alpha(
                    path, frames_bgra, fps, lossless=False, crf=10
                )
                return
            raise RuntimeError(
                f"VP9 alpha write failed (exit {r.returncode}, "
                f"size={path.stat().st_size if path.is_file() else 0}):\n{tail}"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def time_remap_file(
    inp: Path,
    out: Path,
    b=0.42,
    d=4.2,
    f=2.4,
    t=1.15,
    sampling="blend",
    *,
    out_webm: Path | None = None,
    prefer_alpha: bool = True,
) -> dict:
    """Read input, spring time-remap, write browser mp4 (and optional alpha webm).

    When the source has real alpha (VP9 webm / ProRes / etc.), remaps BGRA and:
      - ``out``: H.264 preview (alpha flattened on gray 200, same as bgremove)
      - ``out_webm``: VP9+alpha if provided (or sibling ``.webm`` of out)

    Returns dict: has_alpha, out, out_webm (path|None).
    """
    inp = Path(inp)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    has_alpha = prefer_alpha and probe_has_alpha(inp)
    if has_alpha:
        frames, fps, ok_alpha = read_video_rgba(inp)
        has_alpha = ok_alpha
    else:
        frames, fps = read_video(inp)

    # Alpha cutouts: nearest avoids soft double-edge ghosts that look like quality loss.
    use_sampling = sampling
    if has_alpha and sampling == "blend":
        use_sampling = "nearest"

    n = len(frames)
    src_indices = remap_indices(n, fps, b, d, f, t)
    remapped = [sample_frame(frames, s, use_sampling) for s in src_indices]

    write_video(out, remapped, fps)

    webm_path = None
    if has_alpha and remapped and remapped[0].ndim == 3 and remapped[0].shape[2] == 4:
        webm_path = Path(out_webm) if out_webm else out.with_suffix(".webm")
        try:
            if webm_path.is_file():
                try:
                    webm_path.unlink()
                except OSError:
                    pass
            write_video_vp9_alpha(webm_path, remapped, fps, lossless=True)
        except Exception as exc:
            print(f"[time_remap] alpha webm failed: {exc}", flush=True)
            webm_path = None

    return {
        "has_alpha": bool(webm_path and Path(webm_path).is_file() and Path(webm_path).stat().st_size > 1000),
        "out": out,
        "out_webm": webm_path if webm_path and Path(webm_path).is_file() else None,
        "sampling": use_sampling,
    }
