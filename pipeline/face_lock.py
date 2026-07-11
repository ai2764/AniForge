"""Freeze lower-face / mouth region across a character video.

SCAIL2/Wan often invents lip motion even with strong anti-speech prompts.
This post-process pastes the first frame's lower face (mouth/jaw) onto every
later frame, warped into the currently detected face box, with a soft mask.

Uses OpenCV Haar face detector (no extra deps).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _CASCADE = cv2.CascadeClassifier(path)
        if _CASCADE.empty():
            raise RuntimeError(f"failed to load face cascade: {path}")
    return _CASCADE


def _ffmpeg() -> str:
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).is_file():
        return env
    import shutil

    return shutil.which("ffmpeg") or "ffmpeg"


def detect_face(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return largest face (x, y, w, h) or None."""
    faces = _cascade().detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(max(32, gray.shape[1] // 12), max(32, gray.shape[0] // 12)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if faces is None or len(faces) == 0:
        return None
    # largest area
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def lower_face_roi(
    face: tuple[int, int, int, int],
    *,
    top_frac: float = 0.42,
    side_inset: float = 0.12,
    bottom_pad: float = 0.08,
) -> tuple[int, int, int, int]:
    """Mouth/jaw band inside a face box (x,y,w,h)."""
    x, y, w, h = face
    y0 = y + int(h * top_frac)
    y1 = y + h + int(h * bottom_pad)
    x0 = x + int(w * side_inset)
    x1 = x + w - int(w * side_inset)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def soft_ellipse_mask(h: int, w: int, feather: float = 0.22) -> np.ndarray:
    """0..1 float mask, ellipse with soft edge."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(h / 2.0, 1.0), max(w / 2.0, 1.0)
    # slightly taller ellipse for chin
    ry *= 1.05
    nx = (xx - cx) / rx
    ny = (yy - cy) / ry
    r = np.sqrt(nx * nx + ny * ny)
    # hard core then feather
    core = 1.0 - feather
    m = np.clip((1.0 - r) / max(feather, 1e-3), 0.0, 1.0)
    m = np.where(r <= core, 1.0, m)
    m = m * m * (3 - 2 * m)  # smoothstep
    return m.astype(np.float32)


def _clip_roi(x, y, w, h, W, H):
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def paste_lower_face(
    dst: np.ndarray,
    src_frame0: np.ndarray,
    face0: tuple[int, int, int, int],
    face_t: tuple[int, int, int, int],
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Warp frame0 lower-face into face_t lower-face on dst (BGR)."""
    H, W = dst.shape[:2]
    rx0, ry0, rw0, rh0 = lower_face_roi(face0)
    rxt, ryt, rwt, rht = lower_face_roi(face_t)

    c0 = _clip_roi(rx0, ry0, rw0, rh0, W, H)
    ct = _clip_roi(rxt, ryt, rwt, rht, W, H)
    if c0 is None or ct is None:
        return dst
    x0, y0, w0, h0 = c0
    xt, yt, wt, ht = ct
    if w0 < 4 or h0 < 4 or wt < 4 or ht < 4:
        return dst

    patch = src_frame0[y0 : y0 + h0, x0 : x0 + w0]
    patch_r = cv2.resize(patch, (wt, ht), interpolation=cv2.INTER_LINEAR)
    mask = soft_ellipse_mask(ht, wt)
    s = float(np.clip(strength, 0.0, 1.0))
    mask = mask * s
    m3 = mask[..., None]
    region = dst[yt : yt + ht, xt : xt + wt].astype(np.float32)
    blended = patch_r.astype(np.float32) * m3 + region * (1.0 - m3)
    out = dst.copy()
    out[yt : yt + ht, xt : xt + wt] = blended.round().astype(np.uint8)
    return out


def lock_mouth_in_video(
    inp: Path,
    out: Path | None = None,
    *,
    strength: float = 1.0,
    in_place: bool = True,
) -> Path:
    """Rewrite video so mouth/lower-face stays as in frame 0.

    Writes H.264 yuv420p via ffmpeg (browser-safe). If ``in_place``, replaces
    ``inp`` after writing a temp file.
    """
    inp = Path(inp)
    if not inp.is_file():
        raise FileNotFoundError(inp)
    if out is None:
        out = inp.with_name(inp.stem + "_mouthlock.mp4")
    out = Path(out)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {inp}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {inp}")

    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    face0 = detect_face(gray0)
    if face0 is None:
        # Center-lower fallback for full-body portraits (face ~ upper third)
        H, W = frames[0].shape[:2]
        fw, fh = int(W * 0.35), int(H * 0.22)
        face0 = ((W - fw) // 2, int(H * 0.08), fw, fh)
        print(f"[face_lock] no face on f0 — using fallback box {face0}", flush=True)
    else:
        print(f"[face_lock] face0={face0} frames={len(frames)}", flush=True)

    locked = [frames[0]]
    last_face = face0
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        face_t = detect_face(gray) or last_face
        last_face = face_t
        locked.append(
            paste_lower_face(frames[i], frames[0], face0, face_t, strength=strength)
        )

    # Write via OpenCV temp + ffmpeg H.264 (same pattern as spring_time_remap)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp.mp4")
    h, w = locked[0].shape[:2]
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer {tmp}")
    try:
        for fr in locked:
            writer.write(fr)
    finally:
        writer.release()

    r = subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        tmp.unlink()
    except OSError:
        pass
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"face_lock encode failed: {(r.stderr or '')[-800:]}")

    if in_place and out.resolve() != inp.resolve():
        # replace original
        bak = inp.with_suffix(inp.suffix + ".pre_mouthlock")
        try:
            if not bak.is_file():
                inp.replace(bak)
            else:
                inp.unlink(missing_ok=True)
        except OSError:
            pass
        # copy out -> inp
        import shutil

        shutil.copy2(out, inp)
        try:
            out.unlink()
        except OSError:
            pass
        return inp
    return out
