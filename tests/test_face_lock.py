import numpy as np
import cv2
from pathlib import Path

from pipeline.face_lock import (
    soft_ellipse_mask,
    lower_face_roi,
    paste_lower_face,
    lock_mouth_in_video,
)


def test_soft_mask_peaks_center():
    m = soft_ellipse_mask(40, 60)
    assert m.shape == (40, 60)
    assert float(m[20, 30]) > 0.9
    assert float(m[0, 0]) < 0.2


def test_paste_lower_face_changes_mouth_band():
    base = np.zeros((200, 160, 3), dtype=np.uint8)
    base[:] = (30, 30, 30)
    # face0 region white mouth band
    f0 = base.copy()
    face = (40, 20, 80, 100)
    rx, ry, rw, rh = lower_face_roi(face)
    f0[ry : ry + rh, rx : rx + rw] = (0, 0, 255)
    # dst has green mouth
    dst = base.copy()
    dst[ry : ry + rh, rx : rx + rw] = (0, 255, 0)
    out = paste_lower_face(dst, f0, face, face, strength=1.0)
    # center of mouth band should be closer to red (from f0) than pure green
    cy, cx = ry + rh // 2, rx + rw // 2
    b, g, r = out[cy, cx]
    assert int(r) > int(g)


def test_lock_mouth_in_video_roundtrip(tmp_path):
    path = Path(tmp_path) / "clip.mp4"
    w, h, n, fps = 128, 192, 8, 12.0
    # synthetic face-ish blob moves slightly, mouth color changes over time
    frames = []
    for i in range(n):
        fr = np.zeros((h, w, 3), dtype=np.uint8)
        fr[:] = (20, 20, 20)
        # face rectangle
        x, y, fw, fh = 34, 20 + i, 60, 70
        fr[y : y + fh, x : x + fw] = (180, 160, 150)
        # mouth band changes color — lock should freeze to frame0 pink
        my = y + int(fh * 0.55)
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        fr[my : y + fh, x + 8 : x + fw - 8] = color
        frames.append(fr)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for fr in frames:
        wr.write(fr)
    wr.release()
    out = lock_mouth_in_video(path, out=Path(tmp_path) / "locked.mp4", in_place=False)
    assert out.is_file() and out.stat().st_size > 0
