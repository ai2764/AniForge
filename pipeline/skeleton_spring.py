"""Joint-space spring on Kimodo SOMA keypoints: damped-spring follower per joint,
then render the body-subset skeleton for the scail motion guide.

Vendored from camera-lab/tasks/live2d/skeletons/skeleton_spring.py and adapted
for the Kimodo SOMASkeleton77 body subset (pipeline.skeletons). This module is
self-contained: BONES/COLORS/SOFT/N_JOINTS come from pipeline.skeletons, no
fixed 22/30-joint slicing.

Knobs:
  omega   base stiffness (rad/s-ish); higher = tighter follow, less lag
  zeta    damping ratio; <1 underdamped (overshoot/bounce), 1 = no overshoot
  soft_scale  extra softness multiplier applied to distal joints (lower their omega)
"""
from __future__ import annotations
import numpy as np
import cv2

from pipeline.skeletons import BONES, COLORS, SOFT, N_JOINTS


def spring_follow(target, fps, omega, zeta, soft_scale):
    """Per-joint damped-spring follower. target: [T, J, 3] -> [T, J, 3].

    Explicit integration is only stable while omega*h is small; the substep
    count is chosen from omega so a high omega (near-rigid) does not blow up.
    """
    T, J, _ = target.shape
    dt = 1.0 / fps
    # per-joint angular frequency: distal joints get a LOWER omega (more lag /
    # follow-through) but still high enough to track the gesture. Softening
    # only scales omega down to ~0.55x at most; the overshoot itself comes
    # from zeta<1.
    w = omega * (1.0 - 0.45 * SOFT[:J] * soft_scale)   # [J]
    w = np.maximum(w, 6.0)
    # keep omega*h <= ~0.2 for a stable explicit step (was fixed at 8 substeps,
    # which exploded for large omega and produced garbage side-profile poses)
    sub = max(8, int(np.ceil(w.max() * dt / 0.2)))
    h = dt / sub
    x = target[0].copy()
    v = np.zeros_like(x)
    out = np.empty_like(target)
    out[0] = x
    for i in range(1, T):
        tgt = target[i]
        for _ in range(sub):
            a = (w**2)[:, None] * (tgt - x) - 2.0 * zeta * w[:, None] * v
            v += a * h
            x += v * h
        out[i] = x
    return out


def frame_fixed(all_kpts):
    """One consistent (cx, cy, scale) for the whole sequence so the body stays
    anchored - per-frame recentering makes scail read it as a turning/zooming body."""
    k = all_kpts[:, :N_JOINTS, :]
    x, y = k[:, :, 0], k[:, :, 1]
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    scale = max(x.max() - x.min(), y.max() - y.min(), 0.1) * 1.3
    return cx, cy, scale


def draw(kpts, size=512, frame=None):
    img = np.ones((size, size, 3), np.uint8) * 240
    if frame is None:
        x, y = kpts[:, 0], kpts[:, 1]
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        scale = max(x.max() - x.min(), y.max() - y.min(), 0.1) * 1.3
    else:
        cx, cy, scale = frame

    def px(p):
        return (int((p[0] - cx) / scale * size + size / 2),
                int(size / 2 - (p[1] - cy) / scale * size))

    for (a, b), c in zip(BONES, COLORS):
        p1, p2 = px(kpts[a]), px(kpts[b])
        cv2.line(img, p1, p2, c, 5, cv2.LINE_AA)   # COLORS is already BGR
    for i in range(len(kpts)):
        cv2.circle(img, px(kpts[i]), 4, (50, 50, 50), -1, cv2.LINE_AA)
    return img


def render(kpts, path, fps, size=512, fixed=True):
    frame = frame_fixed(kpts) if fixed else None
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(kpts.shape[0]):
        w.write(draw(kpts[i, :N_JOINTS], size, frame))
    w.release()
