"""Pose-anchored generate for sitting / lying images.

Standing remains on pipeline.generate (SOMA, no pin). Sitting/lying extract a
3D pose from the image (HMR2), pin it across all Kimodo frames, render smplx22
skeleton guides, and drive SCAIL.

VRAM: Phase1 (HMR) + Phase2 (Kimodo) run as subprocesses so models unload on
exit. Phase3 (SCAIL) uses the caller's ComfyClient (comfy-managed).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.generate import (
    DEFAULT_IDLE_PROMPT,
    IDLE_MOTION_KEEP,
    align_motion_to_base_pose,
    dampen_idle_joints,
    prepare_idle_source_motion,
    FPS,
    JOINT_SPRING,
    SCAIL_ACTION_POSITIVE,
    SCAIL_IDLE_POSITIVE,
    TIME_SPRING,
    _output_size,
    _pad_to_aspect,
    align_4k1,
    ensure_mouth_still,
    plan_steps,
    sanitize_action,
)
from pipeline.skeleton_spring import spring_follow
from pipeline.scail import drive_character
from pipeline.spring_time_remap import time_remap_file

HERE = Path(__file__).resolve().parent
PYEXE = os.environ.get("MP_PYEXE", os.environ.get("COMFY_PYEXE", "")) or None


# smplx22 guide renderer (front XY; mirrors skeleton_spring style)
_PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_SP = (100, 200, 100)
_LA = (50, 150, 255)
_RA = (255, 100, 100)
_LL = (100, 100, 255)
_RL = (255, 50, 200)
_GRP = {
    3: _SP, 6: _SP, 9: _SP, 12: _SP, 15: _SP,
    13: _LA, 16: _LA, 18: _LA, 20: _LA,
    14: _RA, 17: _RA, 19: _RA, 21: _RA,
    1: _LL, 4: _LL, 7: _LL, 10: _LL,
    2: _RL, 5: _RL, 8: _RL, 11: _RL,
}
_BON = [(i, _PAR[i]) for i in range(1, 22)]
_COL = [_GRP[i] for i in range(1, 22)]


def skeleton_camera_from_joints(joints, margin: float = 1.3) -> dict:
    """Orthographic camera (cx, cy, sc) from joints [J,3] or [T,J,3] (front XY)."""
    P = np.asarray(joints, dtype=np.float64)
    if P.ndim == 2:
        P = P[None, ...]
    x, y = P[:, :, 0], P[:, :, 1]
    cx = float(0.5 * (x.min() + x.max()))
    cy = float(0.5 * (y.min() + y.max()))
    sc = float(max(float(x.max() - x.min()), float(y.max() - y.min()), 0.1) * margin)
    return {"cx": cx, "cy": cy, "sc": sc}


def render_smplx_guide(
    P: np.ndarray,
    path: Path,
    size: int = 512,
    fps: float = 30.0,
    camera: dict | None = None,
) -> None:
    """Render [T,22,3] posed_joints to browser-playable H.264 mp4 + PNG still (front XY).

    PNG is written next to the mp4 (same stem) so the UI can show a skeleton even
    if the browser rejects the video codec.

    ``camera`` optional dict {cx, cy, sc}. When set (e.g. from extract pose), every
    clip in a session shares the same framing so frame-0 matches extract still.
    """
    from pipeline.spring_time_remap import write_video

    P = np.asarray(P, dtype=np.float64)
    if camera is not None:
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        sc = float(camera["sc"])
    else:
        cam = skeleton_camera_from_joints(P)
        cx, cy, sc = cam["cx"], cam["cy"], cam["sc"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in range(P.shape[0]):
        img = np.ones((size, size, 3), np.uint8) * 240

        def px(p):
            return (
                int((p[0] - cx) / sc * size + size / 2),
                int(size / 2 - (p[1] - cy) / sc * size),
            )

        for (a, b), c in zip(_BON, _COL):
            cv2.line(img, px(P[f, a]), px(P[f, b]), c, 5, cv2.LINE_AA)
        for j in range(22):
            cv2.circle(img, px(P[f, j]), 4, (50, 50, 50), -1, cv2.LINE_AA)
        frames.append(img)
    # Always write a still for UI preview (BGR)
    png_path = path.with_suffix(".png")
    cv2.imwrite(str(png_path), frames[0])
    write_video(path, frames, fps)


def _pyexe() -> str:
    if PYEXE and Path(PYEXE).is_file():
        return PYEXE
    # Prefer the comfy-scail env where HMR2 / Kimodo deps live.
    cand = Path(r"C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe")
    if cand.is_file():
        return str(cand)
    import sys
    return sys.executable


def _run_subprocess(name: str, args: list[str]) -> None:
    print(f"[anchored] {name}: {' '.join(args)}", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    r = subprocess.run([_pyexe(), *args], env=env)
    # Pause so CUDA contexts can tear down after process exit (esp. Windows).
    # Kimodo/HMR leave VRAM busy for a few seconds after the PID dies.
    pause = 5.0 if "kimodo" in name.lower() or "phase1" in name.lower() or "hmr" in name.lower() else 2.0
    time.sleep(pause)
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {r.returncode}) after {time.time()-t0:.0f}s")
    print(f"[anchored] {name} ok in {time.time()-t0:.0f}s", flush=True)
    # Best-effort: free ComfyUI VRAM (wait until it actually drops).
    try:
        from pipeline.comfy import ComfyClient
        fre = ComfyClient().free_vram(
            interrupt=False, clear_queue=False, wait_s=25, min_free_gb=8.0
        )
        print(f"[anchored] ComfyUI free_vram after {name}: {fre}", flush=True)
    except Exception as exc:
        print(f"[anchored] ComfyUI free_vram skipped: {exc}", flush=True)


def generate_anchored(
    image: Path,
    action_prompt: str,
    idle_prompt,
    overshoot: set,
    run_dir: Path,
    client,
    *,
    pose_mode: str = "sitting",
    seed: int | None = None,
    scale: float = 1.0,
    duration: float = 3.0,
    n_frames: int = 90,
    comfy_input: Path = Path("C:/Users/AIBOX/dev/ComfyUI-scail/input"),
) -> dict:
    """Pose-anchored generate. pose_mode in {standing, sitting, lying}.

    All modes: HMR extract + pin pelvis (Hips) only — limbs free.
    Action prompt is required; idle falls back to DEFAULT_IDLE_PROMPT.
    """
    pose_mode = (pose_mode or "sitting").strip().lower()
    if pose_mode not in ("standing", "sitting", "lying"):
        raise ValueError(f"pose_mode must be standing|sitting|lying, got {pose_mode!r}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image = Path(image)
    plan = plan_steps(overshoot or set())
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    result = {
        "idle": None,
        "action": None,
        "errors": {},
        "seed": seed,
        "pose_mode": pose_mode,
    }

    idle_text = ensure_mouth_still((idle_prompt or "").strip() or DEFAULT_IDLE_PROMPT)
    # Action prompt is user-authored only — no product default (mouth still is enforced).
    if not (action_prompt or "").strip():
        raise ValueError("action_prompt is required (no default)")
    action_text = sanitize_action(action_prompt)

    constraint_path = run_dir / "constraint.json"
    job_path = run_dir / "kimodo_job.json"
    lock_mode = pose_mode  # sitting | lying

    # Phase 1 — HMR extract + grounded pin / full lock
    try:
        _run_subprocess("phase1_extract", [
            str(HERE / "phase1_extract.py"),
            str(image),
            str(constraint_path),
            str(n_frames),
            lock_mode,
        ])
    except Exception as exc:
        result["errors"]["pose_extract"] = str(exc)
        return result

    # Phase 2 — standalone Kimodo (idle + action)
    job = {
        "constraint_json": str(constraint_path),
        "outdir": str(run_dir),
        "seed": seed,
        "duration": duration,
        "steps": 100,
        "jobs": [
            {"name": "idle", "prompt": idle_text},
            {"name": "action", "prompt": action_text},
        ],
    }
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _run_subprocess("gen_kimodo_standalone", [
            str(HERE / "gen_kimodo_standalone.py"),
            str(job_path),
        ])
    except Exception as exc:
        result["errors"]["kimodo"] = str(exc)
        return result

    idle_npz = run_dir / f"idle_seed{seed}.npz"
    action_npz = run_dir / f"action_seed{seed}.npz"
    if not idle_npz.is_file() and not action_npz.is_file():
        result["errors"]["kimodo"] = "no NPZ produced"
        return result

    out_w, out_h = _output_size(image, scale=scale)
    result["size"] = [out_w, out_h]
    result["scale"] = scale

    # Phase 3 — skeleton guide + SCAIL
    for label, npz_path, positive in (
        ("idle", idle_npz, SCAIL_IDLE_POSITIVE),
        ("action", action_npz, SCAIL_ACTION_POSITIVE),
    ):
        if not npz_path.is_file():
            result["errors"][f"{label}_npz"] = f"missing {npz_path.name}"
            continue
        try:
            P = np.load(npz_path)["posed_joints"]
            extract_pose = run_dir / "extract_pose.npy"
            base = np.load(extract_pose) if extract_pose.is_file() else None
            if label == "idle":
                P = prepare_idle_source_motion(P)
                P = dampen_idle_joints(P, keep=IDLE_MOTION_KEEP, base_pose=base)
                np.savez(npz_path, posed_joints=P)
            elif label == "action":
                # Extract pose strength = 100%: full stick at frame 0.
                P = align_motion_to_base_pose(P, base, keep=1.0)
                if plan["joint"]:
                    P = spring_follow(
                        P.astype(np.float64), FPS,
                        omega=JOINT_SPRING["omega"],
                        zeta=JOINT_SPRING["zeta"],
                        soft_scale=JOINT_SPRING["soft"],
                    )
                    if base is not None:
                        P = align_motion_to_base_pose(P, base, keep=1.0)
                np.savez(npz_path, posed_joints=P)
            skel = run_dir / f"{label}_skel.mp4"
            guide = run_dir / f"{label}_guide.mp4"
            cam = skeleton_camera_from_joints(base) if base is not None else None
            render_smplx_guide(P, skel, camera=cam)
            _pad_to_aspect(skel, guide, out_w, out_h)
            out_mp4 = run_dir / f"{label}.mp4"
            drive_character(
                client, guide, image, out_mp4,
                length=align_4k1(P.shape[0]),
                width=out_w, height=out_h,
                prefix=f"mp_{pose_mode}_{label}",
                seed=seed,
                positive=positive,
                comfy_input=comfy_input,
            )
            if label == "action" and plan["time"]:
                try:
                    timed = run_dir / "action_timed.mp4"
                    time_remap_file(out_mp4, timed, **TIME_SPRING)
                    out_mp4 = timed
                except Exception as exc:
                    result["errors"]["time_remap"] = str(exc)
            result[label] = out_mp4
        except Exception as exc:
            result["errors"][label] = str(exc)

    return result
