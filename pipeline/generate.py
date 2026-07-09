"""Generate orchestration: one reference image + prompts -> idle.mp4 + action.mp4.

Wires together kimodo (text -> motion NPZ), skeleton_spring (spring overshoot +
skeleton render), scail (character drive), and spring_time_remap (playback
overshoot) into two independent clips. Each clip is produced in its own
try/except so a ComfyUI failure on one branch does not prevent the other
branch's clip from being returned.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from pipeline.kimodo import generate_motion
from pipeline.skeletons import load_posed_joints
from pipeline.skeleton_spring import spring_follow, render
from pipeline.scail import drive_character
from pipeline.spring_time_remap import time_remap_file

# Appended to every Kimodo motion prompt so diffusion does not invent speech.
MOUTH_STILL_CLAUSE = (
    "Mouth closed and still the entire time, lips sealed, silent, "
    "no talking, no speaking, no lip movement, no chewing."
)

# Pose-agnostic: used for standing / sitting / lying (do not say "stand").
# Source motion should have *some* natural idle amplitude; UI keep scales it down.
DEFAULT_IDLE_PROMPT = (
    "A person holds their current pose in a relaxed idle with natural gentle motion: "
    "soft continuous breathing, a slight weight shift, and subtle sway of the torso, "
    "head, and arms. Keep the same overall posture; do not stand up or sit down; "
    "no walking, no waving, no dancing, no large gestures. " + MOUTH_STILL_CLAUSE
)

# Default UI / API keep after extract anchor: extract + keep * (Kimodo deltas).
IDLE_MOTION_KEEP = 0.06
# If Kimodo idle is almost static, boost residual to this RMS so the keep slider works.
IDLE_SOURCE_REF_STD = 0.012
# Action motion amount default: full Kimodo deltas on extract pose.
ACTION_MOTION_KEEP = 1.0
# Default Kimodo clip length for action (seconds @ model fps, usually 30).
ACTION_DURATION_SEC = 2.0
IDLE_DURATION_SEC = 3.0
# After retarget, if upper-body motion is weaker than this RMS, amplify it
# so keep=100% is clearly different from idle micro-motion.
ACTION_UPPER_REF_STD = 0.035

# SCAIL positives for idle/action character drive (body only; face quiet).
SCAIL_IDLE_POSITIVE = (
    "a character in a calm idle pose, full body, consistent identity, "
    "mouth closed, lips sealed, silent, no talking"
)
SCAIL_ACTION_POSITIVE = (
    "a character performing an action, full body, consistent identity, "
    "mouth closed, lips sealed, silent, no talking, no lip sync"
)
# SCAIL negative: template default plus mouth/speech blockers.
SCAIL_NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, static, "
    "talking, speaking, open mouth, mouth open, lip sync, lips moving, "
    "mouth moving, chewing, singing, dialogue, shouting"
)

FPS = 30                      # Kimodo output is 30 fps
JOINT_SPRING = dict(omega=20.0, zeta=0.35, soft=1.0)   # action joint-overshoot defaults
TIME_SPRING = dict(b=0.42, d=4.2, f=2.4, t=1.15)       # action time-overshoot params


def plan_steps(overshoot):
    """overshoot: set[str] of selected post-steps ("joint", "time")."""
    return {"joint": "joint" in overshoot, "time": "time" in overshoot}


def ensure_mouth_still(prompt: str) -> str:
    """Append mouth-still clause unless the prompt already forbids speech."""
    p = (prompt or "").strip()
    low = p.lower()
    if "mouth closed" in low or "lips sealed" in low or "no talking" in low:
        return p
    if not p:
        return MOUTH_STILL_CLAUSE
    return p.rstrip(". ") + ". " + MOUTH_STILL_CLAUSE


def prepare_idle_source_motion(posed_joints, ref_std: float = IDLE_SOURCE_REF_STD):
    """Ensure idle source has usable temporal amplitude for the keep slider.

    If Kimodo returns a near-frozen clip (common with ultra-still prompts),
    amplify residual (or synthesize a mild sway) up toward ``ref_std``.
    Does not change the mean pose shape much; only residual scale.
    """
    import numpy as np

    P = np.asarray(posed_joints, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] < 2:
        return P.astype(np.float64, copy=False)
    mean = P.mean(axis=0, keepdims=True)
    residual = P - mean
    std = float(np.sqrt(np.mean(residual * residual)))
    try:
        target = float(ref_std)
    except (TypeError, ValueError):
        target = IDLE_SOURCE_REF_STD
    target = max(1e-6, target)

    if std < 1e-8:
        # Completely static: gentle sinusoidal sway on central chain.
        T, J, _ = P.shape
        t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
        residual = np.zeros_like(P)
        for j in (0, 3, 6, 9, 12, 15):  # pelvis / spine / neck / head-ish
            if j < J:
                residual[:, j, 0] = 0.012 * np.sin(t)
                residual[:, j, 1] = 0.008 * np.sin(t + 0.7)
                residual[:, j, 2] = 0.006 * np.sin(t + 1.3)
        std = float(np.sqrt(np.mean(residual * residual))) or 1e-8

    # Normalize weak sources up to target RMS so keep=100% is visibly not frozen.
    if std < target:
        residual = residual * (target / std)
    return mean + residual


def dampen_idle_joints(posed_joints, keep: float = IDLE_MOTION_KEEP, base_pose=None):
    """Idle = extract base + keep × (Kimodo deltas from frame 0).

    posed_joints: [T, J, 3]  (preferably after ``prepare_idle_source_motion``)
    base_pose: extract FK [J, 3] — always fully stuck (strength = 100%).
    keep: 0 = frozen extract; 1 = full Kimodo temporal deltas on extract pose.

        P'[t] = base + keep * (P[t] - P[0])
    """
    import numpy as np

    P = np.asarray(posed_joints, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] < 1:
        return P.astype(np.float64, copy=False)
    try:
        k = float(keep)
    except (TypeError, ValueError):
        k = IDLE_MOTION_KEEP
    k = max(0.0, min(1.0, k))

    deltas = P - P[0:1]
    if base_pose is not None:
        base = np.asarray(base_pose, dtype=np.float64).reshape(1, P.shape[1], 3)
    else:
        base = P[0:1]
    if k <= 0.0:
        return np.repeat(base, P.shape[0], axis=0) if base.shape[0] == 1 else base
    if k >= 1.0:
        return base + deltas
    return base + k * deltas


# SMPLX22 lower body (pelvis/legs/feet) — keep seated/lying root from extract.
# Names: pelvis, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle, L_foot, R_foot
SMPLX22_LOWER_BODY = (0, 1, 2, 4, 5, 7, 8, 10, 11)
# Spine + arms + head (everything not lower body)
SMPLX22_UPPER_BODY = tuple(
    i for i in range(22) if i not in SMPLX22_LOWER_BODY
)


def align_motion_to_base_pose(
    posed_joints,
    base_pose,
    keep: float = 1.0,
    *,
    lock_lower_body: bool = False,
    boost_upper: bool = False,
    upper_ref_std: float = ACTION_UPPER_REF_STD,
):
    """Stick frame 0 to ``base_pose`` (extract), scale Kimodo deltas by ``keep``.

        P'[t] = base + keep * (P[t] - P[0])
        P'[0] = base   (hard)

    keep=1: full action deltas. keep=0: frozen base.
    lock_lower_body: pin pelvis+legs to base every frame (sitting/lying).
    boost_upper: if upper-body residual is tiny, amplify toward upper_ref_std
    so keep=100% is visibly different from idle.
    """
    import numpy as np

    P = np.asarray(posed_joints, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] < 1:
        return P.astype(np.float64, copy=False)
    try:
        k = float(keep)
    except (TypeError, ValueError):
        k = 1.0
    k = max(0.0, min(1.0, k))

    if base_pose is None:
        base = P[0].copy()
    else:
        base = np.asarray(base_pose, dtype=np.float64).reshape(P.shape[1], 3).copy()

    # Full deltas first (keep applied after optional upper boost).
    deltas = P - P[0:1]
    out = base[None, ...] + deltas
    out[0] = base

    if boost_upper and out.shape[0] >= 2:
        upper = [j for j in SMPLX22_UPPER_BODY if j < out.shape[1]]
        if upper:
            up_res = out[:, upper, :] - out[0:1, upper, :]
            up_std = float(np.sqrt(np.mean(up_res * up_res)))
            try:
                target = float(upper_ref_std)
            except (TypeError, ValueError):
                target = ACTION_UPPER_REF_STD
            target = max(1e-6, target)
            if up_std < 1e-8:
                # Completely static arms: synthesize a clear raise-ish sway on wrists/shoulders.
                T = out.shape[0]
                t = np.linspace(0.0, np.pi, T)  # go up and slightly down
                for j in (16, 17, 18, 19, 20, 21):  # shoulders, elbows, wrists
                    if j < out.shape[1]:
                        # lift in +Y (up) and slight side
                        side = 0.04 if j % 2 == 0 else -0.04
                        out[:, j, 1] = base[j, 1] + 0.18 * np.sin(t)
                        out[:, j, 0] = base[j, 0] + side * np.sin(t)
                out[0] = base
            elif up_std < target:
                scale = min(target / up_std, 12.0)
                out[:, upper, :] = out[0:1, upper, :] + up_res * scale

    # Apply amount slider on residual from base (frame 0 stays base).
    if k < 1.0:
        out = base[None, ...] + k * (out - base[None, ...])
        out[0] = base
    if lock_lower_body:
        lower = [j for j in SMPLX22_LOWER_BODY if j < out.shape[1]]
        out[:, lower, :] = base[lower, :]
        out[0] = base
    return out


def sanitize_action(prompt):
    """Drop turning/turn words from an action prompt (fixed-frame rendering
    means a body-turn description would fight the camera-locked guide).
    Also soft-strips speech verbs so SCAIL is less likely to animate a talking mouth."""
    p = re.sub(r"\b(turning|turns|turn)\b", "", prompt or "", flags=re.I)
    p = re.sub(
        r"\b(talking|talks|talk|speaking|speaks|speak|saying|says|said|"
        r"whispering|whispers|shouting|shouts|singing|sings|chatting)\b",
        "",
        p,
        flags=re.I,
    )
    p = re.sub(r"\s{2,}", " ", p).strip(" ,.")
    return ensure_mouth_still(p)


def align_4k1(n):
    """SCAIL length must be 4k+1."""
    return ((int(n) - 1) // 4) * 4 + 1 if n > 1 else 1


def _output_size(image_path, long_cap=1280, short_cap=720, mult=16, scale=1.0):
    """Output width/height matching the input image aspect ratio.

    Fits inside a max frame (longer side <= long_cap, shorter <= short_cap),
    then multiplies by ``scale`` (0.25–1.0, default 1.0 = full cap). Rounded to
    a multiple of ``mult`` (SCAIL/VAE stride). Never changes the image aspect.
    """
    from PIL import Image
    with Image.open(image_path) as im:
        w, h = im.size
    fit = min(long_cap / max(w, h), short_cap / min(w, h))
    try:
        s = float(scale)
    except (TypeError, ValueError):
        s = 1.0
    s = max(0.25, min(1.0, s))
    fit *= s
    r = lambda v: max(mult, int(round(v * fit / mult)) * mult)
    return r(w), r(h)


def _pad_to_aspect(inp, out, out_w, out_h, base=512):
    """Pad the square skeleton render to the OUTPUT aspect (out_w:out_h) with
    black bars, so the later SCAIL resize to out_w x out_h does not distort it."""
    import subprocess
    import os
    aspect = out_w / out_h
    if aspect < 1:          # portrait: pad top/bottom
        W, H = base, int(round(base / aspect))
    elif aspect > 1:        # landscape: pad left/right
        W, H = int(round(base * aspect)), base
    else:
        W, H = base, base
    W += W % 2
    H += H % 2
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    subprocess.run([ffmpeg, "-y", "-i", str(inp),
                    "-vf", f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True, capture_output=True)


def _build_guide(client, prompt, npz_path, skel_path, guide_path, *,
                 model, seed, comfy_output, joint, out_w, out_h,
                 dampen_idle=False):
    """Kimodo motion -> optional joint-space spring -> fixed-frame render ->
    aspect-matched pad. Returns (guide_path, scail_length)."""
    npz = generate_motion(client, prompt, npz_path, model=model, seed=seed,
                          comfy_output=comfy_output)
    kpts = load_posed_joints(npz)
    if dampen_idle:
        kpts = dampen_idle_joints(kpts, keep=IDLE_MOTION_KEEP)
    if joint:
        kpts = spring_follow(kpts, FPS, omega=JOINT_SPRING["omega"],
                             zeta=JOINT_SPRING["zeta"], soft_scale=JOINT_SPRING["soft"])
    render(kpts, skel_path, FPS, fixed=True)
    _pad_to_aspect(skel_path, guide_path, out_w, out_h)
    return guide_path, align_4k1(len(kpts))


def generate(image: Path, action_prompt: str, idle_prompt, overshoot: set,
             run_dir: Path, client, *, motion_model="Kimodo-SOMA-RP-v1", seed=None,
             scale=1.0,
             comfy_input=Path("C:/Users/AIBOX/dev/ComfyUI-scail/input"),
             comfy_output=Path("C:/Users/AIBOX/dev/ComfyUI-scail/output")) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image = Path(image)
    plan = plan_steps(overshoot)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    out_w, out_h = _output_size(image, scale=scale)

    result = {"idle": None, "action": None, "errors": {}, "seed": seed,
              "size": [out_w, out_h], "scale": scale}

    # Phase 1: generate BOTH motions -> padded skeleton guides.
    # Doing both Kimodo passes back-to-back keeps its model hot in VRAM instead
    # of swapping Kimodo<->SCAIL between clips.
    idle_guide = action_guide = None
    idle_len = action_len = None
    try:
        idle_guide, idle_len = _build_guide(
            client, ensure_mouth_still(idle_prompt or DEFAULT_IDLE_PROMPT),
            run_dir / "idle.npz", run_dir / "idle_skel.mp4", run_dir / "idle_guide.mp4",
            model=motion_model, seed=seed, comfy_output=comfy_output, joint=False,
            out_w=out_w, out_h=out_h, dampen_idle=True)
    except Exception as exc:
        result["errors"]["idle_motion"] = str(exc)
    try:
        action_guide, action_len = _build_guide(
            client, sanitize_action(action_prompt),
            run_dir / "action.npz", run_dir / "action_skel.mp4", run_dir / "action_guide.mp4",
            model=motion_model, seed=seed, comfy_output=comfy_output, joint=plan["joint"],
            out_w=out_w, out_h=out_h)
    except Exception as exc:
        result["errors"]["action_motion"] = str(exc)

    # Phase 2: drive the character image with BOTH guides (SCAIL model stays hot).
    if idle_guide is not None:
        try:
            result["idle"] = drive_character(
                client, idle_guide, image, run_dir / "idle.mp4",
                length=idle_len, width=out_w, height=out_h, prefix="mp_idle", seed=seed,
                positive=SCAIL_IDLE_POSITIVE,
                comfy_input=comfy_input)
        except Exception as exc:
            result["errors"]["idle_scail"] = str(exc)

    if action_guide is not None:
        try:
            action_path = drive_character(
                client, action_guide, image, run_dir / "action.mp4",
                length=action_len, width=out_w, height=out_h, prefix="mp_action", seed=seed,
                positive=SCAIL_ACTION_POSITIVE,
                comfy_input=comfy_input)
            if plan["time"]:
                try:
                    time_remap_file(action_path, run_dir / "action_timed.mp4", **TIME_SPRING)
                    action_path = run_dir / "action_timed.mp4"
                except Exception as exc:
                    result["errors"]["time_remap"] = str(exc)  # keep the un-timed action_path
            result["action"] = action_path
        except Exception as exc:
            result["errors"]["action_scail"] = str(exc)

    return result
