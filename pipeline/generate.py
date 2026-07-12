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

# Live2D-style idle (product): short loop, livelier torso, arms/legs locked.
# Pose-neutral wording (no stand/sit/lie) so it fits every pose_mode. No mouth
# clause: the Kimodo skeleton has no jaw/face joints, so mouth wording is inert
# here — mouth control lives in the SCAIL prompts, which render the actual face.
DEFAULT_IDLE_PROMPT = (
    "A calm, continuously moving idle loop. Soft steady breathing lifts and lowers "
    "the chest and upper belly every cycle; the torso and spine gently sway and lean "
    "a little with the breath, and the shoulders rise and ease each time. The head "
    "makes small natural nods and slow side-to-side sways so the body never looks "
    "frozen. Keep every movement gentle, subtle, and continuous, with no large joint "
    "rotations, returning smoothly to a seamless loop."
)

# UI keep: residual from Kimodo on head/torso only (arms/legs locked in post).
# Bumped 0.12 -> 0.20 for more visible torso micro-motion.
IDLE_MOTION_KEEP = 0.20
# Soft ceiling if we still normalize a near-static Kimodo residual.
IDLE_SOURCE_REF_STD = 0.010
# Live2D-like vertical bob ~0.9% of body height (0.55% read too static on SCAIL).
IDLE_BREATH_PERIOD_S = 1.05
IDLE_BOB_HEIGHT_FRAC = 0.009
# Action motion amount default: full Kimodo deltas on extract pose.
ACTION_MOTION_KEEP = 1.0
# Default Kimodo clip length for action (seconds @ model fps, usually 30).
ACTION_DURATION_SEC = 2.0
# Idle: short seamless loop (Live2D lobby/idle ~2s).
IDLE_DURATION_SEC = 2.0
# After retarget, if upper-body motion is weaker than this RMS, amplify it
# so keep=100% is clearly different from idle micro-motion.
ACTION_UPPER_REF_STD = 0.035

# SCAIL-2 official (zai-org): prompt describes the *finished video*, not instructions;
# long detailed prompts usually beat short/empty ones. Pose still comes from the
# guide; text reinforces identity, framing, motion, and mouth state (distill CFG~1
# weakens negatives, so mouth calm belongs in the positive too).
SCAIL_MOUTH_LOCK = ""  # legacy name; unused

# ~90–120 word English paragraphs (official enhancer targets ~90–140 for replace).
SCAIL_IDLE_POSITIVE = (
    "A full-body character matching the reference image stands facing a fixed "
    "frontal camera under soft even lighting, with clothing, hairstyle, colors, "
    "and proportions held consistent across every frame. The character keeps the "
    "same overall posture in a calm seamless idle loop: soft continuous chest and "
    "torso breathing with a gentle spine sway, a slight head nod and sway, and "
    "shoulders easing with the breath. Arms, "
    "hands, hips, and feet stay still and planted with no waving, gesturing, "
    "walking, dancing, or large joint rotations. The face remains calm and still, "
    "mouth closed, lips sealed, silent, with no talking, lip motion, or expression "
    "change. Identity and wardrobe stay locked to the reference throughout."
)

SCAIL_ACTION_POSITIVE = (
    "A full-body character matching the reference image stands facing a fixed "
    "frontal camera under soft even lighting, with clothing, hairstyle, colors, "
    "and proportions held consistent across every frame. The character performs one "
    "clear upper-body action while the hips and feet stay planted; no walking, "
    "stepping away, or body turn that breaks the frontal framing. Motion is smooth "
    "and readable, limited to the limbs described by the action, without extra "
    "gestures. The face stays calm and still, mouth closed, lips sealed, silent, "
    "with no talking or lip motion. Identity and wardrobe remain locked to the "
    "reference for the entire clip."
)

# SCAIL negative: quality + anti-speech / anti-face-motion blockers.
SCAIL_NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, static, "
    "talking, speaking, speech, dialogue, conversation, monologue, "
    "open mouth, mouth open, mouth opening, mouth closing, mouth moving, "
    "lips moving, lip movement, lip sync, lipsync, lip flap, mouth flap, "
    "jaw opening, jaw moving, chewing, eating, singing, shouting, yelling, "
    "whispering, smiling while talking, teeth showing, tongue out, "
    "facial animation, face morphing, emotive mouth, viseme, phoneme"
)


def _strip_scail_action_for_embed(action_prompt: str) -> str:
    """Motion-only snippet for embedding into a finished-video SCAIL prompt."""
    p = (action_prompt or "").strip()
    if not p:
        return ""
    # Drop mouth-still boilerplate; the SCAIL template already states it.
    p = re.sub(
        r"(?i)\.?\s*Mouth closed and still[^.]*(?:\.|$)",
        ". ",
        p,
    )
    p = re.sub(
        r"(?i)\b(?:lips sealed|silent|no talking|no speaking|no lip movement|"
        r"no chewing)\b[,.]?",
        "",
        p,
    )
    p = re.sub(r"\s{2,}", " ", p).strip(" ,.")
    # Avoid sounding like an edit instruction.
    p = re.sub(
        r"(?i)^(please\s+)?(make|have|let)\s+(the\s+)?(character|person|her|him|them)\s+",
        "",
        p,
    )
    return p.strip(" ,.")


def build_scail_positive(which: str, action_prompt: str | None = None) -> str:
    """Build SCAIL-2 positive text: finished-video description (official style).

    which: \"idle\" | \"action\"
    For action, folds a cleaned user/Kimodo action line into the video description
    so motion wording is concrete without becoming an edit instruction.
    """
    kind = (which or "idle").strip().lower()
    if kind == "idle":
        return SCAIL_IDLE_POSITIVE

    motion = _strip_scail_action_for_embed(action_prompt or "")
    if not motion:
        return SCAIL_ACTION_POSITIVE

    # One paragraph: scene + concrete motion + constraints (official rules 1–7).
    return (
        "A full-body character matching the reference image stands facing a fixed "
        "frontal camera under soft even lighting, with clothing, hairstyle, colors, "
        "and proportions held consistent across every frame. In this clip the "
        f"character moves as follows: {motion.rstrip('.')}."
        " Hips and feet remain planted; no walking, stepping away, or turning that "
        "breaks the frontal framing. Motion stays limited to the described limbs "
        "without extra gestures. The face remains calm and still, mouth closed, "
        "lips sealed, silent, with no talking or lip motion. Identity and wardrobe "
        "stay locked to the reference throughout."
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


# SMPLX22 free DOF for Live2D-style idle (head + chest + light shoulders).
# Locked: pelvis, legs, elbows, wrists.
_IDLE_FREE_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17)


def shape_live2d_idle(
    posed_joints,
    base_pose=None,
    keep: float = IDLE_MOTION_KEEP,
    *,
    fps: float = FPS,
    period_s: float = IDLE_BREATH_PERIOD_S,
    bob_frac: float = IDLE_BOB_HEIGHT_FRAC,
):
    """Live2D-product idle: short seamless loop, head/chest micro, arms/legs fixed.

    1. Anchor to extract ``base_pose`` (frame 0 exact).
    2. Apply Kimodo residual only on free joints, scaled by ``keep``.
    3. Add low-frequency sine breath/bob (~0.95s period, ~0.2% height).
    4. Lock arms/hands/legs/pelvis to base every frame.
    5. Blend last ~12% of frames to frame 0 for permanent loop.
    """
    import numpy as np

    P = np.asarray(posed_joints, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] < 1:
        return P.astype(np.float64, copy=False)
    T, J, _ = P.shape
    try:
        k = float(keep)
    except (TypeError, ValueError):
        k = IDLE_MOTION_KEEP
    k = max(0.0, min(1.0, k))

    if base_pose is not None:
        base = np.asarray(base_pose, dtype=np.float64).reshape(J, 3).copy()
    else:
        base = P[0].copy()

    free = [j for j in _IDLE_FREE_JOINTS if j < J]
    lock = [j for j in range(J) if j not in free]

    deltas = P - P[0:1]
    out = np.repeat(base[None, ...], T, axis=0)
    if k > 0.0 and T >= 2:
        for j in free:
            out[:, j, :] = base[j] + k * deltas[:, j, :]

    try:
        period = float(period_s) if period_s and period_s > 0 else IDLE_BREATH_PERIOD_S
    except (TypeError, ValueError):
        period = IDLE_BREATH_PERIOD_S
    try:
        f = float(fps) if fps and fps > 0 else float(FPS)
    except (TypeError, ValueError):
        f = float(FPS)
    dur = max(T / f, 1e-6)
    n_cycles = max(1, int(round(dur / period)))
    omega = 2.0 * np.pi * n_cycles / dur
    t = np.arange(T, dtype=np.float64) / f
    phase = np.sin(omega * t)
    phase2 = np.sin(omega * t + 0.45)

    if J > 15:
        height = float(np.linalg.norm(base[15] - base[0]))
    else:
        height = float(np.linalg.norm(base[min(9, J - 1)] - base[0]))
    height = max(height, 0.5)
    try:
        bf = float(bob_frac) if bob_frac and bob_frac > 0 else IDLE_BOB_HEIGHT_FRAC
    except (TypeError, ValueError):
        bf = IDLE_BOB_HEIGHT_FRAC
    amp = height * bf

    # Stronger chest/spine rise-fall so breath reads after SCAIL (still arms locked).
    for j, a in ((9, 1.15), (6, 0.95), (3, 0.7), (12, 0.75), (15, 0.55)):
        if j < J:
            out[:, j, 1] = out[:, j, 1] + amp * a * phase
            out[:, j, 0] = out[:, j, 0] + amp * 0.28 * a * phase2
            out[:, j, 2] = out[:, j, 2] + amp * 0.12 * a * phase
    for j, s in ((16, 0.45), (17, 0.45), (13, 0.35), (14, 0.35)):
        if j < J:
            out[:, j, 1] = out[:, j, 1] + amp * 0.55 * phase
            out[:, j, 0] = out[:, j, 0] + amp * 0.18 * s * phase2

    if lock:
        out[:, lock, :] = base[lock, :]

    if T >= 8:
        blend_n = max(2, int(round(0.12 * T)))
        for i in range(blend_n):
            a = (i + 1) / float(blend_n)
            idx = T - blend_n + i
            out[idx] = (1.0 - a) * out[idx] + a * out[0]
        if lock:
            out[:, lock, :] = base[lock, :]

    out[0] = base
    return out


# SMPLX22 lower body (pelvis/legs/feet) — used for the upper/lower split (boost).
# Names: pelvis, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle, L_foot, R_foot
SMPLX22_LOWER_BODY = (0, 1, 2, 4, 5, 7, 8, 10, 11)
# Spine + arms + head (everything not lower body)
SMPLX22_UPPER_BODY = tuple(
    i for i in range(22) if i not in SMPLX22_LOWER_BODY
)
# Seated/lying action lock set: pin only the pelvis root so it does not drift,
# leaving legs and feet free to follow Kimodo. Use SMPLX22_LOWER_BODY here to
# also stick knees/ankles/feet if legs flail out of the seated/lying pose.
SMPLX22_HIPS_ONLY = (0,)

LOWER_BODY_ACTION_RE = re.compile(
    r"\b("
    r"leg|legs|foot|feet|ankle|ankles|knee|knees|"
    r"step|steps|stepping|walk|walking|kick|kicking|"
    r"jump|jumping|crouch|crouching|squat|squatting"
    r")\b",
    re.I,
)

CLOSE_LOWER_BODY_ACTION_RE = re.compile(
    r"\b("
    r"clos(?:e|es|ed|ing)\s+(?:his\s+|her\s+|their\s+|the\s+)?legs|"
    r"bring(?:s|ing)?\s+(?:both\s+)?feet\s+(?:in|inward|inwards|together|close)|"
    r"(?:feet|legs)\s+(?:are\s+)?(?:close\s+together|together)|"
    r"narrow\s+stance"
    r")\b",
    re.I,
)


def lower_body_action_requested(prompt: str | None) -> bool:
    return bool(LOWER_BODY_ACTION_RE.search(prompt or ""))


def close_lower_body_action_requested(prompt: str | None) -> bool:
    return bool(CLOSE_LOWER_BODY_ACTION_RE.search(prompt or ""))


def _close_standing_feet(posed_joints, *, max_foot_gap_ratio: float = 0.12):
    """Ease knees/ankles/feet toward a narrow stance while preserving frame 0."""
    import numpy as np

    out = np.asarray(posed_joints, dtype=np.float64).copy()
    if out.ndim != 3 or out.shape[0] < 2 or out.shape[1] <= 11:
        return out

    start_gap = float(np.linalg.norm(out[0, 10, [0, 2]] - out[0, 11, [0, 2]]))
    if start_gap < 1e-6:
        return out

    target_foot_gap = max(start_gap * float(max_foot_gap_ratio), 1e-4)
    pair_targets = (
        (4, 5, start_gap * 0.45),        # knees should narrow, not collapse
        (7, 8, target_foot_gap * 1.15),  # ankles almost together
        (10, 11, target_foot_gap),       # feet together read
    )
    alpha = np.linspace(0.0, 1.0, out.shape[0], dtype=np.float64)
    alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

    for left, right, target_gap in pair_targets:
        if right >= out.shape[1]:
            continue
        mid = 0.5 * (out[:, left, [0, 2]] + out[:, right, [0, 2]])
        sep = out[:, right, [0, 2]] - out[:, left, [0, 2]]
        gap = np.linalg.norm(sep, axis=1)
        scale = np.ones_like(gap)
        mask = gap > target_gap
        scale[mask] = target_gap / np.maximum(gap[mask], 1e-8)
        narrowed = sep * scale[:, None]
        left_xz = mid - 0.5 * narrowed
        right_xz = mid + 0.5 * narrowed
        out[:, left, [0, 2]] = (
            (1.0 - alpha[:, None]) * out[:, left, [0, 2]]
            + alpha[:, None] * left_xz
        )
        out[:, right, [0, 2]] = (
            (1.0 - alpha[:, None]) * out[:, right, [0, 2]]
            + alpha[:, None] * right_xz
        )

    return out


def align_motion_to_base_pose(
    posed_joints,
    base_pose,
    keep: float = 1.0,
    *,
    lock_lower_body: bool = False,
    preserve_lower_pose: bool = False,
    close_lower_body: bool = False,
    boost_upper: bool = False,
    upper_ref_std: float = ACTION_UPPER_REF_STD,
):
    """Stick frame 0 to ``base_pose`` (extract), scale Kimodo deltas by ``keep``.

        P'[t] = base + keep * (P[t] - P[0])
        P'[0] = base   (hard)

    keep=1: full action deltas. keep=0: frozen base.
    lock_lower_body: pin only the pelvis root to base every frame (sitting/lying),
    leaving legs and feet free.
    preserve_lower_pose: for standing lower-body actions, preserve Kimodo's
    lower-body pose relative to its pelvis instead of only applying frame deltas.
    close_lower_body: for explicit "feet together / close legs" actions, ease
    the generated standing lower body into a narrow stance.
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

    if preserve_lower_pose and not lock_lower_body and out.shape[0] >= 2:
        lower = [j for j in SMPLX22_LOWER_BODY if j != 0 and j < out.shape[1]]
        if lower and P.shape[1] > 0:
            target = out[:, 0:1, :] + (P[:, lower, :] - P[:, 0:1, :])
            alpha = np.linspace(0.0, 1.0, out.shape[0], dtype=np.float64)
            # Ease in so frame 0 remains exactly the approved extract pose.
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            out[:, lower, :] = (
                (1.0 - alpha[:, None, None]) * out[:, lower, :]
                + alpha[:, None, None] * target
            )
            out[0] = base

    if close_lower_body and preserve_lower_pose and not lock_lower_body:
        out = _close_standing_feet(out)
        out[0] = base

    # Apply amount slider on residual from base (frame 0 stays base).
    if k < 1.0:
        out = base[None, ...] + k * (out - base[None, ...])
        out[0] = base
    if lock_lower_body:
        lock = [j for j in SMPLX22_HIPS_ONLY if j < out.shape[1]]
        out[:, lock, :] = base[lock, :]
        out[0] = base
    return out


def sanitize_action(prompt):
    """Drop turning/turn words from an action prompt (fixed-frame rendering
    means a body-turn description would fight the camera-locked guide).
    Also soft-strips speech verbs. Kimodo drives a jawless skeleton, so mouth
    constraints belong to the later SCAIL rendering prompt, not this motion text."""
    p = re.sub(r"\b(turning|turns|turn)\b", "", prompt or "", flags=re.I)
    p = re.sub(
        r"\b(talking|talks|talk|speaking|speaks|speak|saying|says|said|"
        r"whispering|whispers|shouting|shouts|singing|sings|chatting)\b",
        "",
        p,
        flags=re.I,
    )
    p = re.sub(r"\s{2,}", " ", p).strip(" ,.")
    return p


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
             comfy_input: Path | None = None,
             comfy_output: Path | None = None) -> dict:
    from pipeline.paths import comfy_input_dir, comfy_output_dir

    if comfy_input is None:
        comfy_input = comfy_input_dir()
    if comfy_output is None:
        comfy_output = comfy_output_dir()
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
            client, (idle_prompt or DEFAULT_IDLE_PROMPT),
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
            idle_path = drive_character(
                client, idle_guide, image, run_dir / "idle.mp4",
                length=idle_len, width=out_w, height=out_h, prefix="mp_idle", seed=seed,
                positive=build_scail_positive("idle"),
                negative=SCAIL_NEGATIVE,
                comfy_input=comfy_input)
            try:
                from pipeline.face_lock import lock_mouth_in_video
                lock_mouth_in_video(idle_path, in_place=True, strength=1.0)
            except Exception as lock_exc:
                result.setdefault("warnings", {})["idle_mouth_lock"] = str(lock_exc)
            result["idle"] = idle_path
        except Exception as exc:
            result["errors"]["idle_scail"] = str(exc)

    if action_guide is not None:
        try:
            action_path = drive_character(
                client, action_guide, image, run_dir / "action.mp4",
                length=action_len, width=out_w, height=out_h, prefix="mp_action", seed=seed,
                positive=build_scail_positive("action", action_prompt),
                negative=SCAIL_NEGATIVE,
                comfy_input=comfy_input)
            try:
                from pipeline.face_lock import lock_mouth_in_video
                lock_mouth_in_video(action_path, in_place=True, strength=1.0)
            except Exception as lock_exc:
                result.setdefault("warnings", {})["action_mouth_lock"] = str(lock_exc)
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
