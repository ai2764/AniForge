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

DEFAULT_IDLE_PROMPT = (
    "A person holds their current pose in a relaxed idle, breathing calmly, "
    "with only tiny subtle micro-movements of the head and torso. "
    "Keep the same overall posture; no large joint rotations, no big arm or leg swings, "
    "no twisting or turning."
)
FPS = 30                      # Kimodo output is 30 fps
JOINT_SPRING = dict(omega=20.0, zeta=0.35, soft=1.0)   # action joint-overshoot defaults
TIME_SPRING = dict(b=0.42, d=4.2, f=2.4, t=1.15)       # action time-overshoot params


def plan_steps(overshoot):
    """overshoot: set[str] of selected post-steps ("joint", "time")."""
    return {"joint": "joint" in overshoot, "time": "time" in overshoot}


def sanitize_action(prompt):
    """Drop turning/turn words from an action prompt (fixed-frame rendering
    means a body-turn description would fight the camera-locked guide)."""
    return re.sub(r"\b(turning|turns|turn)\b", "", prompt, flags=re.I).strip()


def align_4k1(n):
    """SCAIL length must be 4k+1."""
    return ((int(n) - 1) // 4) * 4 + 1 if n > 1 else 1


def _output_size(image_path, long_cap=1280, short_cap=720, mult=16):
    """Output width/height from the input image's aspect ratio, scaled to fit a
    720p frame (longer side <= long_cap, shorter side <= short_cap), rounded to a
    multiple of `mult` (SCAIL/VAE stride)."""
    from PIL import Image
    with Image.open(image_path) as im:
        w, h = im.size
    scale = min(long_cap / max(w, h), short_cap / min(w, h))
    r = lambda v: max(mult, int(round(v * scale / mult)) * mult)
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
                 model, seed, comfy_output, joint, out_w, out_h):
    """Kimodo motion -> optional joint-space spring -> fixed-frame render ->
    aspect-matched pad. Returns (guide_path, scail_length)."""
    npz = generate_motion(client, prompt, npz_path, model=model, seed=seed,
                          comfy_output=comfy_output)
    kpts = load_posed_joints(npz)
    if joint:
        kpts = spring_follow(kpts, FPS, omega=JOINT_SPRING["omega"],
                             zeta=JOINT_SPRING["zeta"], soft_scale=JOINT_SPRING["soft"])
    render(kpts, skel_path, FPS, fixed=True)
    _pad_to_aspect(skel_path, guide_path, out_w, out_h)
    return guide_path, align_4k1(len(kpts))


def generate(image: Path, action_prompt: str, idle_prompt, overshoot: set,
             run_dir: Path, client, *, motion_model="Kimodo-SOMA-RP-v1", seed=None,
             comfy_input=Path("C:/Users/AIBOX/dev/ComfyUI-scail/input"),
             comfy_output=Path("C:/Users/AIBOX/dev/ComfyUI-scail/output")) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image = Path(image)
    plan = plan_steps(overshoot)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    out_w, out_h = _output_size(image)

    result = {"idle": None, "action": None, "errors": {}, "seed": seed,
              "size": [out_w, out_h]}

    # Phase 1: generate BOTH motions -> padded skeleton guides.
    # Doing both Kimodo passes back-to-back keeps its model hot in VRAM instead
    # of swapping Kimodo<->SCAIL between clips.
    idle_guide = action_guide = None
    idle_len = action_len = None
    try:
        idle_guide, idle_len = _build_guide(
            client, idle_prompt or DEFAULT_IDLE_PROMPT,
            run_dir / "idle.npz", run_dir / "idle_skel.mp4", run_dir / "idle_guide.mp4",
            model=motion_model, seed=seed, comfy_output=comfy_output, joint=False,
            out_w=out_w, out_h=out_h)
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
                positive="a character in a calm idle pose, full body, consistent identity",
                comfy_input=comfy_input)
        except Exception as exc:
            result["errors"]["idle_scail"] = str(exc)

    if action_guide is not None:
        try:
            action_path = drive_character(
                client, action_guide, image, run_dir / "action.mp4",
                length=action_len, width=out_w, height=out_h, prefix="mp_action", seed=seed,
                positive="a character performing an action, full body, consistent identity",
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
