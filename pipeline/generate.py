"""Generate orchestration: one reference image + prompts -> idle.mp4 + action.mp4.

Wires together kimodo (text -> motion NPZ), skeleton_spring (spring overshoot +
skeleton render), scail (character drive), and spring_time_remap (playback
overshoot) into two independent clips. Each clip is produced in its own
try/except so a ComfyUI failure on one branch does not prevent the other
branch's clip from being returned.
"""
from __future__ import annotations

import re
from pathlib import Path

from pipeline.kimodo import generate_motion
from pipeline.skeletons import load_posed_joints
from pipeline.skeleton_spring import spring_follow, render
from pipeline.scail import drive_character
from pipeline.spring_time_remap import time_remap_file

DEFAULT_IDLE_PROMPT = ("A person stands in place in a relaxed idle stance, breathing calmly, "
                       "swaying gently from side to side, with small subtle movements of the head and arms.")
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


def _pad_to_9x16(inp, out):
    """512x512 skeleton render -> 512x888, black bars, keep frames/fps."""
    import subprocess
    import os
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    subprocess.run([ffmpeg, "-y", "-i", str(inp),
                    "-vf", "pad=512:888:0:(888-512)/2:color=black",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True, capture_output=True)


def generate(image: Path, action_prompt: str, idle_prompt, overshoot: set,
             run_dir: Path, client, *, motion_model="Kimodo-SOMA-RP-v1",
             comfy_input=Path("C:/Users/AIBOX/dev/ComfyUI-scail/input"),
             comfy_output=Path("C:/Users/AIBOX/dev/ComfyUI-scail/output")) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    image = Path(image)
    plan = plan_steps(overshoot)

    result = {"idle": None, "action": None, "errors": {}}

    try:
        npz = generate_motion(client, idle_prompt or DEFAULT_IDLE_PROMPT, run_dir / "idle.npz",
                               model=motion_model, comfy_output=comfy_output)
        kpts = load_posed_joints(npz)
        render(kpts, run_dir / "idle_skel.mp4", FPS, fixed=True)
        _pad_to_9x16(run_dir / "idle_skel.mp4", run_dir / "idle_guide.mp4")
        result["idle"] = drive_character(
            client, run_dir / "idle_guide.mp4", image, run_dir / "idle.mp4",
            length=align_4k1(len(kpts)), prefix="mp_idle",
            positive="a character in a calm idle pose, full body, consistent identity",
            comfy_input=comfy_input)
    except Exception as exc:
        result["errors"]["idle"] = str(exc)

    try:
        npz = generate_motion(client, sanitize_action(action_prompt), run_dir / "action.npz",
                               model=motion_model, comfy_output=comfy_output)
        kpts = load_posed_joints(npz)
        if plan["joint"]:
            kpts = spring_follow(kpts, FPS, omega=JOINT_SPRING["omega"],
                                  zeta=JOINT_SPRING["zeta"], soft_scale=JOINT_SPRING["soft"])
        render(kpts, run_dir / "action_skel.mp4", FPS, fixed=True)
        _pad_to_9x16(run_dir / "action_skel.mp4", run_dir / "action_guide.mp4")
        action_path = drive_character(
            client, run_dir / "action_guide.mp4", image, run_dir / "action.mp4",
            length=align_4k1(len(kpts)), prefix="mp_action",
            positive="a character performing an action, full body, consistent identity",
            comfy_input=comfy_input)
        if plan["time"]:
            time_remap_file(action_path, run_dir / "action_timed.mp4", **TIME_SPRING)
            action_path = run_dir / "action_timed.mp4"
        result["action"] = action_path
    except Exception as exc:
        result["errors"]["action"] = str(exc)

    return result
