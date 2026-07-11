"""Mage standee full pipeline: joint + time overshoot, RMBG-2.0, cool cast action."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.comfy import ComfyClient
from pipeline.stages import (
    create_session,
    stage_action,
    stage_bgremove,
    stage_extract,
    stage_idle,
    stage_joint_overshoot,
    stage_scail,
    stage_time_overshoot,
)

IMAGE = Path(
    r"C:\Users\AIBOX\dev\youtube-video-lab\tasks\live2d\opening-images\立绘"
    r"\ChatGPT Image 2026年7月9日 21_37_02.png"
)
ACTION = (
    "Dramatic mage cast: sweep the right arm up in a wide high arc above the "
    "shoulder, palm open gathering energy, chest opens slightly; then thrust "
    "the open palm forward at chest height in a sharp cast and snap the fingers "
    "closed. High energy, flashy upper body only. Hips and feet fixed, no "
    "stepping, no waist bend, no lunging toward camera. Mouth closed and still "
    "the entire time, lips sealed, jaw locked, silent, no talking, no lip movement."
)
LOG = ROOT / "runs" / "_mage_full_log.txt"
SUMMARY = ROOT / "runs" / "_mage_full_summary.json"
SCALE = 1.0
SEED = 42
BG_MODEL = "RMBG-2.0 HQ"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check(name: str, result: dict) -> bool:
    errs = {k: v for k, v in (result.get("errors") or {}).items() if k not in ("time_alpha",)}
    if errs:
        log(f"FAIL {name}: {json.dumps(errs, ensure_ascii=False)}")
        return False
    if result.get("errors"):
        log(f"WARN {name}: {result['errors']}")
    log(f"OK {name}")
    return True


def free(client: ComfyClient, min_gb: float = 6.0) -> None:
    try:
        log(str(client.free_vram(interrupt=False, clear_queue=True, wait_s=25, min_free_gb=min_gb)))
    except Exception as e:
        log(f"free skip: {e}")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    if not IMAGE.is_file():
        log(f"ERROR missing: {IMAGE}")
        return 2
    log("=== mage full pipeline ===")
    log(f"image={IMAGE}")
    log(f"action={ACTION}")

    client = ComfyClient()
    try:
        client.object_info()
    except Exception as e:
        log(f"ERROR Comfy: {e}")
        return 2
    free(client)

    sess = create_session(
        IMAGE.read_bytes(),
        IMAGE.name,
        pose_mode="standing",
        seed=SEED,
        scale=SCALE,
    )
    run_id = sess["run_id"]
    run_dir = ROOT / "runs" / run_id
    log(f"run_id={run_id} size={sess.get('size')}")

    if not check("extract", stage_extract(run_id)):
        return 3
    if not check("idle", stage_idle(run_id, idle_motion_keep=0.08)):
        return 4
    if not check(
        "action",
        stage_action(
            run_id,
            action_prompt=ACTION,
            action_motion_keep=1.0,
            action_duration=2.0,
        ),
    ):
        return 5
    if not check("joint", stage_joint_overshoot(run_id)):
        return 6

    free(client, 8.0)
    if not check("scail_idle", stage_scail(run_id, which="idle", client=client)):
        return 7
    free(client, 8.0)
    if not check("scail_action", stage_scail(run_id, which="action", client=client)):
        return 8

    free(client, 6.0)
    r = stage_bgremove(run_id, which="both", model=BG_MODEL)
    has = any(r.get(k) for k in ("idle_nobg", "action_nobg", "idle_nobg_alpha", "action_nobg_alpha"))
    if r.get("errors"):
        log(f"bg notes: {r['errors']}")
    if not has:
        log("FAIL bgremove")
        return 9
    log(f"OK bgremove alpha idle={r.get('idle_nobg_alpha')} action={r.get('action_nobg_alpha')}")

    free(client, 4.0)
    if not check("time", stage_time_overshoot(run_id)):
        return 10

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "slug": "mage",
        "image": str(IMAGE),
        "scale": SCALE,
        "seed": SEED,
        "bg_model": BG_MODEL,
        "joint_overshoot": True,
        "time_overshoot": True,
        "action_prompt": ACTION,
        "outputs": {
            "idle": str(run_dir / "idle.mp4"),
            "action": str(run_dir / "action.mp4"),
            "action_timed": str(run_dir / "action_timed.mp4"),
            "idle_nobg_alpha": str(run_dir / "idle_nobg_alpha.mov"),
            "action_nobg_alpha": str(run_dir / "action_nobg_alpha.mov"),
            "action_timed_webm": str(run_dir / "action_timed.webm"),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "mage_full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    free(client)
    log("=== DONE ===")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log(traceback.format_exc())
        raise SystemExit(1)
