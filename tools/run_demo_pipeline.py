"""End-to-end demo: image → extract → idle → action → SCAIL → bgremove."""
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
    stage_scail,
)

IMAGE = Path(r"C:\Users\AIBOX\Downloads\ChatGPT Image 2026年7月8日 13_00_08.png")
ACTION = (
    "Raise the right hand to shoulder height with a clear small wave, "
    "then lower it slightly. Mouth closed and still."
)
OUT_LOG = ROOT / "runs" / "_demo_pipeline_log.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    if not IMAGE.is_file():
        log(f"ERROR: image not found: {IMAGE}")
        return 1

    log("=== free Comfy VRAM (best effort) ===")
    try:
        fre = ComfyClient().free_vram(
            interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0
        )
        log(f"free_vram: {fre}")
    except Exception as e:
        log(f"free_vram skip: {e}")

    log("=== 1 create_session (standing) ===")
    sess = create_session(
        IMAGE.read_bytes(),
        IMAGE.name,
        pose_mode="standing",
        seed=42,
        scale=0.5,  # faster SCAIL; still decent for demo
    )
    run_id = sess["run_id"]
    log(f"run_id={run_id} size={sess.get('size')} seed={sess.get('seed')}")

    def check(name, result):
        errs = result.get("errors") or {}
        if errs:
            log(f"FAIL {name}: {json.dumps(errs, ensure_ascii=False)}")
            return False
        log(f"OK {name}: keys={[k for k in result if k not in ('errors',)][:12]}")
        return True

    log("=== 2 extract ===")
    if not check("extract", stage_extract(run_id)):
        return 2

    log("=== 3 idle skeleton (Live2D-style 2s) ===")
    if not check("idle", stage_idle(run_id, idle_motion_keep=0.08)):
        return 3

    log("=== 4 action skeleton (2s, raise hand) ===")
    if not check(
        "action",
        stage_action(
            run_id,
            action_prompt=ACTION,
            action_motion_keep=1.0,
            action_duration=2.0,
        ),
    ):
        return 4

    client = ComfyClient()
    try:
        client.object_info()
    except Exception as e:
        log(f"ERROR: ComfyUI unreachable: {e}")
        return 5

    log("=== 5 SCAIL idle ===")
    r = stage_scail(run_id, which="idle", client=client)
    if not check("scail_idle", r):
        return 6

    log("=== 6 SCAIL action ===")
    r = stage_scail(run_id, which="action", client=client)
    if not check("scail_action", r):
        return 7

    log("=== 7 free Comfy before bgremove ===")
    try:
        log(str(client.free_vram(interrupt=False, clear_queue=True, wait_s=30, min_free_gb=6.0)))
    except Exception as e:
        log(f"free before bg: {e}")

    log("=== 8 bgremove both ===")
    r = stage_bgremove(run_id, which="both", model="RMBG-2.0 HQ")
    # partial ok
    has = any(r.get(k) for k in ("idle_nobg", "action_nobg", "idle_nobg_webm", "action_nobg_webm"))
    if r.get("errors"):
        log(f"bgremove errors: {r['errors']}")
    if not has:
        log("FAIL bgremove: no outputs")
        return 8
    log(f"OK bgremove: idle_nobg={r.get('idle_nobg')} action_nobg={r.get('action_nobg')}")
    log(f"webm: idle={r.get('idle_nobg_webm')} action={r.get('action_nobg_webm')}")

    run_dir = ROOT / "runs" / run_id
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "outputs": {
            "idle": str(run_dir / "idle.mp4"),
            "action": str(run_dir / "action.mp4"),
            "idle_nobg": str(run_dir / "idle_nobg.mp4"),
            "action_nobg": str(run_dir / "action_nobg.mp4"),
            "idle_nobg_webm": str(run_dir / "idle_nobg.webm"),
            "action_nobg_webm": str(run_dir / "action_nobg.webm"),
        },
    }
    (run_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log("=== DONE ===")
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log(traceback.format_exc())
        raise
