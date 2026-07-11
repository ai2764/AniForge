"""Re-run idle + action (+ joint) + SCAIL + bg + time on existing extract session."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.comfy import ComfyClient
from pipeline.generate import DEFAULT_IDLE_PROMPT
from pipeline.stages import (
    stage_action,
    stage_bgremove,
    stage_idle,
    stage_joint_overshoot,
    stage_scail,
    stage_time_overshoot,
)


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--idle-prompt", default="")
    ap.add_argument("--slug", default="rerun")
    ap.add_argument("--no-joint", action="store_true")
    ap.add_argument("--bg-model", default="RMBG-2.0 HQ")
    args = ap.parse_args()

    run_id = args.run_id.strip()
    run_dir = ROOT / "runs" / run_id
    if not run_dir.is_dir():
        print(f"missing {run_dir}", file=sys.stderr)
        return 2

    logf = ROOT / "runs" / f"_{args.slug}_idle_action_log.txt"
    logf.write_text("", encoding="utf-8")
    idle_prompt = (args.idle_prompt or "").strip() or DEFAULT_IDLE_PROMPT

    log(logf, f"=== idle+action re-run run_id={run_id} ===")
    log(logf, f"idle_prompt={idle_prompt[:160]}")
    log(logf, f"action={args.action}")

    client = ComfyClient()
    ok = True
    try:
        fre = client.free_vram(interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0)
        log(logf, str(fre))

        log(logf, "--- idle ---")
        r = stage_idle(run_id, idle_prompt=idle_prompt, client=client)
        if r.get("errors"):
            log(logf, f"FAIL idle: {r['errors']}")
            return 1
        log(logf, f"OK idle motion_std={r.get('motion_std')} keep={r.get('idle_motion_keep')}")

        fre = client.free_vram(interrupt=False, clear_queue=True, wait_s=20, min_free_gb=4.0)
        log(logf, str(fre))
        log(logf, "--- action ---")
        r = stage_action(run_id, action_prompt=args.action, client=client)
        if r.get("errors"):
            log(logf, f"FAIL action: {r['errors']}")
            return 1
        log(logf, f"OK action motion_std={r.get('motion_std')}")

        if not args.no_joint:
            log(logf, "--- joint ---")
            r = stage_joint_overshoot(run_id)
            if r.get("errors"):
                log(logf, f"FAIL joint: {r['errors']}")
                return 1
            log(logf, "OK joint")

        fre = client.free_vram(interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0)
        log(logf, str(fre))
        log(logf, "--- scail both ---")
        r = stage_scail(run_id, which="both", client=client)
        if r.get("errors"):
            log(logf, f"FAIL scail: {r['errors']}")
            ok = False
        else:
            log(logf, "OK scail")

        if ok:
            log(logf, "--- bgremove ---")
            fre = client.free_vram(interrupt=False, clear_queue=True, wait_s=20, min_free_gb=4.0)
            log(logf, str(fre))
            r = stage_bgremove(run_id, which="both", model=args.bg_model)
            if r.get("errors"):
                log(logf, f"WARN bgremove: {r['errors']}")
            else:
                log(logf, "OK bgremove")

            log(logf, "--- time ---")
            r = stage_time_overshoot(run_id)
            if r.get("errors") and any(k != "time_alpha" for k in (r.get("errors") or {})):
                log(logf, f"WARN time: {r['errors']}")
            else:
                log(logf, "OK time")
    except Exception:
        log(logf, traceback.format_exc())
        ok = False
    finally:
        try:
            log(logf, str(client.free_vram(interrupt=False, clear_queue=True, wait_s=15, min_free_gb=4.0)))
        except Exception as e:
            log(logf, f"final free skip: {e}")

    summary = {
        "run_id": run_id,
        "slug": args.slug,
        "ok": ok,
        "action": args.action,
        "outputs": {
            "idle": str(run_dir / "idle.mp4"),
            "action": str(run_dir / "action.mp4"),
            "action_timed": str(run_dir / "action_timed.mp4"),
            "idle_nobg_alpha": str(run_dir / "idle_nobg_alpha.mov"),
            "action_nobg_alpha": str(run_dir / "action_nobg_alpha.mov"),
        },
    }
    sp = ROOT / "runs" / f"_{args.slug}_idle_action_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(logf, f"=== DONE ok={ok} ===")
    log(logf, json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
