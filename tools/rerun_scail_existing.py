"""Re-run SCAIL (+ optional bgremove/time) using existing skeleton motion.

Usage:
  python tools/rerun_scail_existing.py --run-id 1a93a60250a443539c1381ec5e99b547
  python tools/rerun_scail_existing.py --run-id ... --which both --no-bg --no-time
"""
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
from pipeline.generate import build_scail_positive
from pipeline.stages import stage_bgremove, stage_scail, stage_time_overshoot


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--which", default="both", choices=("idle", "action", "both"))
    ap.add_argument("--bg-model", default="RMBG-2.0 HQ")
    ap.add_argument("--no-bg", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--slug", default="mecha_pilot2")
    args = ap.parse_args()

    run_id = args.run_id.strip()
    run_dir = ROOT / "runs" / run_id
    if not run_dir.is_dir():
        print(f"missing run dir: {run_dir}", file=sys.stderr)
        return 2

    logf = ROOT / "runs" / f"_{args.slug}_scail_reprompt_log.txt"
    summary_path = ROOT / "runs" / f"_{args.slug}_scail_reprompt_summary.json"
    logf.write_text("", encoding="utf-8")

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    log(logf, f"=== SCAIL re-run existing motion run_id={run_id} which={args.which} ===")
    log(logf, f"action_prompt={(meta.get('action_prompt') or '')[:200]}")
    idle_p = build_scail_positive("idle")
    act_p = build_scail_positive("action", meta.get("action_prompt"))
    log(logf, f"idle_scail_positive words={len(idle_p.split())}")
    log(logf, f"action_scail_positive words={len(act_p.split())}")
    log(logf, act_p[:320] + ("..." if len(act_p) > 320 else ""))

    for name in ("idle_guide.mp4", "action_guide.mp4", "idle_skel.mp4", "action_skel.mp4"):
        p = run_dir / name
        log(logf, f"  {name}: {'OK' if p.is_file() else 'MISSING'} {p.stat().st_size if p.is_file() else 0}")

    client = ComfyClient()
    ok = True
    try:
        log(logf, "--- scail ---")
        r = stage_scail(run_id, which=args.which, client=client)
        errs = r.get("errors") or {}
        log(
            logf,
            json.dumps(
                {
                    "idle": r.get("idle"),
                    "action": r.get("action"),
                    "errors": errs,
                },
                ensure_ascii=False,
            ),
        )
        if errs:
            log(logf, f"FAIL scail: {errs}")
            ok = False
        else:
            log(logf, "OK scail")

        if ok and not args.no_bg:
            log(logf, "--- bgremove ---")
            fre = client.free_vram(
                interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0
            )
            log(logf, str(fre))
            r = stage_bgremove(
                run_id, which=args.which, model=args.bg_model
            )
            errs = r.get("errors") or {}
            log(logf, f"bgremove errors={errs} keys={[k for k in r if k not in ('errors',)]}")
            if errs:
                log(logf, f"WARN bgremove: {errs}")
            else:
                log(logf, "OK bgremove")

        if ok and not args.no_time and args.which in ("action", "both"):
            log(logf, "--- time ---")
            fre = client.free_vram(
                interrupt=False, clear_queue=True, wait_s=20, min_free_gb=4.0
            )
            log(logf, str(fre))
            r = stage_time_overshoot(run_id)
            errs = r.get("errors") or {}
            log(logf, f"time errors={errs}")
            if errs and any(k != "time_alpha" for k in errs):
                log(logf, f"WARN time: {errs}")
            else:
                log(logf, "OK time")
    except Exception:
        log(logf, traceback.format_exc())
        ok = False
    finally:
        try:
            log(
                logf,
                str(
                    client.free_vram(
                        interrupt=False, clear_queue=True, wait_s=15, min_free_gb=4.0
                    )
                ),
            )
        except Exception as e:
            log(logf, f"final free skip: {e}")

    meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in (
        "idle_scail_positive",
        "action_scail_positive",
        "idle_scail_done",
        "action_scail_done",
    ):
        v = meta2.get(k)
        if isinstance(v, str) and len(v) > 100:
            log(logf, f"{k}: {v[:100]}... ({len(v.split())} words)")
        else:
            log(logf, f"{k}: {v}")

    summary = {
        "run_id": run_id,
        "slug": args.slug,
        "mode": "scail_reprompt_existing_motion",
        "which": args.which,
        "ok": ok,
        "outputs": {
            "idle": str(run_dir / "idle.mp4"),
            "action": str(run_dir / "action.mp4"),
            "action_timed": str(run_dir / "action_timed.mp4"),
            "idle_nobg_alpha": str(run_dir / "idle_nobg_alpha.mov"),
            "action_nobg_alpha": str(run_dir / "action_nobg_alpha.mov"),
            "action_timed_webm": str(run_dir / "action_timed.webm"),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(logf, f"=== DONE ok={ok} ===")
    log(logf, json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
