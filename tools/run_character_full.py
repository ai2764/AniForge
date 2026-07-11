"""Generic standee full pipeline: extract→idle→action→joint→SCAIL→RMBG→time.

Usage:
  python tools/run_character_full.py --slug nurse --image "path\\to.png" --action "..."
  python tools/run_character_full.py --preset mecha2
  python tools/run_character_full.py --preset nurse
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

STANDING_DIR = Path(
    r"C:\Users\AIBOX\dev\youtube-video-lab\tasks\live2d\opening-images\立绘"
)

PRESETS: dict[str, dict] = {
    "mecha2": {
        "slug": "mecha_pilot2",
        "image": STANDING_DIR / "mecha polit2.png",
        # Single clean beat only — multi-step prompts often look broken in Kimodo.
        "action": (
            "Raise the right hand straight up beside the temple in one clear "
            "military salute, hold still for a moment, then lower the same hand "
            "smoothly back to the side. Only the right arm moves. Left arm stays "
            "still. Torso upright, hips and feet fixed, no stepping, no twist. "
            "Mouth closed and still, lips sealed, silent, no talking."
        ),
    },
    "nurse": {
        "slug": "nurse",
        "image": STANDING_DIR / "nurse.png",
        "action": (
            "Slowly bring both hands to the waist and rest them firmly on the "
            "hips in a relaxed hands-on-hips pose, then hold still. Smooth, "
            "gentle, soft motion, not snappy. Both arms move together; "
            "clipboard can settle at the hip if needed. Torso upright, hips and "
            "feet fixed, no stepping, no twist. Mouth closed and still, lips "
            "sealed, silent, no talking."
        ),
    },
}

SCALE = 1.0
SEED = 42
BG_MODEL = "RMBG-2.0 HQ"


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check(logf: Path, name: str, result: dict) -> bool:
    errs = {
        k: v
        for k, v in (result.get("errors") or {}).items()
        if k not in ("time_alpha",)
    }
    if errs:
        log(logf, f"FAIL {name}: {json.dumps(errs, ensure_ascii=False)}")
        return False
    if result.get("errors"):
        log(logf, f"WARN {name}: {result['errors']}")
    log(logf, f"OK {name}")
    return True


def free(logf: Path, client: ComfyClient, min_gb: float = 6.0) -> None:
    try:
        log(
            logf,
            str(
                client.free_vram(
                    interrupt=False, clear_queue=True, wait_s=25, min_free_gb=min_gb
                )
            ),
        )
    except Exception as e:
        log(logf, f"free skip: {e}")


def run_one(slug: str, image: Path, action: str) -> int:
    logf = ROOT / "runs" / f"_{slug}_full_log.txt"
    summary_path = ROOT / "runs" / f"_{slug}_full_summary.json"
    logf.write_text("", encoding="utf-8")

    if not image.is_file():
        log(logf, f"ERROR missing image: {image}")
        return 2

    log(logf, f"=== {slug} full pipeline ===")
    log(logf, f"image={image}")
    log(logf, f"action={action}")
    log(logf, f"scale={SCALE} seed={SEED} bg={BG_MODEL} joint=on time=on")

    client = ComfyClient()
    try:
        client.object_info()
    except Exception as e:
        log(logf, f"ERROR Comfy: {e}")
        return 2
    free(logf, client)

    sess = create_session(
        image.read_bytes(),
        image.name,
        pose_mode="standing",
        seed=SEED,
        scale=SCALE,
    )
    run_id = sess["run_id"]
    run_dir = ROOT / "runs" / run_id
    log(logf, f"run_id={run_id} size={sess.get('size')}")

    if not check(logf, "extract", stage_extract(run_id)):
        return 3
    if not check(logf, "idle", stage_idle(run_id, idle_motion_keep=0.08)):
        return 4
    if not check(
        logf,
        "action",
        stage_action(
            run_id,
            action_prompt=action,
            action_motion_keep=1.0,
            action_duration=2.0,
        ),
    ):
        return 5
    # Light joint spring only (full spring can look broken on short gestures).
    if not check(logf, "joint", stage_joint_overshoot(run_id)):
        return 6

    free(logf, client, 8.0)
    if not check(logf, "scail_idle", stage_scail(run_id, which="idle", client=client)):
        return 7
    free(logf, client, 8.0)
    if not check(
        logf, "scail_action", stage_scail(run_id, which="action", client=client)
    ):
        return 8

    free(logf, client, 6.0)
    r = stage_bgremove(run_id, which="both", model=BG_MODEL)
    has = any(
        r.get(k)
        for k in (
            "idle_nobg",
            "action_nobg",
            "idle_nobg_alpha",
            "action_nobg_alpha",
        )
    )
    if r.get("errors"):
        log(logf, f"bg notes: {r['errors']}")
    if not has:
        log(logf, "FAIL bgremove")
        return 9
    log(
        logf,
        f"OK bgremove idle_alpha={r.get('idle_nobg_alpha')} "
        f"action_alpha={r.get('action_nobg_alpha')}",
    )

    free(logf, client, 4.0)
    if not check(logf, "time", stage_time_overshoot(run_id)):
        return 10

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "slug": slug,
        "image": str(image),
        "scale": SCALE,
        "seed": SEED,
        "bg_model": BG_MODEL,
        "joint_overshoot": True,
        "time_overshoot": True,
        "action_prompt": action,
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
    (run_dir / f"{slug}_full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    free(logf, client)
    log(logf, "=== DONE ===")
    log(logf, json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS.keys()))
    p.add_argument("--slug")
    p.add_argument("--image", type=Path)
    p.add_argument("--action")
    p.add_argument(
        "--all-remaining",
        action="store_true",
        help="Run mecha2 then nurse sequentially",
    )
    args = p.parse_args(argv)

    jobs: list[tuple[str, Path, str]] = []
    if args.all_remaining:
        for key in ("mecha2", "nurse"):
            pr = PRESETS[key]
            jobs.append((pr["slug"], pr["image"], pr["action"]))
    elif args.preset:
        pr = PRESETS[args.preset]
        jobs.append((pr["slug"], pr["image"], pr["action"]))
    else:
        if not args.slug or not args.image or not args.action:
            p.error("need --preset / --all-remaining / or --slug --image --action")
        jobs.append((args.slug, args.image, args.action))

    rc = 0
    for slug, image, action in jobs:
        try:
            code = run_one(slug, image, action)
        except Exception:
            print(traceback.format_exc(), flush=True)
            code = 1
        if code != 0:
            rc = code
            print(f"[batch] {slug} failed code={code}, continue next", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
