"""Standalone CLI for the AniForge pipeline — no web server needed.

Runs the full flow (Kimodo motion -> optional joint spring -> SCAIL2 character
drive -> optional time spring) for one image + prompts, and writes idle.mp4 and
action.mp4. Only depends on a running ComfyUI (Kimodo + SCAIL2 nodes) at 8188
and ffmpeg on PATH — the `server/` app is not involved.

Examples:
  python run.py char.png --action "raises the right arm and waves"
  python run.py char.png --action "waves" --joint --time --seed 42 --out out/
  python run.py char.png --action "bows" --idle "stands calmly, breathing"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.comfy import ComfyClient
from pipeline.generate import generate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate an idle + action live2d-style portrait from one image.")
    p.add_argument("image", type=Path, help="Character image (full-body portrait).")
    p.add_argument("--action", required=True, help="Action prompt (the click-triggered motion).")
    p.add_argument("--idle", default=None, help="Idle prompt (optional; blank uses a default relaxed idle).")
    p.add_argument("--joint", action="store_true", help="Apply joint-space spring overshoot to the action.")
    p.add_argument("--time", action="store_true", help="Apply time-space spring overshoot to the action.")
    p.add_argument("--seed", type=int, default=None, help="Seed (default: random). Same seed + prompt reproduces a result.")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: runs/cli_<timestamp>).")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI base URL.")
    args = p.parse_args(argv)

    if not args.image.exists():
        p.error(f"image not found: {args.image}")

    overshoot = set()
    if args.joint:
        overshoot.add("joint")
    if args.time:
        overshoot.add("time")

    run_dir = args.out or (Path("runs") / f"cli_{int(time.time())}")

    client = ComfyClient(base_url=args.comfy_url)
    print(f"Generating (image={args.image.name}, overshoot={sorted(overshoot) or 'none'}) ...")
    t0 = time.time()
    result = generate(args.image, args.action, args.idle, overshoot,
                      run_dir=run_dir, client=client, seed=args.seed)

    print(f"Done in {time.time() - t0:.0f}s   seed={result.get('seed')}")
    print(f"  idle:   {result.get('idle')}")
    print(f"  action: {result.get('action')}")
    if result.get("errors"):
        print(f"  errors: {result['errors']}", file=sys.stderr)

    # Non-zero exit if neither clip was produced.
    return 0 if (result.get("idle") or result.get("action")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
