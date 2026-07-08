"""CLI entry for the seated/lying pose-anchored pipeline.

    python pipeline/seated/run_seated_pipeline.py [ref_image] [--pose sitting|lying]

Uses generate_anchored (HMR subprocess + Kimodo subprocess + SCAIL via ComfyUI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.comfy import ComfyClient
from pipeline.seated.generate_anchored import generate_anchored


def main(argv=None):
    p = argparse.ArgumentParser(description="Pose-anchored motion portrait (sitting/lying)")
    p.add_argument(
        "image",
        nargs="?",
        default=r"C:/Users/AIBOX/dev/ComfyUI-scail/input/mp_sitting.png",
        help="Reference character image",
    )
    p.add_argument("--pose", choices=("sitting", "lying"), default="sitting")
    p.add_argument("--action", default="A person gestures energetically with arms and upper body.")
    p.add_argument("--idle", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args(argv)

    image = Path(args.image)
    run_dir = Path(args.run_dir) if args.run_dir else Path(__file__).resolve().parent / "runs" / args.pose
    run_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient()
    result = generate_anchored(
        image,
        args.action,
        args.idle,
        overshoot=set(),
        run_dir=run_dir,
        client=client,
        pose_mode=args.pose,
        seed=args.seed,
    )
    print(result)
    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
