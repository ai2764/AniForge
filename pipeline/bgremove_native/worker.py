from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.bgremove_native.runner import run_bgremove_native


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("model")
    parser.add_argument("formats")
    parser.add_argument("--bg")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--infer-size", type=int, default=0)
    parser.add_argument("--alpha-shrink", type=int, default=0)
    parser.add_argument("--alpha-feather", type=int, default=0)
    args = parser.parse_args(argv)

    t0 = time.time()
    result = run_bgremove_native(
        Path(args.input),
        Path(args.output_dir),
        model=args.model,
        formats=args.formats,
        bg_image=Path(args.bg) if args.bg else None,
        fp16=args.fp16,
        infer_size=args.infer_size,
        alpha_shrink=args.alpha_shrink,
        alpha_feather=args.alpha_feather,
    )
    if result.get("preview"):
        print(f"RESULT:preview:{result['preview']}", flush=True)
    for output in result.get("outputs") or []:
        print(f"RESULT:output:{output}", flush=True)
    if result.get("log"):
        for log_line in str(result["log"]).splitlines():
            if log_line.strip():
                print(f"WARN: {log_line}", flush=True)
    print(f"DONE:{time.time() - t0:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
