"""Force ComfyUI to interrupt, clear queue, unload models, free VRAM.

Usage (repo root):
  python tools/free_vram.py
  python tools/free_vram.py --interrupt   # cancel running SCAIL too
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.comfy import ComfyClient  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Free ComfyUI VRAM")
    p.add_argument("--interrupt", action="store_true", help="Interrupt running prompt")
    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--wait", type=float, default=60.0)
    p.add_argument("--url", default="http://127.0.0.1:8188")
    args = p.parse_args()

    c = ComfyClient(args.url)
    print("before free_bytes:", c.vram_free_bytes())
    report = c.free_vram(
        interrupt=args.interrupt,
        clear_queue=True,
        wait_s=args.wait,
        min_free_gb=args.min_free_gb,
    )
    print("report:", report)
    print("after free_bytes:", c.vram_free_bytes())
    if not report.get("ok"):
        print(
            "\nVRAM still low. If a SCAIL job is running, re-run with --interrupt,\n"
            "or restart ComfyUI-scail (only reliable full clear on Windows)."
        )
        sys.exit(1)
    print("OK — VRAM freed enough to continue.")


if __name__ == "__main__":
    main()
