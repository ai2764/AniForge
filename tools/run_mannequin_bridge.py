"""Two-hop SCAIL: skeleton guide (+ joint overshoot) -> mannequin -> character.

Uses existing motion guides from a source run (does not re-run Kimodo).

  stage1: pose_video = idle/action skeleton guide
          reference  = neutral mannequin still
  stage2: pose_video = mannequin SCAIL output
          reference  = character still

Example:
  python tools/run_mannequin_bridge.py \\
    --source-run 1a93a60250a443539c1381ec5e99b547 \\
    --mannequin "C:\\...\\neutral_mannequin_front_9x16.png" \\
    --character "C:\\...\\mecha polit2.png"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.comfy import ComfyClient
from pipeline.generate import (
    SCAIL_NEGATIVE,
    align_4k1,
    build_scail_positive,
    _output_size,
)
from pipeline.scail import drive_character

# Proven mannequin prompt (L2D-mesh / studio bridge style).
MANNEQUIN_POSITIVE = (
    "A plain neutral full-body mannequin follows only the body motion from the "
    "driving video. The mannequin is faceless, hairless, identity-free, and wears "
    "a smooth matte light gray bodysuit. Full body centered, empty relaxed hands, "
    "open relaxed fingers, no weapons, no ribbons, no accessories, no costume "
    "details, no character identity, no hair, no face features, no mouth, no text, "
    "no logo, stable fixed frontal camera. Preserve only the body motion, weight "
    "shift, shoulder movement, waist movement, head movement, and arm movement "
    "from the guide video. Remove all original character appearance and clothing "
    "details. Smooth matte surface throughout, no facial animation."
)

MANNEQUIN_NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, "
    "face details, eyes, mouth, lips, teeth, hair, clothing patterns, "
    "costume, identity, character, talking, speaking, open mouth, lip sync, "
    "jaw moving, facial animation, smile, expression"
)


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_frames_from_guide_or_npz(run_dir: Path, label: str, seed: int) -> int:
    for name in (
        f"{label}_joint_seed{seed}.npz",
        f"{label}_seed{seed}.npz",
    ):
        p = run_dir / name
        if p.is_file():
            return int(np.load(p)["posed_joints"].shape[0])
    return 61


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-run",
        default="1a93a60250a443539c1381ec5e99b547",
        help="Run with existing idle/action guides (+ joint overshoot on action)",
    )
    ap.add_argument(
        "--mannequin",
        default=os.environ.get("MANNEQUIN_IMAGE") or "",
    )
    ap.add_argument(
        "--character",
        default="",
        help="Character ref image; default = source-run input.png",
    )
    ap.add_argument("--which", default="both", choices=("idle", "action", "both"))
    ap.add_argument("--pose-strength-1", type=float, default=0.95)
    ap.add_argument("--pose-strength-2", type=float, default=0.90)
    args = ap.parse_args()

    src_id = args.source_run.strip()
    src = ROOT / "runs" / src_id
    if not src.is_dir():
        print(f"missing source run: {src}", file=sys.stderr)
        return 2

    mannequin = Path(args.mannequin) if args.mannequin else Path("")
    if not mannequin.is_file():
        print(
            "missing mannequin image: pass --mannequin or set MANNEQUIN_IMAGE",
            file=sys.stderr,
        )
        return 2

    meta_src = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    seed = int(meta_src.get("seed") or 42)
    action_prompt = meta_src.get("action_prompt") or ""
    char = Path(args.character) if args.character else (src / "input.png")
    if not char.is_file():
        print(f"missing character: {char}", file=sys.stderr)
        return 2

    # Output size: match source character session (guides already padded to this).
    size = meta_src.get("size")
    if size and len(size) == 2:
        out_w, out_h = int(size[0]), int(size[1])
    else:
        out_w, out_h = _output_size(char, scale=float(meta_src.get("scale") or 1.0))

    run_id = uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = ROOT / "runs" / f"_mannequin_bridge_{run_id[:8]}_log.txt"
    logf.write_text("", encoding="utf-8")

    # Stage assets into new run
    shutil.copy2(mannequin, run_dir / "mannequin.png")
    shutil.copy2(char, run_dir / "character.png")
    shutil.copy2(char, run_dir / "input.png")

    labels: list[str] = []
    if args.which in ("idle", "both"):
        labels.append("idle")
    if args.which in ("action", "both"):
        labels.append("action")

    for lab in labels:
        g = src / f"{lab}_guide.mp4"
        if not g.is_file():
            g = src / f"{lab}_skel.mp4"
        if not g.is_file():
            log(logf, f"FAIL missing guide for {lab}")
            return 2
        shutil.copy2(g, run_dir / f"{lab}_skel_guide.mp4")
        # Also keep joint flag note for action
        if lab == "action":
            jn = src / f"action_joint_seed{seed}.npz"
            if jn.is_file():
                shutil.copy2(jn, run_dir / jn.name)
            an = src / f"action_seed{seed}.npz"
            if an.is_file():
                shutil.copy2(an, run_dir / an.name)

    meta = {
        "mode": "mannequin_bridge",
        "source_run": src_id,
        "seed": seed,
        "size": [out_w, out_h],
        "pose_mode": meta_src.get("pose_mode", "standing"),
        "action_prompt": action_prompt,
        "joint_overshoot": True,
        "mannequin": str(mannequin),
        "character": str(char),
        "pose_strength_stage1": args.pose_strength_1,
        "pose_strength_stage2": args.pose_strength_2,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(logf, f"=== mannequin bridge run_id={run_id} ===")
    log(logf, f"source={src_id} joint_overshoot={meta_src.get('joint_overshoot')}")
    log(logf, f"mannequin={mannequin}")
    log(logf, f"character={char}")
    log(logf, f"size={out_w}x{out_h} seed={seed} labels={labels}")
    log(logf, f"action_prompt={action_prompt[:160]}")

    man_img = run_dir / "mannequin.png"
    char_img = run_dir / "character.png"
    client = ComfyClient()
    outputs: dict = {}
    ok = True

    try:
        for lab in labels:
            guide_skel = run_dir / f"{lab}_skel_guide.mp4"
            n = n_frames_from_guide_or_npz(src, lab, seed)
            length = align_4k1(n)
            log(logf, f"--- {lab}: stage1 skeleton -> mannequin (n={n} len={length}) ---")

            man_out = run_dir / f"{lab}_mannequin.mp4"
            fre = client.free_vram(
                interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0
            )
            log(logf, f"free {fre}")
            try:
                drive_character(
                    client,
                    guide_skel,
                    man_img,
                    man_out,
                    length=length,
                    width=out_w,
                    height=out_h,
                    pose_strength=args.pose_strength_1,
                    prefix=f"mp_mann1_{lab}",
                    seed=seed,
                    positive=MANNEQUIN_POSITIVE,
                    negative=MANNEQUIN_NEGATIVE,
                )
                outputs[f"{lab}_mannequin"] = str(man_out)
                log(logf, f"OK {man_out.name} bytes={man_out.stat().st_size}")
            except Exception as exc:
                ok = False
                outputs[f"{lab}_mannequin_error"] = str(exc)
                log(logf, f"FAIL stage1 {lab}: {exc}")
                log(logf, traceback.format_exc())
                continue

            # Stage 2: mannequin video drives character
            log(logf, f"--- {lab}: stage2 mannequin -> character ---")
            char_out = run_dir / f"{lab}_via_mannequin.mp4"
            pos = build_scail_positive(lab, action_prompt if lab == "action" else None)
            # Emphasize following body motion from mannequin drive (finished-video style).
            pos = (
                pos
                + " The character follows the body motion of the driving video "
                "precisely while keeping facial features frozen from the reference "
                "still: mouth closed, lips sealed, no talking, no lip motion."
            )
            fre = client.free_vram(
                interrupt=False, clear_queue=True, wait_s=20, min_free_gb=6.0
            )
            log(logf, f"free {fre}")
            try:
                drive_character(
                    client,
                    man_out,
                    char_img,
                    char_out,
                    length=length,
                    width=out_w,
                    height=out_h,
                    pose_strength=args.pose_strength_2,
                    prefix=f"mp_mann2_{lab}",
                    seed=seed,
                    positive=pos,
                    negative=SCAIL_NEGATIVE,
                )
                outputs[f"{lab}_via_mannequin"] = str(char_out)
                # Also mirror main names for convenience
                shutil.copy2(char_out, run_dir / f"{lab}.mp4")
                outputs[lab] = str(run_dir / f"{lab}.mp4")
                log(logf, f"OK {char_out.name} bytes={char_out.stat().st_size}")
            except Exception as exc:
                ok = False
                outputs[f"{lab}_via_mannequin_error"] = str(exc)
                log(logf, f"FAIL stage2 {lab}: {exc}")
                log(logf, traceback.format_exc())
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

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_run": src_id,
        "ok": ok,
        "joint_overshoot": True,
        "pipeline": "skeleton(+joint) -> mannequin -> character",
        "mannequin": str(mannequin),
        "character": str(char),
        "size": [out_w, out_h],
        "outputs": outputs,
    }
    sp = ROOT / "runs" / f"_mannequin_bridge_{run_id[:8]}_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(logf, f"=== DONE ok={ok} ===")
    log(logf, json.dumps(summary, indent=2, ensure_ascii=False))
    print(run_id)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
