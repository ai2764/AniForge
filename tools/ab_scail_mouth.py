"""A/B SCAIL runs on an existing session (reuse skeleton motion).

Variants:
  noj     — action guide from non-springed action NPZ (no joint overshoot)
  serious — same no-joint guide + positive explicitly describing a serious face

Outputs are written alongside originals without overwriting idle.mp4/action.mp4:
  idle_noj.mp4, action_noj.mp4, idle_serious.mp4, action_serious.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.comfy import ComfyClient
from pipeline.generate import (
    SCAIL_NEGATIVE,
    align_4k1,
    build_scail_positive,
    _pad_to_aspect,
)
from pipeline.scail import drive_character
from pipeline.seated.generate_anchored import (
    render_smplx_guide,
    skeleton_camera_from_joints,
)


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def serious_positive(which: str, action_prompt: str | None = None) -> str:
    """Finished-video prompt with explicit serious/stoic face (official style)."""
    face = (
        "The face holds a serious, stern, stoic expression throughout: brows "
        "slightly drawn, eyes calm and focused, jaw set firm, mouth tightly "
        "closed with lips sealed flat in a thin line, no smile, no grin, no "
        "talking, no lip motion, no chewing, no open mouth, expression frozen "
        "and unchanging."
    )
    if which == "idle":
        return (
            "A full-body character matching the reference image stands facing a "
            "fixed frontal camera under soft even lighting, with clothing, "
            "hairstyle, colors, and proportions held consistent across every "
            "frame. The character keeps the same overall posture in a calm "
            "seamless idle loop: only tiny continuous chest breathing and a very "
            "slight head nod and sway, shoulders barely moving. Arms, hands, hips, "
            "and feet stay still and planted. "
            f"{face} "
            "Identity and wardrobe stay locked to the reference throughout."
        )
    # action
    from pipeline.generate import _strip_scail_action_for_embed

    motion = _strip_scail_action_for_embed(action_prompt or "")
    motion_bit = (
        f"In this clip the character moves as follows: {motion.rstrip('.')}."
        if motion
        else "The character performs one clear upper-body action."
    )
    return (
        "A full-body character matching the reference image stands facing a "
        "fixed frontal camera under soft even lighting, with clothing, "
        "hairstyle, colors, and proportions held consistent across every "
        "frame. "
        f"{motion_bit} "
        "Hips and feet remain planted; no walking or turning that breaks the "
        "frontal framing. Motion stays limited to the described limbs without "
        "extra gestures. "
        f"{face} "
        "Identity and wardrobe stay locked to the reference throughout."
    )


def rebuild_noj_action_guide(run_dir: Path, seed: int, out_w: int, out_h: int) -> Path:
    """Render action_guide from action_seed{N}.npz (no spring joint overshoot)."""
    npz = run_dir / f"action_seed{seed}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    P = np.load(npz)["posed_joints"]
    base_path = run_dir / "extract_pose.npy"
    cam = None
    if base_path.is_file():
        base = np.load(base_path)
        cam = skeleton_camera_from_joints(base)
    skel = run_dir / "action_skel_noj.mp4"
    guide = run_dir / "action_guide_noj.mp4"
    render_smplx_guide(P, skel, camera=cam)
    _pad_to_aspect(skel, guide, out_w, out_h)
    return guide


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="1a93a60250a443539c1381ec5e99b547")
    ap.add_argument(
        "--variants",
        default="noj,serious",
        help="comma list: noj, serious",
    )
    args = ap.parse_args()

    run_id = args.run_id.strip()
    run_dir = ROOT / "runs" / run_id
    logf = ROOT / "runs" / f"_ab_scail_mouth_{run_id[:8]}_log.txt"
    logf.write_text("", encoding="utf-8")

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    seed = int(meta.get("seed") or 42)
    size = meta.get("size") or [720, 1280]
    out_w, out_h = int(size[0]), int(size[1])
    action_prompt = meta.get("action_prompt") or ""
    image = run_dir / "input.png"
    if not image.is_file():
        for p in run_dir.iterdir():
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and "skel" not in p.name:
                image = p
                break

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log(logf, f"=== A/B SCAIL mouth run_id={run_id} variants={variants} ===")
    log(logf, f"image={image} size={out_w}x{out_h} seed={seed}")
    log(logf, f"action_prompt={action_prompt[:160]}")

    idle_guide = run_dir / "idle_guide.mp4"
    if not idle_guide.is_file():
        idle_guide = run_dir / "idle_skel.mp4"
    if not idle_guide.is_file():
        log(logf, "FAIL missing idle guide")
        return 2

    log(logf, "rebuild no-joint action guide...")
    action_guide_noj = rebuild_noj_action_guide(run_dir, seed, out_w, out_h)
    log(logf, f"action_guide_noj={action_guide_noj} bytes={action_guide_noj.stat().st_size}")

    idle_npz = run_dir / f"idle_seed{seed}.npz"
    action_npz = run_dir / f"action_seed{seed}.npz"
    idle_n = int(np.load(idle_npz)["posed_joints"].shape[0]) if idle_npz.is_file() else 61
    action_n = int(np.load(action_npz)["posed_joints"].shape[0]) if action_npz.is_file() else 61

    client = ComfyClient()
    results: dict = {}
    try:
        for var in variants:
            if var == "noj":
                pos_idle = build_scail_positive("idle")
                pos_action = build_scail_positive("action", action_prompt)
                tag = "noj"
            elif var == "serious":
                pos_idle = serious_positive("idle")
                pos_action = serious_positive("action", action_prompt)
                tag = "serious"
            else:
                log(logf, f"skip unknown variant {var}")
                continue

            # Both use no-joint action guide so motion is controlled.
            jobs = [
                ("idle", idle_guide, pos_idle, idle_n),
                ("action", action_guide_noj, pos_action, action_n),
            ]
            log(logf, f"--- variant={tag} ---")
            log(logf, f"idle_positive words={len(pos_idle.split())}")
            log(logf, f"action_positive words={len(pos_action.split())}")
            log(logf, pos_action[:280] + "...")

            for label, guide, positive, n_frames in jobs:
                out_mp4 = run_dir / f"{label}_{tag}.mp4"
                log(logf, f"SCAIL {label}_{tag} guide={guide.name} -> {out_mp4.name}")
                try:
                    fre = client.free_vram(
                        interrupt=False, clear_queue=True, wait_s=25, min_free_gb=6.0
                    )
                    log(logf, f"free {fre}")
                    drive_character(
                        client,
                        guide,
                        image,
                        out_mp4,
                        length=align_4k1(n_frames),
                        width=out_w,
                        height=out_h,
                        prefix=f"mp_ab_{tag}_{label}",
                        seed=seed,
                        positive=positive,
                        negative=SCAIL_NEGATIVE,
                    )
                    results[f"{label}_{tag}"] = str(out_mp4)
                    log(logf, f"OK {out_mp4.name} size={out_mp4.stat().st_size}")
                except Exception as exc:
                    results[f"{label}_{tag}_error"] = str(exc)
                    log(logf, f"FAIL {label}_{tag}: {exc}")
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
        "variants": variants,
        "note": "both variants use no-joint action guide; serious adds face wording",
        "outputs": results,
    }
    sp = ROOT / "runs" / f"_ab_scail_mouth_{run_id[:8]}_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(logf, "=== DONE ===")
    log(logf, json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not any(k.endswith("_error") for k in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
