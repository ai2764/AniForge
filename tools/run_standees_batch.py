"""Batch-run motion-portrait on Live2D opening standees with per-character actions.

Order: session → extract → idle → action → SCAIL idle/action → bgremove.
"""
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

INPUT_DIR = Path(
    r"C:\Users\AIBOX\dev\youtube-video-lab\tasks\live2d\opening-images\立绘"
)
OUT_LOG = ROOT / "runs" / "_standees_batch_log.txt"
BATCH_SUMMARY = ROOT / "runs" / "_standees_batch_summary.json"

# Character-specific upper-body actions (~2s). English Kimodo prompts.
# Mouth closed; hips/feet fixed language for standing standees.
CHARACTERS: list[dict] = [
    {
        "file": "cyber runner.png",
        "slug": "cyber_runner",
        "action": (
            "Raise both hands briefly near chest height in a quick ready stance, "
            "elbows close to the torso. Then point the right hand to the side of "
            "frame at shoulder height with a sharp, compact motion (no lunging, "
            "no reaching toward camera). Small determined head tilt. Hips and "
            "feet fixed. Mouth closed and still."
        ),
    },
    {
        "file": "mage.png",
        "slug": "mage",
        "action": (
            # More dynamic opening: big arm arc first, then cast finish.
            "Start with a bold casting wind-up: sweep the right arm up high above "
            "the shoulder in a wide arc, elbow high, palm open as if gathering "
            "energy. Then thrust the open palm forward at chest height with a "
            "sharp cast motion and finish by closing the fingers. Expressive "
            "upper-body energy; shoulders and chest participate. Hips and feet "
            "fixed, no stepping. Mouth closed and still."
        ),
    },
    {
        "file": "maid.png",
        "slug": "maid",
        "action": (
            "Give a small head bow only; shoulders dip slightly; hips and feet "
            "stay fixed with no waist bend. Hands lightly together in front of "
            "the lower chest, then return upright with a small courteous nod. "
            "Graceful and compact; no large arm swings. Mouth closed and still."
        ),
    },
    {
        "file": "mecha polit.png",
        "slug": "mecha_pilot",
        "action": (
            "Bring the right hand up beside the temple in a crisp salute, hold "
            "briefly, then lower the hand to a firm ready position near the "
            "chest. Confident military bearing; torso forward; no body turn. "
            "Hips and feet fixed. Mouth closed and still."
        ),
    },
    {
        "file": "nurse.png",
        "slug": "nurse",
        "action": (
            "Raise the right hand to shoulder height in a gentle open-palm wave "
            "of reassurance, then rest the hand lightly over the heart area for "
            "a calm caring beat. Soft, measured motion; no frantic gestures. "
            "Hips and feet fixed. Mouth closed and still."
        ),
    },
]

SCALE = 1.0  # 720p-class: short side 720 (e.g. 720x1280 portrait standees)
SEED = 42
POSE_MODE = "standing"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ok(name: str, result: dict) -> bool:
    errs = result.get("errors") or {}
    if errs:
        log(f"FAIL {name}: {json.dumps(errs, ensure_ascii=False)}")
        return False
    log(f"OK {name}")
    return True


def free_vram(client: ComfyClient | None = None, min_free_gb: float = 6.0) -> None:
    try:
        c = client or ComfyClient()
        fre = c.free_vram(
            interrupt=False, clear_queue=True, wait_s=25, min_free_gb=min_free_gb
        )
        log(f"free_vram: {fre}")
    except Exception as e:
        log(f"free_vram skip: {e}")


def run_one(char: dict, client: ComfyClient) -> dict:
    image = INPUT_DIR / char["file"]
    slug = char["slug"]
    entry: dict = {
        "slug": slug,
        "file": char["file"],
        "action": char["action"],
        "status": "running",
    }
    if not image.is_file():
        entry["status"] = "failed"
        entry["error"] = f"missing image: {image}"
        log(entry["error"])
        return entry

    log(f"======== {slug} ({char['file']}) ========")
    free_vram(client)

    sess = create_session(
        image.read_bytes(),
        image.name,
        pose_mode=POSE_MODE,
        seed=SEED,
        scale=SCALE,
    )
    run_id = sess["run_id"]
    entry["run_id"] = run_id
    run_dir = ROOT / "runs" / run_id
    log(f"run_id={run_id} size={sess.get('size')}")

    if not ok("extract", stage_extract(run_id)):
        entry["status"] = "failed"
        entry["error"] = "extract"
        return entry

    if not ok("idle", stage_idle(run_id, idle_motion_keep=0.08)):
        entry["status"] = "failed"
        entry["error"] = "idle"
        return entry

    if not ok(
        "action",
        stage_action(
            run_id,
            action_prompt=char["action"],
            action_motion_keep=1.0,
            action_duration=2.0,
        ),
    ):
        entry["status"] = "failed"
        entry["error"] = "action"
        return entry

    try:
        client.object_info()
    except Exception as e:
        entry["status"] = "failed"
        entry["error"] = f"Comfy unreachable: {e}"
        log(entry["error"])
        return entry

    if not ok("scail_idle", stage_scail(run_id, which="idle", client=client)):
        entry["status"] = "failed"
        entry["error"] = "scail_idle"
        free_vram(client)
        return entry

    if not ok("scail_action", stage_scail(run_id, which="action", client=client)):
        entry["status"] = "failed"
        entry["error"] = "scail_action"
        free_vram(client)
        return entry

    free_vram(client, min_free_gb=6.0)
    r = stage_bgremove(run_id, which="both", model="RMBG-2.0 HQ")
    has = any(
        r.get(k)
        for k in (
            "idle_nobg",
            "action_nobg",
            "idle_nobg_webm",
            "action_nobg_webm",
        )
    )
    if r.get("errors"):
        log(f"bgremove warnings: {r['errors']}")
    if not has:
        log("FAIL bgremove: no outputs")
        entry["status"] = "failed"
        entry["error"] = "bgremove"
        free_vram(client)
        return entry
    log(f"OK bgremove idle={r.get('idle_nobg')} action={r.get('action_nobg')}")

    entry["status"] = "ok"
    entry["outputs"] = {
        "idle": str(run_dir / "idle.mp4"),
        "action": str(run_dir / "action.mp4"),
        "idle_nobg": str(run_dir / "idle_nobg.mp4"),
        "action_nobg": str(run_dir / "action_nobg.mp4"),
        "idle_nobg_webm": str(run_dir / "idle_nobg.webm"),
        "action_nobg_webm": str(run_dir / "action_nobg.webm"),
        "idle_nobg_alpha": str(run_dir / "idle_nobg_alpha.mov"),
        "action_nobg_alpha": str(run_dir / "action_nobg_alpha.mov"),
    }
    (run_dir / "standee_summary.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    free_vram(client)
    return entry


def main() -> int:
    OUT_LOG.write_text("", encoding="utf-8")  # fresh log for this run
    log("=== standees batch start ===")
    log(f"input_dir={INPUT_DIR}")
    if not INPUT_DIR.is_dir():
        log(f"ERROR: input dir missing: {INPUT_DIR}")
        return 2

    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    chars = CHARACTERS
    if only:
        want = set(only)
        chars = [c for c in CHARACTERS if c["slug"] in want]
        if not chars:
            log(f"ERROR: no matching slugs in {only}")
            return 3

    client = ComfyClient()
    try:
        client.object_info()
        log("Comfy reachable")
    except Exception as e:
        log(f"ERROR: ComfyUI unreachable: {e}")
        return 2

    free_vram(client)
    results: list[dict] = []
    for char in chars:
        try:
            results.append(run_one(char, client))
        except Exception:
            log(f"EXCEPTION {char['slug']}:\n{traceback.format_exc()}")
            results.append(
                {
                    "slug": char["slug"],
                    "file": char["file"],
                    "status": "failed",
                    "error": "exception",
                }
            )
            free_vram(client)

    summary = {
        "input_dir": str(INPUT_DIR),
        "scale": SCALE,
        "seed": SEED,
        "pose_mode": POSE_MODE,
        "results": results,
    }
    BATCH_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("=== BATCH DONE ===")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = sum(1 for r in results if r.get("status") != "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
