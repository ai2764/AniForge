"""Re-apply mouth lock + RMBG + time overshoot on cyber_runner run."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.face_lock import lock_mouth_in_video
from pipeline.stages import stage_bgremove, stage_time_overshoot
from pipeline.comfy import ComfyClient

RUN = Path("runs/b15d20346e3c4cc08d356972d6837f55")
assert RUN.is_dir(), RUN

for label in ("idle", "action"):
    p = RUN / f"{label}.mp4"
    print("mouth lock", p)
    lock_mouth_in_video(p, in_place=True, strength=1.0)
    print(" ok", p.stat().st_size)

try:
    c = ComfyClient()
    print("free", c.free_vram(interrupt=False, clear_queue=True, wait_s=15, min_free_gb=4.0))
except Exception as e:
    print("free skip", e)

print("bgremove RMBG-2.0 HQ...")
r = stage_bgremove(RUN.name, which="both", model="RMBG-2.0 HQ")
print("bg", {k: r.get(k) for k in r if "nobg" in k or k == "errors"})

print("time overshoot...")
t = stage_time_overshoot(RUN.name)
print("time", {k: t.get(k) for k in t if k in ("action","action_timed","action_nobg_alpha","has_alpha","errors","warnings","time_source")})
print("DONE")
