"""Standalone Kimodo generation (NO ComfyUI): load model once, gen clips under a
pose-lock constraint, save NPZs. Process exit frees all VRAM (auto-unload).

Primary interface: argv[1] = job.json
{
  "constraint_json": "...",
  "outdir": "...",
  "seed": 42,
  "duration": 3.0,
  "steps": 100,
  "jobs": [{"name": "idle", "prompt": "..."}, {"name": "action", "prompt": "..."}]
}

Legacy CLI (still works):
  <constraint.json> <idle_prefix> <action_prefix>
  writes to ComfyUI-scail/output/{prefix}_seed42.npz
"""
import sys
import os
import time
import json
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, KIM)
import numpy as np
import torch


def vram():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()


from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.tools import seed_everything
from kimodo.model.kimodo_model import sanitize_texts


def _legacy_job():
    cjson = sys.argv[1]
    idle_prefix = sys.argv[2] if len(sys.argv) > 2 else "kimodo_saidle"
    action_prefix = sys.argv[3] if len(sys.argv) > 3 else "kimodo_saaction"
    outdir = r"C:/Users/AIBOX/dev/ComfyUI-scail/output"
    return {
        "constraint_json": cjson,
        "outdir": outdir,
        "seed": 42,
        "duration": 3.0,
        "steps": 100,
        "jobs": [
            {
                "name": idle_prefix,
                "prompt": (
                    "A person holds their current pose in a relaxed idle, breathing calmly, "
                    "with only tiny subtle micro-movements of the head and torso. "
                    "Keep the same overall posture; no large joint rotations."
                ),
            },
            {
                "name": action_prefix,
                "prompt": (
                    "A person gestures energetically, waving and raising both arms, "
                    "leaning the upper body side to side, with lively expressive head and arm movement."
                ),
            },
        ],
    }


if len(sys.argv) < 2:
    sys.exit("usage: gen_kimodo_standalone.py <job.json> | <constraint.json> <idle_prefix> <action_prefix>")

arg1 = Path(sys.argv[1])
if arg1.suffix.lower() == ".json" and arg1.is_file():
    raw = json.loads(arg1.read_text(encoding="utf-8"))
    # job file if it has "jobs"; else treat as bare constraint (legacy path needs prefixes)
    if isinstance(raw, dict) and "jobs" in raw:
        job = raw
    else:
        job = _legacy_job()
else:
    job = _legacy_job()

CJSON = job["constraint_json"]
OUTDIR = Path(job.get("outdir") or r"C:/Users/AIBOX/dev/ComfyUI-scail/output")
OUTDIR.mkdir(parents=True, exist_ok=True)
SEED = int(job.get("seed") or 42)
DUR = float(job.get("duration") or 3.0)
STEPS = int(job.get("steps") or 100)
JOBS = job["jobs"]

print("VRAM before load:", vram(), flush=True)
t0 = time.time()
model = load_model("Kimodo-SMPLX-RP-v1", device="cuda", return_resolved_name=False)
print(
    f"model loaded in {time.time()-t0:.0f}s, skeleton={model.skeleton.name} fps={model.fps}",
    flush=True,
)
print("VRAM after load:", vram(), flush=True)

for item in JOBS:
    name = item["name"]
    prompt = item["prompt"]
    seed_everything(SEED)
    texts = sanitize_texts([prompt])
    nf = [int(DUR * model.fps)] * len(texts)
    cons = load_constraints_lst(CJSON, model.skeleton)
    t1 = time.time()
    out = model(
        texts, nf,
        num_denoising_steps=STEPS,
        num_samples=1,
        multi_prompt=len(texts) > 1,
        constraint_lst=cons,
        post_processing=False,
        return_numpy=True,
    )
    single = {
        k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v)
        for k, v in out.items()
    }
    path = OUTDIR / f"{name}_seed{SEED}.npz"
    np.savez(path, **single)
    P = single["posed_joints"]
    print(f"  {name}: gen {time.time()-t1:.0f}s -> {path}  posed_joints {P.shape}", flush=True)

print("VRAM after both gens:", vram(), flush=True)
