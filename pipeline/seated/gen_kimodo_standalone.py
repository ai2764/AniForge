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
# Point the HF hub cache at a fast local dir for THIS Kimodo subprocess only, so
# the LLM2Vec text encoder (~15GB) + Kimodo weights load off a fast drive instead
# of the shared archive. Scoped here (before any HF import) so bgremove/HMR/other
# projects keep using the default cache. No-op if the env var is unset.
_kim_hf = os.environ.get("KIMODO_HF_HUB_CACHE")
if _kim_hf:
    os.environ["HF_HUB_CACHE"] = _kim_hf
    os.environ["HUGGINGFACE_HUB_CACHE"] = _kim_hf
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from pipeline.paths import comfy_output_dir, kimodo_package_dir

KIM = str(kimodo_package_dir())
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


def _clamp_constraint_json(cjson_path: str, n_frames: int) -> str:
    """Truncate a constraint's per-frame arrays to the generated clip length.

    Kimodo indexes ``m_sliced[frame_indices]`` against a tensor of length ``nf``;
    if the constraint has more frames than the clip (e.g. a 90-frame extract pin
    used for a 60-frame idle) it triggers a CUDA "index out of bounds" assert. The
    pins are static (same pose every frame), so truncating to ``n_frames`` is
    equivalent and safe. Returns the original path when no truncation is needed.
    """
    raw = json.loads(Path(cjson_path).read_text(encoding="utf-8"))
    changed = False
    for c in raw:
        for key in ("frame_indices", "local_joints_rot", "root_positions"):
            v = c.get(key)
            if isinstance(v, list) and len(v) > n_frames:
                c[key] = v[:n_frames]
                changed = True
    if not changed:
        return cjson_path
    out = Path(cjson_path).with_name(Path(cjson_path).stem + f"_n{n_frames}.json")
    out.write_text(json.dumps(raw), encoding="utf-8")
    return str(out)


def _legacy_job():
    cjson = sys.argv[1]
    idle_prefix = sys.argv[2] if len(sys.argv) > 2 else "kimodo_saidle"
    action_prefix = sys.argv[3] if len(sys.argv) > 3 else "kimodo_saaction"
    outdir = str(comfy_output_dir())
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

# Empty / missing constraint_json => free text-to-motion (used for standing).
_raw_cjson = job.get("constraint_json") or ""
CJSON = str(_raw_cjson).strip() if _raw_cjson else ""
if CJSON and not Path(CJSON).is_file():
    sys.exit(f"constraint_json not found: {CJSON}")
OUTDIR = Path(job.get("outdir") or comfy_output_dir())
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
print(f"constraint: {CJSON or '(none — free motion)'}", flush=True)

for item in JOBS:
    name = item["name"]
    prompt = item["prompt"]
    # Offset seed per clip name so idle/action with similar prompts still differ.
    clip_seed = SEED + (1 if name == "action" else 0)
    seed_everything(clip_seed)
    texts = sanitize_texts([prompt])
    nf = [int(DUR * model.fps)] * len(texts)
    cons = (
        load_constraints_lst(_clamp_constraint_json(CJSON, nf[0]), model.skeleton)
        if CJSON else []
    )
    t1 = time.time()
    out = model(
        texts, nf,
        num_denoising_steps=STEPS,
        num_samples=1,
        multi_prompt=len(texts) > 1,
        constraint_lst=cons if cons else None,
        post_processing=False,
        return_numpy=True,
    )
    single = {
        k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v)
        for k, v in out.items()
    }
    # Filename keeps session seed for stages to find; clip_seed only affects sampling.
    path = OUTDIR / f"{name}_seed{SEED}.npz"
    np.savez(path, **single)
    P = single["posed_joints"]
    motion = float(np.asarray(P).std(axis=0).mean())
    print(
        f"  {name}: gen {time.time()-t1:.0f}s seed={clip_seed} -> {path}  "
        f"posed_joints {P.shape} motion_std={motion:.4f}",
        flush=True,
    )

print("VRAM after both gens:", vram(), flush=True)

# Force CUDA release before process exit (Windows often holds memory otherwise).
try:
    del model
except Exception:
    pass
try:
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Some drivers need a second pass
        gc.collect()
        torch.cuda.empty_cache()
    print("VRAM after explicit unload:", vram(), flush=True)
except Exception as exc:
    print(f"VRAM unload warning: {exc}", flush=True)
