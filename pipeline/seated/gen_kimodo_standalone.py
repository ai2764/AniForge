"""Standalone Kimodo generation (NO ComfyUI): load model once, gen idle+action with
the butt-pin constraint, save NPZs. Process exit frees all VRAM (auto-unload)."""
import sys, os, time, json, subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, KIM)
import numpy as np, torch

def vram():
    return subprocess.run(["nvidia-smi","--query-gpu=memory.used,power.draw","--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.tools import seed_everything
from kimodo.model.kimodo_model import sanitize_texts

CJSON = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/AIBOX/AppData/Local/Temp/claude/C--Users-AIBOX-dev-motion-portrait/a89c1fb2-249e-404d-b1e4-bb41d0c47b8f/scratchpad/sit_legpin_grounded_constraints.json"
IDLE_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "kimodo_saidle"
ACTION_PREFIX = sys.argv[3] if len(sys.argv) > 3 else "kimodo_saaction"
OUTDIR = r"C:/Users/AIBOX/dev/ComfyUI-scail/output"
DUR, STEPS, SEED = 3.0, 100, 42
JOBS = [(IDLE_PREFIX, "A person in a relaxed idle pose, breathing calmly, swaying gently, with small subtle movements of the head, arms and torso."),
        (ACTION_PREFIX, "A person gestures energetically, waving and raising both arms, leaning the upper body side to side, with lively expressive head and arm movement.")]

print("VRAM before load:", vram(), flush=True)
t0=time.time()
model = load_model("Kimodo-SMPLX-RP-v1", device="cuda", return_resolved_name=False)
print(f"model loaded in {time.time()-t0:.0f}s, skeleton={model.skeleton.name} fps={model.fps}", flush=True)
print("VRAM after load:", vram(), flush=True)

for prefix, prompt in JOBS:
    seed_everything(SEED)
    texts = sanitize_texts([prompt])
    nf = [int(DUR*model.fps)]*len(texts)
    cons = load_constraints_lst(CJSON, model.skeleton)
    t1=time.time()
    out = model(texts, nf, num_denoising_steps=STEPS, num_samples=1,
                multi_prompt=len(texts)>1, constraint_lst=cons,
                post_processing=False, return_numpy=True)
    # squeeze batch (num_samples=1), matching Kimodo_SaveNPZ
    single = {k:(v[0] if hasattr(v,"shape") and v.ndim>0 and v.shape[0]==1 else v) for k,v in out.items()}
    path = os.path.join(OUTDIR, f"{prefix}_seed{SEED}.npz")
    np.savez(path, **single)
    kb = None
    P = single["posed_joints"]
    print(f"  {prefix}: gen {time.time()-t1:.0f}s -> {path}  posed_joints {P.shape}", flush=True)

print("VRAM after both gens:", vram(), flush=True)
