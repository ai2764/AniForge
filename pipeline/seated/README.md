# Seated pose-anchored pipeline (smplx22)

Animate a **seated** character立绘 (idle + action) without SCAIL forcing it upright,
by anchoring Kimodo's motion to the image's actual seated pose.

Image → HMR2 pose extract → **butt-pin** constraint → Kimodo (idle + action) →
smplx22 skeleton guide → scail2 → seated animated立绘.

## Files
- `phase1_extract.py <image> <out_constraint.json> [T]` — HMR2 (4D-Humans) on the image →
  SMPL `body_pose` → smplx22 axis-angle → **ground** (feet at Y=0, pelvis at seat height) →
  writes an **all-frames** `end-effector` constraint pinning `["Hips","LeftFoot","RightFoot"]`
  (pelvis pos+rot + both ankle/foot positions). Legs/upper body stay free.
- `gen_kimodo_standalone.py <constraint.json> <idle_prefix> <action_prefix>` — loads
  `Kimodo-SMPLX-RP-v1` once, generates an idle + an action motion under the constraint,
  saves NPZs to the ComfyUI output dir. Pose-agnostic prompts (constraint owns the pose).
- `run_seated_pipeline.py [ref_image]` — the orchestrator (see VRAM strategy below).
- `fk_render_constraint.py` — diagnostic: FK-render a constraint through Kimodo's own path
  to verify the encoding reads as a real seated pose.

## VRAM strategy (no ComfyUI restart)
ComfyUI's `POST /free` does **not** free the Kimodo model (custom node, not comfy-managed).
So the two VRAM-heavy pre-steps run as **standalone subprocesses** — process exit auto-frees
their VRAM. ComfyUI only ever loads SCAIL (comfy-managed, `/free`-able between requests).

1. **Phase 1** `phase1_extract.py` subprocess — HMR2 (~11 GB) → exit frees it.
2. **Phase 2** `gen_kimodo_standalone.py` subprocess — Kimodo (~16 GB, load once, gen both) → exit frees it.
3. **Phase 3** SCAIL ×2 via `pipeline.scail.drive_character` (ComfyUI).

Measured: Phase1 11.4→8.0 GB on exit; Phase2 23.9→7.7 GB on exit. **Never load Kimodo through
a ComfyUI graph** — it wedges VRAM that only a server restart clears.

## Key findings (why it works)
- Encoding (SMPL body_pose → smplx22 axis-angle) is correct; verified by FK render.
- A single boundary keyframe [0] soft constraint (default cfg `[2,2]`) is too weak + frame 0
  is canonicalized → reverts to standing. Fix: constrain **all frames**.
- Extraction with `root_positions=[0,0,0]` puts feet below the floor → Kimodo's foot-ground
  prior lifts the body to standing. Fix: **ground** the target (raise pelvis to seat height).
- Constraint owns the pose → the text prompt should be **pose-agnostic** (no "seated"/"stands"),
  reusable for standing OR sitting inputs.

## Run
    cd motion-portrait
    python pipeline/seated/run_seated_pipeline.py [path/to/ref.png]

Requires ComfyUI-scail up on :8188 (SCAIL + ComfyUI-Kimodo nodes) and the comfy-scail env
(`MP_PYEXE` to override the interpreter for the subprocesses). Outputs land in `pipeline/seated/runs/`.
