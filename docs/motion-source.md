# Motion Source — Kimodo-SOMA (operational)

Kimodo is the mandatory motion source for this project (no HY-Motion fallback).
As of 2026-07-08 it is operational on ComfyUI-scail:8188.

## What works

- Node pack: `jtydhr88/ComfyUI-Kimodo` (9 nodes) in `ComfyUI-scail/custom_nodes`.
- Checkpoint: `Kimodo-SOMA-RP-v1` (public, ~1GB). Text encoder: LLM2Vec over
  `meta-llama/Meta-Llama-3-8B-Instruct` (gated — access granted for `ai2764`;
  ~15GB cached at `F:/AIModelArchive/CDriveOffload_20260514/caches/huggingface/hub`).
- Comfy env python: `C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe`.
- Graph: `Kimodo_LoadModel → Kimodo_TextEncode → Kimodo_Sampler → Kimodo_PostProcess → Kimodo_SaveNPZ`.
  A 3.0s / 50-step SOMA generation runs in ~56s on the RTX 4090.

## NPZ output format (from `Kimodo_SaveNPZ`)

`Kimodo_SaveNPZ` writes to the ComfyUI output dir. Keys:

| key | shape | note |
| --- | --- | --- |
| `posed_joints` | `(T, 77, 3)` | **joint positions — use this for rendering.** 77-joint SOMASkeleton77 (full finger/toe/face detail) |
| `local_rot_mats` | `(T, 77, 3, 3)` | local joint rotations |
| `global_rot_mats` | `(T, 77, 3, 3)` | global joint rotations |
| `root_positions` | `(T, 3)` | root translation |
| `smooth_root_pos` | `(T, 3)` | smoothed root |
| `foot_contacts` | `(T, 6)` bool | foot contact flags |
| `global_root_heading` | `(T, 2)` | root heading |

`T` = duration * 30 (30 fps; 3.0s → 90 frames).

**Important for the skeleton renderer (plan Task 3):** the NPZ exposes the
**77-joint** `posed_joints`, NOT the 30-joint internal skeleton. Build the bone
topology from `ComfyUI-Kimodo/kimodo/kimodo/skeleton/definitions.py`
`SOMASkeleton77` and render the **body chain** (drop fingers/toes/face for the
scail guide — they are not needed to drive scail and add clutter).

## Dependency resolution (comfy-scail env)

The encoder path (LLM2Vec → transformers 5.1.0 → optimum → gptqmodel) was broken.
Fixes applied (none touched `torch 2.12.0+cu130` or `transformers 5.1.0`):

1. `pip install optimum` (2.2.0), `logbar`, `threadpoolctl`, `device_smi`,
   `tokenicer`, `defuser`.
2. Added a stub `pcre.py` in site-packages that aliases the stdlib `re` module
   (gptqmodel's logger only uses `pcre.compile` + `pcre.Flag`; the real PCRE C
   library is unnecessary and painful on Windows).
3. gptqmodel 7.1.0 vs transformers 5.1.0 name skew: transformers expects
   `Awq*QuantLinear` classes but gptqmodel 7.1.0 names them `Awq*Linear`. Added
   compat aliases (`AwqGEMMQuantLinear = AwqGEMMLinear`, etc.) to the 7 AWQ
   modules under `gptqmodel/nn_modules/qlinear/`.
4. Launch comfy with `PYTHONIOENCODING=utf-8` (logbar prints box-drawing chars
   that crash a GBK console).

These are host-environment patches, not repo code. If the comfy env is rebuilt,
re-apply steps 2–4 (or install a `python-pcre` that builds on Windows, and a
gptqmodel/transformers pair whose AWQ class names agree).
