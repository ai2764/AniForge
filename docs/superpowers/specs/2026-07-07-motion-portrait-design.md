# Motion Portrait — Design

> Status: design (brainstormed 2026-07-07). Standalone repo, separate from camera-lab.
> Implementation plan to follow via writing-plans.

## Goal

A minimal standalone tool that turns **one character image** into an animated
live2d-style portrait: it plays a looping **idle** and, on mouse click, plays a
one-shot **action**, then returns to idle. A single "generate" run produces
**two videos** (idle loop + action clip) from the image plus text prompts.

## Why

The animation pipeline (Kimodo text-to-motion → joint-space spring → scail2
character drive → optional time-space spring) already exists as scripts inside
camera-lab, but there is no simple product surface for it. This tool wraps the
pipeline behind a dead-simple UI and keeps it **independent from camera-lab**,
sharing only the ComfyUI backend (scail:8188).

## Scope

### MVP (in)
- Single-page UI: upload image, action prompt, optional idle prompt, two
  overshoot checkboxes, Generate button, and a click-triggered preview player.
- One run → two mp4s: `idle.mp4` (loops) and `action.mp4` (plays once on click).
- Minimal Python backend (own repo) that orchestrates the pipeline against the
  shared ComfyUI at 8188.
- Overshoot options apply to the **action** clip only; both can be selected and
  they stack.

### Out (v2, explicitly deferred)
- Seamless idle↔action **anchor closure** (first≈last frame). MVP uses a simple
  loop + click-to-play-once + return (hard or cross-fade cut).
- Kimodo **kinematic pose constraints** (pinning first/last frame to a shared
  anchor pose).
- Multiple actions per character, action library, custom idle beyond the default.

## Global constraints
- **All product text (code, comments, string literals, UI, committed data) is
  English only — no Chinese.**
- Independent from camera-lab: separate repo, separate process, own port. The
  only shared dependency is ComfyUI at `http://127.0.0.1:8188`.
- Reuse the existing pipeline scripts (`skeleton_spring.py`, `spring_time_remap.py`)
  — vendor copies into this repo (do not import across repos).

## UI

Single static HTML page, no framework:

```
[ drop / upload character image ]
Action prompt:      [________________]
Idle prompt (opt):  [________________]   (blank = default idle)
Overshoot:          [ ] joint   [ ] time
[ Generate ]
-------------------------------------------------
Preview: portrait loops the idle video;
         mouse click plays action once, then back to idle.
```

Default idle prompt (when the idle field is blank): a relaxed in-place idle —
calm breathing, gentle side-to-side sway, small head/arm movement, feet planted.

## Architecture (three units)

1. **Frontend** (`web/index.html` + a small `app.js`): the form, upload, and the
   click player (an idle `<video loop>` with an action `<video>` swapped in on
   click, reverting on `ended`). Zero framework, self-contained.

2. **Backend** (`server/app.py`, Python `http.server`): one endpoint
   `POST /generate` that accepts the image + prompts + overshoot options,
   orchestrates the pipeline, and returns the two output video URLs. Serves the
   static frontend and the generated videos. Independent process on its own port
   (default 8500). Talks to ComfyUI at 8188 over HTTP; stages inputs into
   ComfyUI's `input/` dir and reads results via `/view` (same approach the
   camera-lab pipeline used).

3. **Pipeline module** (`pipeline/`): the reusable steps, vendored from camera-lab
   and adapted:
   - `kimodo.py` — submit a Kimodo (SOMA) text-to-motion graph to ComfyUI, save
     NPZ (joint positions `[T, J, 3]`).
   - `skeleton_spring.py` — render the SOMA 30-joint skeleton with fixed framing;
     optional joint-space damped-spring overshoot (adaptive-substep integrator).
   - `scail.py` — submit a scail2 graph driving the character image with a
     skeleton guide, collect the output video.
   - `spring_time_remap.py` — optional time-space overshoot on the rendered video
     (chosen params B0.42 D4.2 F2.4 T1.15).

## Data flow (one `/generate`)

```
image + idle_prompt + action_prompt + {joint?, time?}
  │
  ├─ idle:   Kimodo(idle_prompt) → skeleton_spring(SOMA, fixed frame, NO overshoot)
  │            → scail2(image) → idle.mp4
  │
  └─ action: Kimodo(action_prompt) → skeleton_spring(SOMA, fixed frame,
  │            + joint overshoot IF joint) → scail2(image)
  │            → (+ time overshoot IF time) → action.mp4
  │
  → { idle: idle.mp4, action: action.mp4 }
```

Overshoot only affects the action branch. Joint overshoot is applied in the
skeleton (before scail); time overshoot is applied to the final rendered video.
When both are checked they stack.

## Prerequisites (must land before the MVP can run)

These are enabling work, tracked as the first tasks of the plan:

1. **Kimodo operational** on ComfyUI-scail: node `jtydhr88/ComfyUI-Kimodo` is
   installed and loaded; `Kimodo-SOMA-RP-v1` (public) + the shared Llama-3-8B
   encoder (gated, access granted) download and generate an NPZ.
2. **SOMA 30-joint renderer**: adapt `skeleton_spring.py` from the SMPL 22-joint
   layout to SOMA's 30-joint skeleton (bone topology already extracted from
   `kimodo/skeleton/definitions.py` — `SOMASkeleton30.bone_order_names_with_parents`).
   The face/finger/toe joints (jaw, eyes, thumb/middle tips, toes) are extra vs
   SMPL and are not needed for scail driving; render the body chain.

## Error handling

- **Preflight**: before generating, probe `8188/object_info`; if Kimodo or SCAIL
  node classes are missing, fail fast with a clear message (do not spin).
- **Per-clip isolation**: if idle or action fails in ComfyUI, return that clip's
  error plus the other (successful) clip — do not fail the whole run.
- **Turning guard**: fixed-frame skeleton rendering is always on (per-frame
  recentering caused scail to read the body as turning). Light prompt hygiene on
  the action (avoid "turning"; note forward-reach ambiguity) without rewriting
  user intent.
- **Input validation**: image format/size, non-empty action prompt, valid
  overshoot flags.

## Testing

- **Pure units**: SOMA 30-joint bone/index mapping; `skeleton_spring` integrator
  stability (adaptive substeps, no blow-up); `spring_time_remap` params (reuse
  existing tests); overshoot-flags → pipeline-steps composition logic.
- **End-to-end** (host-manual, not CI): one test image through `/generate`,
  assert two valid mp4s (frame count / resolution) against real ComfyUI 8188.
- **Player interaction**: manual (loop idle, click plays action, returns).

## Risks / notes

- Kimodo first-run downloads ~16GB (Llama-3 encoder, shared); the SOMA checkpoint
  is ~1GB. VRAM ~17GB fits the RTX 4090.
- SOMA (30-joint) vs SMPL (22): chose SOMA because it is public (no gate) and
  NVIDIA-primary; the renderer adaptation is the cost. Downstream scail is
  format-tolerant (drove from HY-Motion custom skeletons and from RGB body video).
- Forward-reach (Z-depth) actions are ambiguous in the 2D front projection and
  can still make scail turn the body; this is a content/prompt limitation, not a
  bug, and is out of scope to fully solve in MVP.
- The dramatic L2D bounce (hair/skirt/chest) is secondary soft-body motion absent
  from the skeleton; neither overshoot method produces it. Out of scope.
