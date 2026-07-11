# Joint Overshoot as an Optional Standalone Step (Step-by-step)

**Date:** 2026-07-10
**Scope:** Step-by-step tab only. Run all is untouched.

## Problem

In the Step-by-step flow, joint overshoot is currently a button buried inside
step 4 (Action). Applying it destructively overwrites `action_guide.mp4` (the
file SCAIL reads) with no way to preview-then-decide and no clean way to back
out. The user wants joint overshoot to be its own **optional** step where you
**apply**, **preview** the result, and a **checkbox controls whether the
overshot skeleton carries into SCAIL**.

## Current behavior (baseline)

- `stage_action` (`pipeline/stages.py`) writes aligned joints to
  `action_seed{seed}.npz` and renders the **original** `action_skel.mp4` /
  `action_guide.mp4` from them.
- `stage_joint_overshoot` reads the same `action_seed{seed}.npz`, springs it,
  saves `action_joint_seed{seed}.npz`, and **overwrites** `action_skel.mp4` /
  `action_guide.mp4`. Sets `meta["joint_overshoot"]=True`.
- SCAIL always reads `action_guide.mp4` (fixed path).
- Client (`web/app.js`): `btn-joint` inside step 4 calls
  `doJointOvershoot()`; Run all reads the `overshoot-joint` checkbox.

The original joints (`action_seed{seed}.npz`) are always preserved, so a
non-destructive **revert** is feasible: re-render the guide from that npz.

## Design

### UI (`web/index.html`)

Insert a new `<section>` between step 4 (Action) and step 5 (SCAIL):

- Title: "Joint overshoot (optional)" with its own badge (locked → ready).
- Short description.
- Preview box reusing the action-skeleton preview pattern (shows the overshot
  skeleton video after apply).
- `Apply joint overshoot` button.
- A checkbox **"Carry into SCAIL"** — starts **disabled**; becomes enabled and
  auto-checked after a successful apply.

Move the existing `btn-joint` button and the "use checkbox on Run all tab" note
out of step 4 into this new section. Step 4 keeps only Action-motion controls.

### Interaction flow

1. After the action skeleton finishes, the Joint-overshoot section unlocks.
   The "Carry into SCAIL" checkbox is disabled (nothing applied yet).
2. Click `Apply` → POST `/session/joint-overshoot` → preview shows the overshot
   skeleton → checkbox becomes enabled and auto-checked. `action_guide.mp4` now
   holds the overshot guide.
3. The checkbox means "is the overshoot carried into SCAIL":
   - **Checked** → leave `action_guide.mp4` as the overshot version.
   - **Unchecked** → immediately **revert**: re-render `action_skel.mp4` /
     `action_guide.mp4` from `action_seed{seed}.npz`; preview switches back to
     the original skeleton. SCAIL will use the original.
4. Toggling the checkbox back on after a revert re-applies the overshoot.

Result: preview always reflects exactly what SCAIL will consume (WYSIWYG).

### Revert mechanism (chosen: immediate revert on uncheck)

Toggling the checkbox off runs the revert right away so the preview and the
on-disk guide stay in sync. Rejected alternative: "deferred revert" (reconcile
only when SCAIL action runs) — saves one render but lets the preview drift out
of sync with the final SCAIL input.

### Server (`pipeline/stages.py`, `server/app.py`)

Add a revert capability:

- New stage function `stage_joint_revert(run_id, *, runs_dir)`:
  - Re-render `action_skel.mp4` / `action_guide.mp4` from
    `action_seed{seed}.npz` (same render path as `stage_action`: `render_smplx_guide`
    then `_pad_to_aspect`), using the stored camera/size.
  - Set `meta["joint_overshoot"]=False`, `meta["action_scail_done"]=False`,
    `meta["scail_done"]=False`, `meta["step"]="action_skel"`.
  - Return the same shape as `stage_joint_overshoot` (skeleton url, n_frames,
    motion_std, errors) so the client can reuse `showActionSkel`.
- New route `/session/joint-revert` in `server/app.py` mirroring
  `_handle_session_joint_overshoot` (no Comfy needed).

To render from the original npz we need the camera. `stage_joint_overshoot`
already loads it via `_load_action_base_pose`; `stage_joint_revert` uses the
same source so frame0 matches the approved still.

### Client (`web/app.js`)

- New handlers/state for the section: `jointApplied` flag, apply button, the
  "Carry into SCAIL" checkbox.
- `Apply` handler: call `doJointOvershoot()` (existing), then enable + check the
  carry checkbox.
- Carry checkbox `change` handler:
  - unchecked → call new `doJointRevert()` (POST `/session/joint-revert`),
    update preview via `showActionSkel`.
  - checked → call `doJointOvershoot()` again, update preview.
- Guard: checkbox does nothing while `busy`; disabled until first apply.

## Out of scope

- Run all tab and its `overshoot-joint` checkbox (unchanged).
- Time overshoot.
- Any change to the overshoot spring math.

## Testing

- Existing `tests/test_server.py` / `tests/test_generate.py` cover stages;
  add coverage for `stage_joint_revert` restoring the original guide
  (`meta["joint_overshoot"]` flips back to False; guide re-rendered from
  `action_seed{seed}.npz`).
- Manual: run action skeleton → apply overshoot (preview changes) → uncheck
  (preview reverts) → SCAIL action uses the guide matching the checkbox state.
