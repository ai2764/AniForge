# Two-Column Step Layout + Non-Destructive Joint-Overshoot Preview

**Date:** 2026-07-11
**Scope:** Step-by-step tab only. Run all is untouched.

## Problem

Two related UI gaps in the Step-by-step flow:

1. **Layout.** Each step stacks controls and preview vertically, so the preview
   is pushed far down and steps read as a long single column. The user wants each
   step split horizontally: controls on the left, a preview window on the right
   showing that step's output.
2. **Joint overshoot.** It applies destructively (overwrites `action_guide.mp4`,
   the file SCAIL reads) with no way to preview-then-decide and no before/after
   comparison. The user wants a **Preview** button that renders the overshot
   skeleton from the current `omega/zeta/soft` sliders, a **two-up before/after**
   comparison, and a **Carry into SCAIL** checkbox that decides what SCAIL uses —
   all non-destructive to the original action skeleton.

## Current behaviour (baseline)

- `stage_action` writes aligned joints to `action_seed{seed}.npz` and renders the
  plain `action_skel.mp4` / `action_guide.mp4`.
- `stage_joint_overshoot(run_id, *, apply, omega, zeta, soft)` reads
  `action_seed{seed}.npz`, springs it (when `apply`), and **overwrites**
  `action_skel.mp4` / `action_guide.mp4`; `apply=False` re-renders them plain.
  Sets `meta["joint_overshoot"]`.
- Client: step 5 has 3 sliders + a `joint-carry` checkbox whose change handler
  calls `/session/joint-overshoot` with `apply=1|0` and the slider params.
- Every `.section` renders `.section-head` then controls then a `.preview-box`,
  all full-width in one column.

## Design

### A. Two-column step layout (`web/index.html`, `web/app.js`, inline CSS)

Restructure each step `<section>` body into two columns under the full-width
`.section-head`:

- `.step-body { display: grid; grid-template-columns: minmax(340px, 420px) 1fr; gap: 1rem; }`
- `.controls-col` — all inputs/sliders/buttons/hints for the step.
- `.preview-col` — the existing `.preview-box`(es), moved here unchanged.
- Responsive: `@media (max-width: 860px) { .step-body { grid-template-columns: 1fr; } }`
  so columns stack (controls above preview). Body never scrolls horizontally.

**Preview element IDs are unchanged** — only their wrapping `<div>` moves into
`.preview-col` — so existing `app.js` preview logic (`showActionSkel`, video src
setters, badges) keeps working with no JS changes beyond the joint step.

Right-column preview content per step (all already produced today):

| Step | Preview |
|------|---------|
| 2 Extract | `extract_skel.png` |
| 3 Idle | idle skeleton video |
| 4 Action | action skeleton video |
| 5 Joint overshoot | **two windows** (see B) |
| 6 SCAIL | idle + action character videos |
| 7 Background removal | no-bg previews |
| 8 Time overshoot | timed video |
| Preview | combined click-player |

### B. Joint-overshoot step: non-destructive preview + before/after

**Left column:** `omega` / `zeta` / `soft` sliders, a **Preview** button, and a
**Carry into SCAIL** checkbox.

**Right column:** two side-by-side windows
- "Original" — `action_skel.mp4` (the plain action skeleton; never overwritten).
- "Overshot preview" — `action_joint_skel.mp4` (empty until Preview is clicked).

**Artifacts (the key change):**
- `action_skel.mp4` — plain action skeleton. Overshoot NEVER overwrites it; it is
  the persistent "before".
- `action_joint_skel.mp4` — overshot preview video (the "after"). Produced by
  Preview. `action_joint_seed{seed}.npz` stores the sprung joints.
- `action_guide.mp4` — SCAIL input. Plain by default; rewritten to the overshot
  guide only while Carry is checked.

**Backend:** replace the `apply` boolean of `stage_joint_overshoot` with a
`mode` string with three values (server route `/session/joint-overshoot` reads
`mode` plus `joint_omega/zeta/soft`):

- `mode="preview"` — spring `action_seed{seed}.npz` with the given params, save
  `action_joint_seed{seed}.npz`, render `action_joint_skel.mp4` (padded like the
  skeleton preview). Do **not** touch `action_skel.mp4` / `action_guide.mp4`.
  Return the preview skeleton url + `n_frames` + `motion_std`.
- `mode="carry"` — render `action_guide.mp4` from `action_joint_seed{seed}.npz`
  (the last previewed overshoot). Set `meta["joint_overshoot"]=True`,
  `meta["action_scail_done"]=False`, `meta["scail_done"]=False`. If
  `action_joint_seed{seed}.npz` is missing (no prior preview — e.g. the Run-all
  one-shot path), first spring `action_seed{seed}.npz` with the given params and
  save it, then render the guide. This makes `carry` self-sufficient.
- `mode="uncarry"` — render `action_guide.mp4` from `action_seed{seed}.npz`
  (plain). Set `meta["joint_overshoot"]=False` and the same scail-invalidation
  flags. `action_skel.mp4` is already plain, so it is left untouched.

Rendering uses the same camera/size source as today (`_load_action_base_pose` +
`render_smplx_guide` + `_pad_to_aspect`) so frame0 matches the approved still.

**Interaction (`web/app.js`):**
1. After the action skeleton finishes, the Joint step unlocks. Preview enabled;
   Carry disabled (nothing previewed yet). Overshot window empty.
2. Click **Preview** → `mode=preview` with slider values → right window loads
   `action_joint_skel.mp4`. Carry becomes enabled (unchecked).
3. **Carry** checkbox change:
   - checked → `mode=carry` (guide becomes overshot; SCAIL will use overshoot).
   - unchecked → `mode=uncarry` (guide back to plain).
4. Adjusting a slider (on `change`, i.e. release) re-runs Preview; if Carry is
   currently checked, it also re-runs `carry` so the guide tracks the new params.
5. Guard: buttons/checkbox no-op while `busy`.

WYSIWYG: the "Overshot preview" window is exactly what SCAIL consumes whenever
Carry is checked; the "Original" window is always the plain skeleton for contrast.

### Run all (unchanged)

`btn-run-all` still reads the Run-all `overshoot-joint` checkbox. When checked it
calls `/session/joint-overshoot` with `mode="carry"` and default (or Run-all)
params; because `carry` self-springs when no preview npz exists, this is a single
call that produces the overshot guide for SCAIL. Run-all visuals/flow do not
change. Note: in Run all, `action_skel.mp4` stays plain and the overshoot lives in
`action_guide.mp4` — acceptable since Run all shows character videos, not the
skeleton preview.

## Out of scope

- Time overshoot step, background removal step (their previews just move columns).
- The overshoot spring math (`spring_follow`, `JOINT_SPRING`).
- Idle/SCAIL prompt content.

## Testing

- `tests/test_server.py` / `tests/test_generate.py`: add coverage that
  `mode="preview"` writes `action_joint_skel.mp4` + `action_joint_seed*.npz` and
  leaves `action_skel.mp4` / `action_guide.mp4` byte-identical; `mode="carry"`
  rewrites `action_guide.mp4` and flips `meta["joint_overshoot"]=True`;
  `mode="uncarry"` restores the plain guide and flips it back to False.
- Manual: run action → Preview (right window changes, left stays) → check Carry
  (guide = overshot) → uncheck (guide = plain) → move a slider (preview updates,
  and re-carries if checked) → SCAIL action consumes the guide matching Carry.
- Layout: verify two columns on desktop and clean vertical stacking under 860px
  with no horizontal body scroll.
