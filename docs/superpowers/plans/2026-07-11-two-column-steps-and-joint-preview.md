# Two-Column Steps + Joint-Overshoot Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each step-by-step section a two-column layout (controls left, preview right) and rework joint overshoot into a non-destructive Preview + before/after comparison controlled by a Carry-into-SCAIL checkbox.

**Architecture:** Backend `stage_joint_overshoot` gains a three-value `mode` (`preview`/`carry`/`uncarry`) that never overwrites `action_skel.mp4`: `preview` renders the overshot to a separate `action_joint_skel.mp4`; `carry`/`uncarry` only re-pad the overshot or plain skeleton into `action_guide.mp4` (the SCAIL input). Frontend wraps each section body in a CSS grid and wires the joint step to Preview/Carry.

**Tech Stack:** Python 3.12 (stdlib `http.server`, numpy), vanilla JS + HTML + inline CSS. Heavy render deps (`render_smplx_guide`, `_pad_to_aspect`, `spring_follow`) run via `COMFY_PYTHON`; unit tests monkeypatch them.

## Global Constraints

- Step-by-step tab only. Run all behaviour and visuals stay unchanged (it calls `mode="carry"`, which self-springs when no preview npz exists).
- `action_skel.mp4` is NEVER overwritten by overshoot — it is the persistent "before".
- Preview element IDs already used by `app.js` must not change; only their wrapper `<div>` moves into `.preview-col`.
- Body must never scroll horizontally; columns stack under `max-width: 860px`.
- Server launch requires `COMFY_PYTHON=C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe`; restart via the PowerShell Start-Process pattern (see repo `_server_out.log`). Run pytest with that interpreter.

---

### Task 1: Backend — `mode` (preview/carry/uncarry) on `stage_joint_overshoot` + route

**Files:**
- Modify: `pipeline/stages.py:693-762` (replace `apply` param with `mode`)
- Modify: `server/app.py` (`_handle_session_joint_overshoot`, ~line 412)
- Test: `tests/test_stage_joint.py` (create)

**Interfaces:**
- Consumes (existing, do not change): `_load_meta(run_dir)->dict`, `_save_meta(run_dir, meta)`, `_rel_url(path)->str|None`, `_load_action_base_pose(run_dir, seed)->(base, cam, src)`, `_output_size`, `_find_image`, `spring_follow`, `align_motion_to_base_pose`, `render_smplx_guide(P, out, camera=)`, `_pad_to_aspect(inp, out, out_w, out_h)`, `JOINT_SPRING`, `FPS`.
- Produces: `stage_joint_overshoot(run_id, *, mode="preview", omega=None, zeta=None, soft=None, runs_dir=RUNS_DIR) -> dict`. Return keys: `run_id`, `errors` (dict), `mode`, `seed`, `size`; on `preview` also `skeleton` (url of `action_joint_skel.mp4`), `n_frames`, `motion_std`; on `carry`/`uncarry` also `joint_overshoot` (bool), `guide` (url of `action_guide.mp4`).
- Artifacts: `preview` → `action_joint_skel.mp4` + `action_joint_seed{seed}.npz`; `carry`/`uncarry` → `action_guide.mp4`. `action_skel.mp4` untouched by all three.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_joint.py`:

```python
import json
import numpy as np
import pipeline.stages as stages


def _make_run(tmp_path, monkeypatch):
    run_id = "r1"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    seed = 42
    (run_dir / "meta.json").write_text(json.dumps(
        {"seed": seed, "scale": 1.0, "size": [64, 128],
         "pose_mode": "standing", "action_done": True}), encoding="utf-8")
    np.savez(run_dir / f"action_seed{seed}.npz",
             posed_joints=np.zeros((4, 22, 3), dtype=np.float64))
    (run_dir / "action_skel.mp4").write_bytes(b"PLAIN")  # the persistent "before"
    (run_dir / "input.png").write_bytes(b"x")  # _find_image is called before size check

    # Neutralize heavy render/spring: create the output file, identity spring.
    monkeypatch.setattr(stages, "spring_follow",
                        lambda P, fps, **k: P + 1.0)
    monkeypatch.setattr(stages, "render_smplx_guide",
                        lambda P, out, camera=None: out.write_bytes(b"OVERSHOT"))
    monkeypatch.setattr(stages, "_pad_to_aspect",
                        lambda inp, out, w, h, **k: out.write_bytes(inp.read_bytes()))
    monkeypatch.setattr(stages, "_load_action_base_pose",
                        lambda rd, s: (None, None, "extract"))
    return run_dir, seed


def test_preview_is_nondestructive(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    out = stages.stage_joint_overshoot("r1", mode="preview", runs_dir=tmp_path)
    assert out["errors"] == {}
    assert (run_dir / "action_joint_skel.mp4").read_bytes() == b"OVERSHOT"
    assert (run_dir / f"action_joint_seed{seed}.npz").is_file()
    # before-file and SCAIL guide untouched
    assert (run_dir / "action_skel.mp4").read_bytes() == b"PLAIN"
    assert not (run_dir / "action_guide.mp4").is_file()


def test_carry_then_uncarry_toggles_guide_and_meta(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    stages.stage_joint_overshoot("r1", mode="preview", runs_dir=tmp_path)
    c = stages.stage_joint_overshoot("r1", mode="carry", runs_dir=tmp_path)
    assert c["errors"] == {}
    assert c["joint_overshoot"] is True
    assert (run_dir / "action_guide.mp4").read_bytes() == b"OVERSHOT"  # from joint skel
    assert json.loads((run_dir / "meta.json").read_text())["joint_overshoot"] is True

    u = stages.stage_joint_overshoot("r1", mode="uncarry", runs_dir=tmp_path)
    assert u["joint_overshoot"] is False
    assert (run_dir / "action_guide.mp4").read_bytes() == b"PLAIN"  # from action_skel
    assert json.loads((run_dir / "meta.json").read_text())["joint_overshoot"] is False


def test_carry_self_springs_without_preview(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    # No preview call first (Run-all one-shot path).
    c = stages.stage_joint_overshoot("r1", mode="carry", runs_dir=tmp_path)
    assert c["errors"] == {}
    assert (run_dir / f"action_joint_seed{seed}.npz").is_file()  # sprang on demand
    assert (run_dir / "action_guide.mp4").read_bytes() == b"OVERSHOT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe" -m pytest tests/test_stage_joint.py -q`
Expected: FAIL — `stage_joint_overshoot() got an unexpected keyword argument 'mode'`.

- [ ] **Step 3: Write minimal implementation**

Replace `stage_joint_overshoot` (`pipeline/stages.py:693-762`) with:

```python
def stage_joint_overshoot(
    run_id: str,
    *,
    mode: str = "preview",
    omega: float | None = None,
    zeta: float | None = None,
    soft: float | None = None,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Non-destructive joint-overshoot on the action skeleton.

    mode="preview": spring raw ``action_seed`` with the given params, save
        ``action_joint_seed`` + render ``action_joint_skel.mp4``. Never touches
        ``action_skel.mp4`` / ``action_guide.mp4``.
    mode="carry": (re)pad the overshot skeleton into ``action_guide.mp4`` so SCAIL
        uses it. Self-springs a preview first if none exists (Run-all one-shot).
    mode="uncarry": re-pad the plain ``action_skel.mp4`` into ``action_guide.mp4``.
    """
    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)
    if not meta.get("action_done"):
        return {"run_id": run_id, "errors": {"joint": "run action skeleton first"}, "mode": mode}

    seed = meta["seed"]
    scale = meta.get("scale", 1.0)
    image = _find_image(run_dir)
    out_w, out_h = meta.get("size") or _output_size(image, scale=scale)
    out: dict = {"run_id": run_id, "errors": {}, "mode": mode, "seed": seed, "size": [out_w, out_h]}

    raw = run_dir / f"action_seed{seed}.npz"
    joint_npz = run_dir / f"action_joint_seed{seed}.npz"
    joint_skel = run_dir / "action_joint_skel.mp4"
    action_skel = run_dir / "action_skel.mp4"
    guide = run_dir / "action_guide.mp4"

    def _spring_preview():
        P = np.asarray(np.load(raw)["posed_joints"], dtype=np.float64)
        P = spring_follow(
            P, FPS,
            omega=JOINT_SPRING["omega"] if omega is None else float(omega),
            zeta=JOINT_SPRING["zeta"] if zeta is None else float(zeta),
            soft_scale=JOINT_SPRING["soft"] if soft is None else float(soft),
        )
        pose_mode = meta.get("pose_mode", "standing")
        base, cam, _src = _load_action_base_pose(run_dir, seed)
        if base is not None:
            P = align_motion_to_base_pose(
                P, base, keep=1.0, lock_lower_body=pose_mode in ("sitting", "lying"),
            )
        np.savez(joint_npz, posed_joints=P)
        render_smplx_guide(P, joint_skel, camera=cam)
        return P

    try:
        if not raw.is_file():
            out["errors"]["joint"] = "missing action_seed npz — re-run action motion"
            return out

        if mode == "preview":
            P = _spring_preview()
            out["skeleton"] = _rel_url(joint_skel)
            out["n_frames"] = int(P.shape[0])
            out["motion_std"] = float(np.asarray(P).std(axis=0).mean())
            return out

        if mode == "carry":
            if not joint_skel.is_file():
                _spring_preview()  # self-spring for the Run-all one-shot path
            _pad_to_aspect(joint_skel, guide, out_w, out_h)
            meta["joint_overshoot"] = True
        elif mode == "uncarry":
            _pad_to_aspect(action_skel, guide, out_w, out_h)
            meta["joint_overshoot"] = False
        else:
            out["errors"]["joint"] = f"unknown mode {mode!r}"
            return out

        meta["action_scail_done"] = False
        meta["scail_done"] = False
        meta["step"] = "action_joint" if mode == "carry" else "action"
        _save_meta(run_dir, meta)
        out["joint_overshoot"] = bool(meta["joint_overshoot"])
        out["guide"] = _rel_url(guide)
        return out
    except Exception as exc:
        out["errors"]["joint"] = str(exc)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe" -m pytest tests/test_stage_joint.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Update the server route**

In `server/app.py` `_handle_session_joint_overshoot`, replace the `apply` block with `mode`:

```python
            # mode: preview (render overshot preview) | carry | uncarry (guide only)
            mode = (str(form.getvalue("mode", "preview")).strip().lower() or "preview")

            def _optfloat(name):
                raw = form.getvalue(name, "")
                try:
                    return float(raw) if str(raw).strip() != "" else None
                except (TypeError, ValueError):
                    return None

            result = stage_joint_overshoot(
                run_id,
                mode=mode,
                omega=_optfloat("joint_omega"),
                zeta=_optfloat("joint_zeta"),
                soft=_optfloat("joint_soft"),
                runs_dir=RUNS_DIR,
            )
```

- [ ] **Step 6: Restart server and smoke-test all three modes**

Restart (PowerShell Start-Process with `COMFY_PYTHON`), then on an existing run with an action skeleton (`4cae50e5a6704d2fa19bfabce24a0ffe`):

```bash
RID=4cae50e5a6704d2fa19bfabce24a0ffe
B=/c/Users/AIBOX/dev/aniforge/runs/$RID
before=$(md5sum "$B/action_skel.mp4"|cut -d' ' -f1)
curl -s -X POST http://127.0.0.1:8500/session/joint-overshoot -F run_id=$RID -F mode=preview -F joint_omega=30 -F joint_zeta=0.12 | grep -o '"skeleton":[^,]*'
ls "$B/action_joint_skel.mp4" && echo preview-ok
curl -s -X POST http://127.0.0.1:8500/session/joint-overshoot -F run_id=$RID -F mode=carry | grep -o '"joint_overshoot":[a-z]*'
curl -s -X POST http://127.0.0.1:8500/session/joint-overshoot -F run_id=$RID -F mode=uncarry | grep -o '"joint_overshoot":[a-z]*'
after=$(md5sum "$B/action_skel.mp4"|cut -d' ' -f1)
[ "$before" = "$after" ] && echo "action_skel untouched ✓"
```

Expected: preview writes `action_joint_skel.mp4`; carry→`true`, uncarry→`false`; `action_skel.mp4` md5 unchanged.

- [ ] **Step 7: Commit**

```bash
git add pipeline/stages.py server/app.py tests/test_stage_joint.py
git commit -m "feat(joint): non-destructive preview/carry/uncarry modes"
```

---

### Task 2: Two-column layout for every step section

**Files:**
- Modify: `web/index.html` — inline `<style>` (add grid rules near line 170) and each `<section>` body (wrap controls + preview into columns).

**Interfaces:**
- Produces: CSS classes `.step-body`, `.controls-col`, `.preview-col`. Preview element IDs unchanged.
- Consumes: existing `.section`, `.section-head`, `.preview-box`.

- [ ] **Step 1: Add the grid CSS**

In the `<style>` block (after the `.section-head` rules, ~line 184) add:

```css
  .step-body { display: grid; grid-template-columns: minmax(340px, 400px) 1fr; gap: 1rem; align-items: start; }
  .controls-col { display: flex; flex-direction: column; gap: 0.75rem; min-width: 0; }
  .preview-col { display: flex; flex-direction: column; gap: 0.75rem; min-width: 0; }
  .preview-col .preview-box { margin: 0; }
  .preview-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
  @media (max-width: 860px) { .step-body { grid-template-columns: 1fr; } .preview-row { grid-template-columns: 1fr; } }
```

- [ ] **Step 2: Wrap one section body as the pattern (Extract, step 2)**

For `#sec-extract`, keep `.section-head` as the first child; wrap everything after it in `.step-body` with `.controls-col` (buttons/hints) and `.preview-col` (the `.preview-box`). Example final structure:

```html
  <section class="section locked" id="sec-extract" data-step="extract">
    <div class="section-head">
      <h2>2. Extract pose</h2>
      <span class="badge" id="badge-extract">locked</span>
    </div>
    <div class="step-body">
      <div class="controls-col">
        <!-- existing hint(s) + <button id="btn-extract"> ... -->
      </div>
      <div class="preview-col">
        <!-- existing <div class="preview-box" id="box-extract"> ... unchanged -->
      </div>
    </div>
  </section>
```

- [ ] **Step 3: Apply the same wrap to the remaining sections**

Repeat Step 2's `.step-body / .controls-col / .preview-col` wrap for: `#sec-idle`, `#sec-action`, `#sec-scail`, `#sec-bgremove`, `#sec-time`. Move each section's `.preview-box` element(s) verbatim into its `.preview-col`; put all inputs/sliders/buttons/`.pose-hint` into `.controls-col`. Do NOT rename any `id`. Leave `#sec-play` (combined player) and the Run-all section single-column.

- [ ] **Step 4: Verify layout in the browser**

Restart not needed (static file). In the in-app browser, load `http://127.0.0.1:8500/`, switch to Step-by-step. Run:

```javascript
const s=document.querySelector('#sec-action .step-body');
JSON.stringify({has_body:!!s, cols:getComputedStyle(s).gridTemplateColumns});
```

Expected: `has_body:true`, two column tracks at desktop width. Resize to 800px (`resize_window` preset mobile) → one column. Confirm no console errors and body has no horizontal scrollbar.

- [ ] **Step 5: Commit**

```bash
git add web/index.html
git commit -m "feat(ui): two-column step layout (controls left, preview right)"
```

---

### Task 3: Joint-overshoot step — Preview button, two windows, Carry wiring

**Files:**
- Modify: `web/index.html` — `#sec-joint` body (Preview button + two-up `.preview-row`).
- Modify: `web/app.js` — replace `doJointOvershoot`/carry handler with preview + carry/uncarry.

**Interfaces:**
- Consumes: `stage_joint_overshoot` `mode` API (Task 1); existing helpers `postForm`, `setBusy`, `setBadge`, `unlock`, `lock`, `clearErrors`, `fail`, `unlockScailSection`, `runId`, `busy`, `actionSkelReady`, `badgeAction`, `badgeScailAction`, `btnScailAction`, `secJoint`, `badgeJoint`.
- Produces: `doJointPreview()`, `doJointCarry(carry:boolean)`; DOM ids `btn-joint-preview`, `vid-joint-before`, `vid-joint-after`, reuses `joint-carry`, `joint_omega/zeta/soft`.

- [ ] **Step 1: Rebuild the joint section body (HTML)**

Replace the body of `#sec-joint` (everything after `.section-head`) with:

```html
    <div class="step-body">
      <div class="controls-col">
        <p class="pose-hint">Springy follow-through on the action skeleton. Set the spring, click <strong>Preview</strong> to render it (non-destructive), compare against the original, then <strong>Carry into SCAIL</strong> to make SCAIL use the overshot version.</p>
        <div>
          <div class="label-with-info"><label for="joint_omega">Spring frequency (omega) <span id="joint_omega_label">20</span></label></div>
          <input type="range" id="joint_omega" name="joint_omega" min="5" max="40" step="1" value="20">
        </div>
        <div>
          <div class="label-with-info"><label for="joint_zeta">Damping (zeta) <span id="joint_zeta_label">0.35</span></label></div>
          <input type="range" id="joint_zeta" name="joint_zeta" min="0.05" max="1" step="0.05" value="0.35">
        </div>
        <div>
          <div class="label-with-info"><label for="joint_soft">Softness (soft) <span id="joint_soft_label">1.0</span></label></div>
          <input type="range" id="joint_soft" name="joint_soft" min="0" max="2" step="0.1" value="1">
        </div>
        <button type="button" id="btn-joint-preview" disabled>Preview overshoot</button>
        <label id="lbl-joint-carry" style="opacity:0.5"><input type="checkbox" id="joint-carry" disabled> Carry into SCAIL</label>
      </div>
      <div class="preview-col">
        <div class="preview-row">
          <div class="preview-box show" id="box-joint-before">
            <div class="preview-label">Original</div>
            <video id="vid-joint-before" muted loop playsinline autoplay></video>
          </div>
          <div class="preview-box show" id="box-joint-after">
            <div class="preview-label">Overshot preview</div>
            <video id="vid-joint-after" muted loop playsinline autoplay></video>
          </div>
        </div>
      </div>
    </div>
```

(Removes the old single `vid-joint-skel`/`box-joint-skel`.)

- [ ] **Step 2: Update JS element refs**

In `web/app.js` where joint refs are declared, replace the `vidJointSkel` line and add:

```javascript
  const btnJointPreview = document.getElementById("btn-joint-preview");
  const vidJointBefore = document.getElementById("vid-joint-before");
  const vidJointAfter = document.getElementById("vid-joint-after");
```

Remove `const vidJointSkel = ...` (no longer exists).

- [ ] **Step 3: Replace `doJointOvershoot` with preview + carry functions**

Replace the whole `doJointOvershoot` function with:

```javascript
  function bust(url) {
    return url + (url.indexOf("?") >= 0 ? "&" : "?") + "t=" + Date.now();
  }

  /** Render the overshot preview (non-destructive) into the right window. */
  async function doJointPreview() {
    if (!runId) throw "No session.";
    if (!actionSkelReady) throw "Run action skeleton first.";
    setBadge(badgeAction, "running", "running");
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("mode", "preview");
    if (jointOmega) fd.append("joint_omega", jointOmega.value);
    if (jointZeta) fd.append("joint_zeta", jointZeta.value);
    if (jointSoft) fd.append("joint_soft", jointSoft.value);
    const data = await postForm("/session/joint-overshoot", fd);
    if (vidJointAfter && data.skeleton) vidJointAfter.src = bust(data.skeleton);
    setBadge(badgeAction, "preview", "done");
    return data;
  }

  /** Carry the previewed overshoot into SCAIL's guide, or revert to plain. */
  async function doJointCarry(carry) {
    if (!runId) throw "No session.";
    const fd = new FormData();
    fd.append("run_id", runId);
    fd.append("mode", carry ? "carry" : "uncarry");
    if (carry) {
      if (jointOmega) fd.append("joint_omega", jointOmega.value);
      if (jointZeta) fd.append("joint_zeta", jointZeta.value);
      if (jointSoft) fd.append("joint_soft", jointSoft.value);
    }
    const data = await postForm("/session/joint-overshoot", fd);
    actionScailReady = false;
    unlockScailSection();
    if (btnScailAction) {
      btnScailAction.disabled = false;
      if (badgeScailAction) setBadge(badgeScailAction, "ready", "");
    }
    return data;
  }
```

- [ ] **Step 4: Rewire enable/reset points**

Where the action step finishes (`doAction` success, currently sets carry checkbox + unlock `secJoint`), replace with:

```javascript
    if (chkJointCarry) chkJointCarry.checked = false;
    if (secJoint && badgeJoint) unlock(secJoint, badgeJoint, "ready");
    if (btnJointPreview) btnJointPreview.disabled = false;
    setJointCarryEnabled(false); // carry stays disabled until first preview
    if (vidJointBefore) vidJointBefore.src = bust("/runs/" + runId + "/action_skel.mp4");
    if (vidJointAfter) vidJointAfter.removeAttribute("src");
```

In the reset/lock block (where `chkJointCarry.checked=false; lock(secJoint...)`), also add:

```javascript
    if (btnJointPreview) btnJointPreview.disabled = true;
    if (vidJointBefore) vidJointBefore.removeAttribute("src");
    if (vidJointAfter) vidJointAfter.removeAttribute("src");
```

In the `setBusy` body, replace the `setJointCarryEnabled(...)` line so both the button and (only-after-preview) carry follow busy:

```javascript
    if (btnJointPreview) btnJointPreview.disabled = on || !runId || !actionSkelReady;
    // carry enable is managed by preview success; just re-disable while busy
    if (chkJointCarry && on) chkJointCarry.disabled = true;
```

- [ ] **Step 5: Replace the carry/slider handlers**

Replace the existing `runJointCarry` + `chkJointCarry` + `wireJointSlider` block with:

```javascript
  async function runJointPreview() {
    clearErrors();
    setBusy(true, "Rendering overshoot preview…");
    try {
      await doJointPreview();
      setJointCarryEnabled(true); // enable carry once a preview exists
      if (chkJointCarry && chkJointCarry.checked) await doJointCarry(true); // keep guide in sync
      statusEl.textContent = "Overshoot preview ready. Check Carry into SCAIL to use it.";
    } catch (e) {
      fail(e); setBadge(badgeAction, "error", "");
      statusEl.textContent = "Overshoot preview failed.";
    } finally { setBusy(false); }
  }
  if (btnJointPreview) {
    btnJointPreview.addEventListener("click", () => {
      if (!runId || busy) return;
      runJointPreview();
    });
  }
  if (chkJointCarry) {
    chkJointCarry.addEventListener("change", async () => {
      if (!runId || busy) return;
      const carry = chkJointCarry.checked;
      clearErrors();
      setBusy(true, carry ? "Carrying overshoot into SCAIL…" : "Reverting to plain skeleton…");
      try {
        await doJointCarry(carry);
        statusEl.textContent = carry
          ? "Overshoot carried into SCAIL. Re-run SCAIL2 to update character."
          : "Reverted to plain action skeleton. Re-run SCAIL2 to update character.";
      } catch (e) {
        chkJointCarry.checked = !carry;
        fail(e); setBadge(badgeAction, "error", "");
        statusEl.textContent = "Carry toggle failed.";
      } finally { setBusy(false); }
    });
  }
  function wireJointSlider(el, label, fmt) {
    if (!el) return;
    const upd = () => { if (label) label.textContent = fmt(el.value); };
    el.addEventListener("input", upd);
    el.addEventListener("change", () => {
      upd();
      // Re-preview on release once the section is usable; runJointPreview re-carries if checked.
      if (btnJointPreview && !btnJointPreview.disabled && !busy && runId) runJointPreview();
    });
    upd();
  }
  wireJointSlider(jointOmega, jointOmegaLabel, (v) => String(Math.round(Number(v))));
  wireJointSlider(jointZeta, jointZetaLabel, (v) => Number(v).toFixed(2));
  wireJointSlider(jointSoft, jointSoftLabel, (v) => Number(v).toFixed(1));
```

- [ ] **Step 6: Update the Run-all call site**

In the Run-all handler, the `if (jointChecked()) { ... await doJointOvershoot(); }` call must become:

```javascript
        if (jointChecked()) {
          statusEl.textContent = "Run all: joint overshoot on skeleton…";
          await doJointCarry(true); // self-springs preview on the server
        }
```

Search for the other `await doJointOvershoot();` (the create-session flow) and replace it the same way. Then confirm no `doJointOvershoot` references remain.

- [ ] **Step 7: Verify in the browser end-to-end**

Restart server. Load a session that already has an action skeleton (or run create→extract→action). In the in-app browser at step 5:

```javascript
JSON.stringify({
  preview_disabled: document.getElementById('btn-joint-preview').disabled,
  carry_disabled: document.getElementById('joint-carry').disabled,
  before_src: !!document.getElementById('vid-joint-before').getAttribute('src'),
  after_src: !!document.getElementById('vid-joint-after').getAttribute('src'),
  no_old_ref: typeof doJointOvershoot === 'undefined'
});
```

Expected before any preview: `preview_disabled:false`, `carry_disabled:true`, `before_src:true`, `after_src:false`. Click Preview → `after_src:true`, carry enabled. Check Carry → status confirms; server `meta.joint_overshoot=true`. Move a slider → after window updates. No console errors.

- [ ] **Step 8: Commit**

```bash
git add web/index.html web/app.js
git commit -m "feat(ui): joint overshoot preview + before/after compare + carry"
```

---

## Notes for the implementer

- The old `doJointOvershoot(apply)` and its `apply=1|0` API are fully replaced; grep for `doJointOvershoot`, `apply=`, `vid-joint-skel`, `step-overshoot-joint`, `setStepJointEnabled` and remove stragglers.
- There are no JS unit tests in this repo; frontend verification is the browser check in each task. Backend logic is covered by `tests/test_stage_joint.py`.
- Running full renders needs the model warm; the mode logic itself is exercised by the monkeypatched unit tests without a GPU.
