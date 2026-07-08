# Motion Portrait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone tool that turns one character image + text prompts into a live2d-style animated portrait — a looping idle plus a click-triggered action — by orchestrating the shared ComfyUI pipeline (Kimodo → skeleton spring → scail2 → optional time spring) and returning two mp4s.

**Architecture:** Three units in one repo: a single-page static frontend (upload + click player), a minimal Python `http.server` backend exposing `POST /generate`, and a `pipeline/` package that submits graphs to ComfyUI at 8188 and post-processes results. The backend is an independent process; the only shared dependency is ComfyUI.

**Tech Stack:** Python 3 (stdlib `http.server`, `urllib`), numpy, opencv-python (`cv2`), matplotlib (curve output, optional), Pillow; ffmpeg CLI for padding/encoding; ComfyUI at `http://127.0.0.1:8188` with `ComfyUI-Kimodo` and SCAIL2 nodes.

## Global Constraints

- **All product text (code, comments, string literals, UI, committed data) is English only — no Chinese.**
- Independent from camera-lab: separate repo/process/port; only shared dependency is ComfyUI at `http://127.0.0.1:8188`.
- Reuse pipeline scripts by **vendoring copies** into this repo (`pipeline/`); do not import across repos.
- Backend default port **8500**. ComfyUI input dir: `C:/Users/AIBOX/dev/ComfyUI-scail/input`. ComfyUI output fetched via `/view`.
- Overshoot options apply to the **action** clip only; joint overshoot in skeleton space, time overshoot on the rendered video; both may be selected (they stack).
- Chosen time-spring params: `B0.42 D4.2 F2.4 T1.15`. Default joint-spring: `omega 20, zeta 0.35, soft 1.0`.

---

### Task 1: Kimodo-SOMA operational (hard gating spike)

Research/enablement task — get Kimodo text-to-motion producing an NPZ on ComfyUI-scail:8188. **Kimodo is a hard requirement for this MVP: there is NO HY-Motion fallback.** If the dependency chain resists, keep resolving it (or escalate to the user) — do not substitute a different motion source. No product code; ends when Kimodo-SOMA is confirmed producing an NPZ.

**Files:**
- Create: `docs/motion-source.md` (records which source is used + how it was made to work)

**Context:**
- `ComfyUI-Kimodo` (jtydhr88) is installed and its 9 nodes load. `Kimodo-SOMA-RP-v1` (public) + Llama-3-8B-Instruct encoder (gated, access granted, ~15GB cached at `F:/AIModelArchive/CDriveOffload_20260514/caches/huggingface/hub`).
- Blocker: `Kimodo_LoadModel` fails — `LLM2VecEncoder` → `gptqmodel 7.1.0` raises `ImportError('gptqmodel requires optimum version 1.24.0 or higher')` AND `gptqmodel` itself fails to import with `ModuleNotFoundError: No module named 'pcre'`. `optimum 2.2.0` is installed but `optimum.__version__` is absent (may break gptqmodel's version probe).
- Comfy python: `C:/Users/AIBOX/anaconda3/envs/comfy-scail/python.exe`.

- [ ] **Step 1: Reproduce and read the encoder loader**

Run the SOMA text-to-motion graph (LoadModel→TextEncode→Sampler→PostProcess→SaveNPZ) via `POST http://127.0.0.1:8188/prompt` and confirm the `gptqmodel`/`pcre` error. Read `ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo/kimodo/model/llm2vec/llm2vec_wrapper.py` to see whether the GPTQ path is mandatory or selectable (look for a non-quantized / full-precision Llama-3 option).

- [ ] **Step 2: Resolve the dependency chain (in the comfy-scail env only)**

Try, in order, stopping when generation succeeds:
1. Install the missing `pcre` dependency for gptqmodel (`pip install python-pcre` or the package gptqmodel expects — confirm the exact import name it needs).
2. If the optimum version probe is the failure, pin optimum to the range gptqmodel 7.1.0 expects (check gptqmodel's requirement; a 1.24.x–1.x pin may be needed instead of 2.2.0).
3. If GPTQ is optional, switch the encoder config to the non-quantized Llama-3 path (no gptqmodel).
Before each install, run `pip install --dry-run <pkg>` and confirm it does NOT change `torch` or `transformers` (protect the HY-Motion stack). Restart ComfyUI-scail after installs so it re-imports.

- [ ] **Step 3: Confirm an NPZ is produced**

Re-run the graph; on success, load the saved `.npz` and print `keys()` and `keypoints3d`/joint-array shape. Record the exact NPZ key holding joint positions `[T, J, 3]` and the joint count (expect 30 for SOMA).

- [ ] **Step 4: Document the working Kimodo setup**

Write `docs/motion-source.md`: "Kimodo-SOMA-RP-v1, 30-joint, NPZ key `<name>`, encoder deps resolved by: `<exact steps>`." Kimodo is mandatory — do NOT substitute HY-Motion. If deps stay blocked after Step 2's options are exhausted, stop and escalate to the user rather than shipping a different source.

- [ ] **Step 5: Commit**

```bash
git add docs/motion-source.md
git commit -m "docs: confirm Kimodo-SOMA motion source"
```

---

### Task 2: Repo scaffolding + ComfyUI client

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pipeline/__init__.py`, `pipeline/comfy.py`
- Test: `tests/test_comfy.py`

**Interfaces:**
- Produces:
  - `ComfyClient(base_url="http://127.0.0.1:8188", opener=urllib.request.urlopen)`
  - `.submit(graph: dict, client_id: str) -> str` (prompt_id)
  - `.wait(prompt_id: str, timeout=1500) -> dict` (history entry)
  - `.fetch_output(item: dict, dest: Path) -> Path` (GET `/view`, item has `filename/subfolder/type`)
  - `.stage_input(src: Path, name: str, input_dir: Path) -> str` (copy into ComfyUI input dir, return name)
  - `.object_info() -> dict`

- [ ] **Step 1: Write requirements.txt and .gitignore**

`requirements.txt`:
```
numpy
opencv-python
Pillow
matplotlib
```
`.gitignore`:
```
__pycache__/
*.pyc
runs/
```

- [ ] **Step 2: Write the failing test** (`tests/test_comfy.py`)

```python
import json, io
from pipeline.comfy import ComfyClient

class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def make_opener(script):
    calls = []
    def opener(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        return FakeResp(json.dumps(script[len(calls)-1]).encode())
    opener.calls = calls
    return opener

def test_submit_returns_prompt_id():
    opener = make_opener([{"prompt_id": "abc"}])
    c = ComfyClient(opener=opener)
    assert c.submit({"1": {}}, "cid") == "abc"

def test_wait_returns_history_entry():
    opener = make_opener([{"pid": {"status": {"status_str": "success"}, "outputs": {}}}])
    c = ComfyClient(opener=opener)
    entry = c.wait("pid", timeout=5)
    assert entry["status"]["status_str"] == "success"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_comfy.py -v`
Expected: FAIL (ImportError / module not found).

- [ ] **Step 4: Implement `pipeline/comfy.py`**

```python
"""Minimal ComfyUI HTTP client: submit graphs, poll history, fetch outputs."""
from __future__ import annotations
import json, time, shutil, urllib.request, urllib.parse
from pathlib import Path


class ComfyClient:
    def __init__(self, base_url="http://127.0.0.1:8188", opener=urllib.request.urlopen):
        self.base = base_url.rstrip("/")
        self.opener = opener

    def _get(self, path):
        with self.opener(self.base + path, timeout=30) as r:
            return json.load(r)

    def _post(self, path, payload):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.opener(req, timeout=30) as r:
            return json.load(r)

    def object_info(self):
        return self._get("/object_info")

    def submit(self, graph, client_id):
        return self._post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]

    def wait(self, prompt_id, timeout=1500, interval=3):
        t0 = time.time()
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                return hist[prompt_id]
            if time.time() - t0 > timeout:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout}s")
            time.sleep(interval)

    def fetch_output(self, item, dest: Path):
        q = urllib.parse.urlencode({"filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.opener(f"{self.base}/view?{q}", timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest

    @staticmethod
    def stage_input(src: Path, name: str, input_dir: Path) -> str:
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, input_dir / name)
        return name


def first_output(entry, keys=("videos", "gifs", "images")):
    """Return the first output item dict from a history entry, or None."""
    for out in entry.get("outputs", {}).values():
        for k in keys:
            for item in out.get(k, []):
                return item
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_comfy.py -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore pipeline/ tests/test_comfy.py
git commit -m "feat: ComfyUI client + repo scaffolding"
```

---

### Task 3: Vendor + adapt skeleton_spring for the chosen skeleton

**Files:**
- Create: `pipeline/skeleton_spring.py` (vendored from `camera-lab/tasks/live2d/skeletons/skeleton_spring.py`, then adapted), `pipeline/skeletons.py` (bone/joint tables)
- Test: `tests/test_skeleton_spring.py`

**Interfaces:**
- Consumes: the Kimodo NPZ from Task 1 — the joint array is under key **`posed_joints`** with shape **`[T, 77, 3]`** (SOMASkeleton77; 30 fps). Load `posed_joints`, not `keypoints3d`.
- Produces:
  - `spring_follow(target, fps, omega, zeta, soft_scale) -> ndarray` (adaptive-substep damped follower; stable at large omega)
  - `frame_fixed(all_kpts) -> (cx, cy, scale)`
  - `render(kpts, path, fps, size=512, fixed=True)` — writes an mp4 skeleton video
  - `BONES`/`COLORS`/`SOFT`/`N_JOINTS` in `pipeline/skeletons.py` for the SOMASkeleton77 **body chain**

- [ ] **Step 1: Copy the existing script**

Copy `camera-lab/tasks/live2d/skeletons/skeleton_spring.py` → `pipeline/skeleton_spring.py`. It already has the adaptive-substep integrator fix (`sub = max(8, int(ceil(w.max()*dt/0.2)))`) and `frame_fixed`. Move the `BONES`/`COLORS`/`SOFT` tables into `pipeline/skeletons.py`.

- [ ] **Step 2: Write the failing test** (`tests/test_skeleton_spring.py`)

```python
import numpy as np
from pipeline.skeleton_spring import spring_follow
from pipeline import skeletons

def test_integrator_stable_at_high_omega():
    k = np.random.RandomState(0).randn(40, skeletons.N_JOINTS, 3).astype(np.float32) * 0.1
    out = spring_follow(k, 30, 200.0, 1.0, 0.0)
    assert np.isfinite(out).all()
    # near-rigid: high omega tracks the target closely
    assert np.linalg.norm(out - k, axis=2).mean() < 0.05

def test_bones_indices_in_range():
    for a, b in skeletons.BONES:
        assert 0 <= a < skeletons.N_JOINTS
        assert 0 <= b < skeletons.N_JOINTS
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_skeleton_spring.py -v`
Expected: FAIL (module/attr not found until `skeletons.py` + import wiring exist).

- [ ] **Step 4: Build `pipeline/skeletons.py` from SOMASkeleton77**

The NPZ `posed_joints` is `[T, 77, 3]` in `SOMASkeleton77` order. Read the joint order and parent map from `ComfyUI-Kimodo/kimodo/kimodo/skeleton/definitions.py` `SOMASkeleton77.bone_order_names_with_parents` (print the list at implementation time — it is long). Select a **body subset** for the scail guide: the torso/limb chain (Hips, Spine*, Chest, Neck*, Head, shoulders, arms, forearms, hands, legs, shins, feet, toes) — drop the individual finger and eye/jaw joints (they clutter the guide and scail does not need them). Define `BODY_JOINTS` = the ordered list of 77-index positions kept; `N_JOINTS = len(BODY_JOINTS)`; remap `BONES` to indices within `BODY_JOINTS`. Assign `SOFT` per kept joint: core (Hips, Spine*, Chest, Neck*, hips/leg roots) 0.0; mid (shoulders, arms, shins) 0.35; soft (forearms, feet) 0.7; softest (hands, toes) 1.0. Give each bone a color.

Add a loader `load_posed_joints(npz_path) -> ndarray[T, N_JOINTS, 3]` that reads `posed_joints` and gathers `BODY_JOINTS`. Update `skeleton_spring.py` to import `BONES, COLORS, SOFT, N_JOINTS` from `pipeline.skeletons` and operate on the gathered body array (no fixed 22/30 slice).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_skeleton_spring.py -v`
Expected: PASS.

- [ ] **Step 6: Sanity-render from a real NPZ**

Load the NPZ from Task 1, `render(kpts, "runs/skel_check.mp4", fps=30, fixed=True)`, and open it — confirm a clean front-facing skeleton (no collapse). This guards the SOMA bone mapping.

- [ ] **Step 7: Commit**

```bash
git add pipeline/skeleton_spring.py pipeline/skeletons.py tests/test_skeleton_spring.py
git commit -m "feat: vendor + adapt skeleton_spring for the chosen skeleton"
```

---

### Task 4: Vendor spring_time_remap (time overshoot)

**Files:**
- Create: `pipeline/spring_time_remap.py` (vendored from `camera-lab/scripts/spring_time_remap.py`)
- Test: `tests/test_time_remap.py`

**Interfaces:**
- Produces: `remap_indices(n_frames, fps, b, d, f, t) -> list[float]`, `sample_frame(frames, src_float, sampling)`, and a `time_remap_file(inp: Path, out: Path, b, d, f, t, sampling="blend")` helper that reads/writes an mp4.

- [ ] **Step 1: Copy the script** → `pipeline/spring_time_remap.py`. Keep `remap_indices`, `sample_frame`, `read_video`, `write_video`, and add a thin `time_remap_file(inp, out, b=0.42, d=4.2, f=2.4, t=1.15, sampling="blend")` wrapper (read → remap → write, no compare/curve).

- [ ] **Step 2: Write the failing test**

```python
from pipeline.spring_time_remap import remap_indices

def test_chosen_params_small_overshoot():
    idx = remap_indices(105, 24, 0.42, 4.2, 2.4, 1.15)
    back = max((idx[i-1]-idx[i]) for i in range(1, len(idx)))
    assert 3.0 < back < 9.0            # small overshoot, not a full replay
    assert idx[0] == 0 and idx[-1] <= 104
```

- [ ] **Step 3: Run to verify fail**, then rely on the copied implementation. Run: `python -m pytest tests/test_time_remap.py -v` → PASS after copy.

- [ ] **Step 4: Commit**

```bash
git add pipeline/spring_time_remap.py tests/test_time_remap.py
git commit -m "feat: vendor spring_time_remap (time overshoot)"
```

---

### Task 5: Kimodo (motion) submitter

**Files:**
- Create: `pipeline/kimodo.py`
- Test: `tests/test_kimodo.py`

**Interfaces:**
- Consumes: `ComfyClient` (Task 2).
- Produces: `build_kimodo_graph(prompt, duration, seed, model, steps) -> dict`, `generate_motion(client, prompt, duration, seed, out_npz: Path, ...) -> Path` (submits, waits, resolves the SaveNPZ `file_path`/`/view`, writes NPZ to `out_npz`).

- [ ] **Step 1: Write the failing test** (graph shape only — pure, no network)

```python
from pipeline.kimodo import build_kimodo_graph

def test_graph_wires_text_and_sampler():
    g = build_kimodo_graph("A person waves.", duration=3.0, seed=42,
                            model="Kimodo-SOMA-RP-v1", steps=50)
    assert g["1"]["class_type"] == "Kimodo_LoadModel"
    assert g["1"]["inputs"]["model"] == "Kimodo-SOMA-RP-v1"
    assert g["2"]["inputs"]["prompt"] == "A person waves."
    assert g["2"]["inputs"]["model"] == ["1", 0]
    assert g["3"]["inputs"]["conditioning"] == ["2", 0]
    assert g["3"]["inputs"]["duration"] == 3.0
    assert g["5"]["class_type"] == "Kimodo_SaveNPZ"
```


- [ ] **Step 2: Run to verify fail.** Run: `python -m pytest tests/test_kimodo.py -v` → FAIL.

- [ ] **Step 3: Implement `build_kimodo_graph` + `generate_motion`**

```python
"""Submit a Kimodo text-to-motion graph and retrieve the NPZ."""
from __future__ import annotations
import uuid
from pathlib import Path
from .comfy import ComfyClient


def build_kimodo_graph(prompt, duration=3.0, seed=42,
                       model="Kimodo-SOMA-RP-v1", steps=50, prefix="mp_motion"):
    return {
        "1": {"class_type": "Kimodo_LoadModel", "inputs": {"model": model}},
        "2": {"class_type": "Kimodo_TextEncode",
              "inputs": {"model": ["1", 0], "prompt": prompt}},
        "3": {"class_type": "Kimodo_Sampler",
              "inputs": {"model": ["1", 0], "conditioning": ["2", 0],
                         "duration": float(duration), "seed": int(seed),
                         "num_samples": 1, "diffusion_steps": int(steps)}},
        "4": {"class_type": "Kimodo_PostProcess", "inputs": {"motion": ["3", 0]}},
        "5": {"class_type": "Kimodo_SaveNPZ",
              "inputs": {"motion": ["4", 0], "filename_prefix": prefix}},
    }


def generate_motion(client: ComfyClient, prompt, out_npz: Path, *,
                    duration=3.0, seed=42, model="Kimodo-SOMA-RP-v1", steps=50,
                    comfy_output=Path("C:/Users/AIBOX/dev/ComfyUI-scail/output")):
    graph = build_kimodo_graph(prompt, duration, seed, model, steps)
    pid = client.submit(graph, f"mp-kim-{uuid.uuid4().hex[:6]}")
    entry = client.wait(pid)
    if entry["status"]["status_str"] != "success":
        raise RuntimeError(f"Kimodo failed: {entry['status'].get('messages')}")
    # SaveNPZ writes under comfy output; resolve the newest matching file
    node_out = entry["outputs"].get("5", {})
    rel = (node_out.get("file_path") or node_out.get("text") or [None])
    rel = rel[0] if isinstance(rel, list) else rel
    src = (comfy_output / rel) if rel else _newest(comfy_output, "mp_motion")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_npz.write_bytes(Path(src).read_bytes())
    return out_npz


def _newest(root: Path, stem: str) -> Path:
    files = sorted(root.rglob(f"{stem}*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {stem}*.npz under {root}")
    return files[-1]
```

Note: confirm the exact `Kimodo_SaveNPZ` output key/path shape from the Task 1 run and adjust `rel` resolution if needed.

- [ ] **Step 4: Run tests → PASS.** Run: `python -m pytest tests/test_kimodo.py -v`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/kimodo.py tests/test_kimodo.py
git commit -m "feat: Kimodo motion submitter"
```

---

### Task 6: scail2 character-drive submitter

**Files:**
- Create: `pipeline/scail.py`, `pipeline/assets/scail2_video.api.json` (copied template from `camera-lab/workflows/app/scail2_video.api.json`)
- Test: `tests/test_scail.py`

**Interfaces:**
- Consumes: `ComfyClient`, a padded skeleton guide mp4, a staged reference image name.
- Produces: `build_scail_graph(template, guide_name, ref_name, width, height, length, pose_strength, seed, steps, prefix, positive) -> dict`, `drive_character(client, guide_mp4, ref_image, out_mp4, ...) -> Path`.

- [ ] **Step 1: Copy the scail2 template** to `pipeline/assets/scail2_video.api.json`.

- [ ] **Step 2: Write the failing test**

```python
import json
from pathlib import Path
from pipeline.scail import build_scail_graph

TPL = json.loads(Path("pipeline/assets/scail2_video.api.json").read_text(encoding="utf-8"))

def test_scail_graph_sets_guide_ref_and_size():
    g = build_scail_graph(TPL, "guide.mp4", "ref.png", 480, 832, 105, 0.9, 42, 6, "mp_body", "a person")
    assert g["9"]["inputs"]["image"] == "ref.png"
    assert g["11"]["inputs"]["file"] == "guide.mp4"
    assert g["13"]["inputs"]["width"] == 480 and g["13"]["inputs"]["length"] == 105
    assert g["14"]["inputs"]["seed"] == 42
    assert g["17"]["inputs"]["filename_prefix"] == "mp_body"
```

- [ ] **Step 3: Run to verify fail**, then implement `build_scail_graph` (deep-copy template, set nodes 5/9/11/13/14/17 as in camera-lab `build_scail_api`) and `drive_character` (stage ref + guide into ComfyUI input, submit, wait, `fetch_output(first_output(entry), out_mp4)`). Fixed-frame guide is produced by Task 3; no SAM3 mask in MVP.

- [ ] **Step 4: Run tests → PASS.** Commit:

```bash
git add pipeline/scail.py pipeline/assets/scail2_video.api.json tests/test_scail.py
git commit -m "feat: scail2 character-drive submitter"
```

---

### Task 7: Generate orchestration (idle + action → 2 mp4s)

**Files:**
- Create: `pipeline/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: all pipeline modules.
- Produces:
  - `DEFAULT_IDLE_PROMPT` (str) and `sanitize_action(prompt: str) -> str` (drop "turning"/"turn" words).
  - `plan_steps(overshoot: set[str]) -> dict` — pure: which post-steps run. `{"joint": "joint" in overshoot, "time": "time" in overshoot}`.
  - `generate(image: Path, action_prompt: str, idle_prompt: str|None, overshoot: set[str], run_dir: Path, client, *, motion_model, comfy_input, comfy_output) -> dict` returning `{"idle": Path, "action": Path, "errors": {...}}`.

- [ ] **Step 1: Write the failing test** (pure composition logic)

```python
from pipeline.generate import plan_steps, sanitize_action, DEFAULT_IDLE_PROMPT

def test_plan_steps_selects_overshoot():
    assert plan_steps(set()) == {"joint": False, "time": False}
    assert plan_steps({"joint"}) == {"joint": True, "time": False}
    assert plan_steps({"joint", "time"}) == {"joint": True, "time": True}

def test_sanitize_drops_turning():
    assert "turn" not in sanitize_action("she turns and waves").lower()

def test_default_idle_is_in_place():
    assert "in place" in DEFAULT_IDLE_PROMPT.lower()
```

- [ ] **Step 2: Run to verify fail.** Run: `python -m pytest tests/test_generate.py -v` → FAIL.

- [ ] **Step 3: Implement `pipeline/generate.py`**

Idle branch: `generate_motion(idle_prompt) → skeleton_spring.render(fixed=True, NO overshoot) → pad to 512x888 (ffmpeg) → drive_character → idle.mp4`.
Action branch: `generate_motion(action_prompt) → skeleton_spring.render(fixed=True, + joint overshoot if plan["joint"]) → pad → drive_character → (+ time_remap_file if plan["time"]) → action.mp4`.
Wrap each branch in try/except, collect into `errors`, return whatever succeeded (per-clip isolation). Padding helper shells `ffmpeg -vf "pad=512:888:0:(888-512)/2:color=black"`. `length = ((n_frames-1)//4)*4+1` (align_4k1).

```python
DEFAULT_IDLE_PROMPT = ("A person stands in place in a relaxed idle stance, breathing calmly, "
                       "swaying gently from side to side, with small subtle movements of the head and arms.")

def plan_steps(overshoot):
    return {"joint": "joint" in overshoot, "time": "time" in overshoot}

def sanitize_action(prompt):
    import re
    return re.sub(r"\b(turning|turns|turn)\b", "", prompt, flags=re.I).strip()
```

- [ ] **Step 4: Run pure tests → PASS.** Run: `python -m pytest tests/test_generate.py -v`.

- [ ] **Step 5: End-to-end smoke (host-manual)**

With ComfyUI up, run `generate(test_image, "raises the right hand and waves", None, {"joint"}, Path("runs/smoke"), client, ...)`; assert two mp4s exist with >0 frames (probe via cv2). Not in CI.

- [ ] **Step 6: Commit**

```bash
git add pipeline/generate.py tests/test_generate.py
git commit -m "feat: generate orchestration (idle + action -> 2 mp4s)"
```

---

### Task 8: Backend server (`POST /generate` + static serving)

**Files:**
- Create: `server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Produces: an `http.server`-based app: `GET /` → `web/index.html`; `GET /web/*` and `GET /runs/*` → static files; `POST /generate` (multipart: image + fields) → `{"idle": "/runs/<id>/idle.mp4", "action": "/runs/<id>/action.mp4", "errors": {...}}`. Preflight `/object_info` for required node classes; 503 with a clear message if missing.
- Pure helper (unit-testable): `parse_generate_form(fields) -> dict` and `required_nodes_present(object_info) -> bool`.

- [ ] **Step 1: Write the failing test** (pure helpers only)

```python
from server.app import parse_generate_form, required_nodes_present

def test_parse_form_overshoot_multi():
    d = parse_generate_form({"action_prompt": "wave", "idle_prompt": "",
                             "overshoot": ["joint", "time"]})
    assert d["action_prompt"] == "wave"
    assert d["idle_prompt"] is None
    assert d["overshoot"] == {"joint", "time"}

def test_required_nodes_check():
    assert required_nodes_present({"Kimodo_Sampler": {}, "WanSCAILToVideo": {}})
    assert not required_nodes_present({"Kimodo_Sampler": {}})
```

- [ ] **Step 2: Run to verify fail**, then implement `server/app.py` (`BaseHTTPRequestHandler`; multipart parse via `cgi`/`email`; call `pipeline.generate.generate`; serve `web/` and `runs/`). Default port 8500, `--port` arg.

- [ ] **Step 3: Run tests → PASS.** Commit:

```bash
git add server/app.py tests/test_server.py
git commit -m "feat: backend server with POST /generate"
```

---

### Task 9: Frontend (single page + click player)

**Files:**
- Create: `web/index.html`, `web/app.js`

**Interfaces:**
- Consumes: `POST /generate`, response `{idle, action, errors}`.

- [ ] **Step 1: Write `web/index.html`** — a form (image drop/upload, action prompt input, optional idle prompt input, two overshoot checkboxes `joint`/`time`, Generate button) and a preview area with two `<video>` elements (idle `loop muted autoplay`, action hidden). Inline minimal CSS. English only.

- [ ] **Step 2: Write `web/app.js`** — on submit, POST multipart to `/generate`, show a spinner; on success set `idleVideo.src` and `actionVideo.src`. Player: idle plays looped; on `click` of the preview, pause idle, show+play action from 0; on action `ended`, hide action and resume idle. Show `errors` if any clip failed.

- [ ] **Step 3: Manual verify** — start the server, open `http://127.0.0.1:8500`, upload a test image, generate, confirm idle loops and click plays the action then returns.

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/app.js
git commit -m "feat: single-page UI with click-triggered player"
```

---

### Task 10: README + end-to-end verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** — prerequisites (ComfyUI-scail on 8188 with Kimodo/SCAIL nodes + models; ffmpeg on PATH), install (`pip install -r requirements.txt`), run (`python server/app.py`), open `http://127.0.0.1:8500`, usage (upload image, action prompt, overshoot, generate). English only.

- [ ] **Step 2: Full manual run** — from a clean checkout, follow the README, generate for one test image, confirm two clips + click player work. Fix anything that blocks a first-time run.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and usage"
```

## Self-Review

**Spec coverage:** UI (Task 9), backend `POST /generate` port 8500 (Task 8), 2-mp4 idle+action (Task 7), overshoot action-only + stack (Tasks 3/4/7), Kimodo-SOMA, no fallback (Task 1), SOMA 30-joint renderer (Task 3), vendored scripts (Tasks 3/4), preflight + per-clip isolation errors (Tasks 7/8), fixed-frame turning guard (Task 3/7), English-only (all), tests (each task). Prerequisites folded into Task 1/3. Covered.

**Placeholder scan:** Task 1 is an explicit spike (research task) with success criteria and a fallback, not a placeholder; all code tasks carry concrete code. No TBD/TODO left.

**Type consistency:** `generate_motion(client, prompt, out_npz, ...)`, `render(kpts, path, fps, size, fixed)`, `build_scail_graph(...)`, `plan_steps(overshoot)->{"joint","time"}`, `first_output(entry)` are used consistently across tasks.
