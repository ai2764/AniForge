# Native Background Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move background removal into AniForge as a native subprocess-backed implementation while preserving the existing `stage_bgremove()` output contract.

**Architecture:** `pipeline.bgremove` remains the facade used by `pipeline.stages`. By default it launches `pipeline/bgremove_native/worker.py` in a subprocess so torch and matting models never load into the server process; `ANIFORGE_BGREMOVE_BACKEND=external` keeps the current sibling-repo backend available for comparison.

**Tech Stack:** Python 3.10-3.12, pytest, subprocess workers, ffmpeg, OpenCV, Pillow, numpy, torch/torchvision, transformers, kornia, huggingface_hub.

## Global Constraints

- Keep current `stage_bgremove()` response fields and output names unchanged.
- Default backend is native.
- External backend is used only when `ANIFORGE_BGREMOVE_BACKEND=external`.
- Do not import torch from `server/app.py`, `pipeline/stages.py`, or lightweight unit tests.
- Preserve subprocess isolation for matting work.
- Alpha MOV conversion remains a soft error when preview MP4 and WebM alpha succeeded.
- Do not redesign the UI.
- Do not delete the external `videoBGremoval` repo.

---

## File Structure

- Modify `pipeline/bgremove.py`: backend routing, native worker invocation, shared result parsing, external fallback behind env var.
- Create `pipeline/bgremove_native/__init__.py`: package marker and public constants.
- Create `pipeline/bgremove_native/worker.py`: subprocess CLI entry point compatible with current `RESULT:*` output lines.
- Create `pipeline/bgremove_native/runner.py`: pure-Python orchestration callable from tests and worker.
- Create `pipeline/bgremove_native/engines.py`: lazy imports and model wrappers for RMBG/RVM.
- Create `pipeline/bgremove_native/decoder.py`: video frame decoding and metadata.
- Create `pipeline/bgremove_native/encoder.py`: preview MP4, VP9 alpha WebM, optional audio mux helpers.
- Create `pipeline/bgremove_native/compositor.py`: alpha compositing helpers.
- Create `pipeline/bgremove_native/ffmpeg.py`: ffmpeg path resolution.
- Create `tests/test_bgremove_backend.py`: lightweight backend routing and command construction tests.
- Create `tests/test_bgremove_native_runner.py`: lightweight runner/result tests with faked engine/decoder/encoder.
- Modify `.env.example`: document native default and legacy external backend.
- Modify `README.md`: update background-removal setup notes.

---

### Task 1: Backend Routing And Result Parsing

**Files:**
- Modify: `pipeline/bgremove.py`
- Create: `tests/test_bgremove_backend.py`

**Interfaces:**
- Consumes: existing `run_bgremove(input_video, output_dir, *, model, formats, ...) -> dict`.
- Produces:
  - `select_bgremove_backend() -> str`
  - `_parse_worker_results(stdout: str, output_dir: Path, input_video: Path) -> dict`
  - `_run_worker_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]`

- [ ] **Step 1: Write failing backend selection tests**

Add `tests/test_bgremove_backend.py`:

```python
from pathlib import Path

import pytest

import pipeline.bgremove as bg


def test_select_bgremove_backend_defaults_to_native(monkeypatch):
    monkeypatch.delenv("ANIFORGE_BGREMOVE_BACKEND", raising=False)
    assert bg.select_bgremove_backend() == "native"


def test_select_bgremove_backend_accepts_external(monkeypatch):
    monkeypatch.setenv("ANIFORGE_BGREMOVE_BACKEND", "external")
    assert bg.select_bgremove_backend() == "external"


def test_select_bgremove_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ANIFORGE_BGREMOVE_BACKEND", "bogus")
    with pytest.raises(ValueError, match="ANIFORGE_BGREMOVE_BACKEND"):
        bg.select_bgremove_backend()


def test_parse_worker_results_reads_result_lines(tmp_path):
    input_video = tmp_path / "action.mp4"
    input_video.write_bytes(b"fake")
    preview = tmp_path / "preview.mp4"
    webm = tmp_path / "action.webm"
    preview.write_bytes(b"preview")
    webm.write_bytes(b"webm")

    parsed = bg._parse_worker_results(
        f"RESULT:preview:{preview}\nRESULT:output:{webm}\n",
        tmp_path,
        input_video,
    )

    assert parsed["preview"] == preview
    assert parsed["outputs"] == [webm]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py -v
```

Expected: failures because `select_bgremove_backend` and `_parse_worker_results` do not exist.

- [ ] **Step 3: Implement backend selection and result parser**

In `pipeline/bgremove.py`, add:

```python
def select_bgremove_backend() -> str:
    backend = os.environ.get("ANIFORGE_BGREMOVE_BACKEND", "native").strip().lower()
    if backend not in ("native", "external"):
        raise ValueError(
            "ANIFORGE_BGREMOVE_BACKEND must be 'native' or 'external', "
            f"got {backend!r}"
        )
    return backend


def _parse_worker_results(stdout: str, output_dir: Path, input_video: Path) -> dict:
    preview = None
    outputs: list[Path] = []
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT:preview:"):
            preview = Path(line.split(":", 2)[2].strip())
        elif line.startswith("RESULT:output:"):
            outputs.append(Path(line.split(":", 2)[2].strip()))

    output_dir = Path(output_dir)
    input_video = Path(input_video)
    if preview is None:
        candidate = output_dir / "preview.mp4"
        if candidate.is_file():
            preview = candidate
    if not outputs:
        stem = input_video.stem
        for ext in (".webm", ".mp4", ".mov", ".webp"):
            candidate = output_dir / f"{stem}{ext}"
            if candidate.is_file():
                outputs.append(candidate)
    return {"preview": preview, "outputs": outputs, "log": stdout or ""}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/bgremove.py tests/test_bgremove_backend.py
git commit -m "test: cover bgremove backend routing"
```

---

### Task 2: Native Worker Skeleton

**Files:**
- Modify: `pipeline/bgremove.py`
- Create: `pipeline/bgremove_native/__init__.py`
- Create: `pipeline/bgremove_native/worker.py`
- Create: `pipeline/bgremove_native/runner.py`
- Modify: `tests/test_bgremove_backend.py`

**Interfaces:**
- Consumes: `select_bgremove_backend()`.
- Produces:
  - `native_worker_path() -> Path`
  - `_build_native_worker_cmd(...) -> list[str]`
  - `pipeline.bgremove_native.runner.run_bgremove_native(...) -> dict`

- [ ] **Step 1: Add failing native command routing test**

Append to `tests/test_bgremove_backend.py`:

```python
def test_build_native_worker_cmd_points_inside_repo(tmp_path):
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    cmd = bg._build_native_worker_cmd(
        input_video,
        out_dir,
        model="RMBG-2.0 HQ",
        formats="webm",
        fp16=True,
        infer_size=320,
        alpha_shrink=2,
        alpha_feather=3,
        bg_image=None,
    )

    assert "pipeline" in cmd[1]
    assert "bgremove_native" in cmd[1]
    assert "worker.py" in cmd[1]
    assert "--fp16" in cmd
    assert "--infer-size" in cmd and "320" in cmd
    assert "--alpha-shrink" in cmd and "2" in cmd
    assert "--alpha-feather" in cmd and "3" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py::test_build_native_worker_cmd_points_inside_repo -v
```

Expected: FAIL because `_build_native_worker_cmd` does not exist.

- [ ] **Step 3: Create native package skeleton**

Create `pipeline/bgremove_native/__init__.py`:

```python
"""Native background removal subprocess package for AniForge."""

SUPPORTED_MODELS = (
    "RMBG-2.0 HQ",
    "RVM MobileNetV3",
    "RVM ResNet50",
)
```

Create `pipeline/bgremove_native/runner.py`:

```python
from __future__ import annotations

from pathlib import Path


def run_bgremove_native(
    input_video: Path,
    output_dir: Path,
    *,
    model: str = "RMBG-2.0 HQ",
    formats: str = "webm",
    bg_image: Path | None = None,
    fp16: bool = True,
    infer_size: int = 0,
    alpha_shrink: int = 0,
    alpha_feather: int = 0,
) -> dict:
    raise RuntimeError("native background removal runner unavailable before Task 5")
```

Create `pipeline/bgremove_native/worker.py`:

```python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline.bgremove_native.runner import run_bgremove_native


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("model")
    parser.add_argument("formats")
    parser.add_argument("--bg")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--infer-size", type=int, default=0)
    parser.add_argument("--alpha-shrink", type=int, default=0)
    parser.add_argument("--alpha-feather", type=int, default=0)
    args = parser.parse_args(argv)

    t0 = time.time()
    result = run_bgremove_native(
        Path(args.input),
        Path(args.output_dir),
        model=args.model,
        formats=args.formats,
        bg_image=Path(args.bg) if args.bg else None,
        fp16=args.fp16,
        infer_size=args.infer_size,
        alpha_shrink=args.alpha_shrink,
        alpha_feather=args.alpha_feather,
    )
    if result.get("preview"):
        print(f"RESULT:preview:{result['preview']}", flush=True)
    for output in result.get("outputs") or []:
        print(f"RESULT:output:{output}", flush=True)
    print(f"DONE:{time.time() - t0:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Implement native command builder**

In `pipeline/bgremove.py`, add:

```python
def native_worker_path() -> Path:
    return Path(__file__).resolve().parent / "bgremove_native" / "worker.py"


def _build_native_worker_cmd(
    input_video: Path,
    output_dir: Path,
    *,
    model: str,
    formats: str,
    bg_image: Path | None,
    fp16: bool,
    infer_size: int,
    alpha_shrink: int,
    alpha_feather: int,
) -> list[str]:
    worker = native_worker_path()
    if not worker.is_file():
        raise FileNotFoundError(f"missing native bgremove worker: {worker}")
    cmd = [
        comfy_python(),
        str(worker),
        str(Path(input_video).resolve()),
        str(Path(output_dir).resolve()),
        model,
        formats,
    ]
    cmd.append("--fp16" if fp16 else "--no-fp16")
    if infer_size and int(infer_size) > 0:
        cmd.extend(["--infer-size", str(int(infer_size))])
    if alpha_shrink:
        cmd.extend(["--alpha-shrink", str(int(alpha_shrink))])
    if alpha_feather:
        cmd.extend(["--alpha-feather", str(int(alpha_feather))])
    if bg_image is not None:
        cmd.extend(["--bg", str(Path(bg_image).resolve())])
    return cmd
```

- [ ] **Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/bgremove.py pipeline/bgremove_native tests/test_bgremove_backend.py
git commit -m "feat: add native bgremove worker skeleton"
```

---

### Task 3: Native Video Utilities

**Files:**
- Create: `pipeline/bgremove_native/ffmpeg.py`
- Create: `pipeline/bgremove_native/compositor.py`
- Create: `pipeline/bgremove_native/decoder.py`
- Create: `pipeline/bgremove_native/encoder.py`
- Create: `tests/test_bgremove_native_runner.py`

**Interfaces:**
- Produces:
  - `resolve_ffmpeg() -> str`
  - `composite_frame(rgb, alpha, background) -> np.ndarray`
  - `make_rgba(fg, alpha) -> np.ndarray`
  - `VideoInfo(width, height, fps, frame_count)`
  - `VideoDecoder(path).frames()`
  - `write_h264_preview(path, frames, fps) -> Path`
  - `write_vp9_alpha(path, rgba_frames, fps) -> Path`

- [ ] **Step 1: Add lightweight compositor tests**

Create `tests/test_bgremove_native_runner.py`:

```python
import numpy as np

from pipeline.bgremove_native.compositor import composite_frame, make_rgba


def test_composite_frame_blends_alpha():
    fg = np.array([[[100, 50, 0]]], dtype=np.uint8)
    alpha = np.array([[0.25]], dtype=np.float32)
    bg = np.array([[[200, 200, 200]]], dtype=np.uint8)

    out = composite_frame(fg, alpha, bg)

    assert out.shape == (1, 1, 3)
    assert out[0, 0, 0] == 175


def test_make_rgba_adds_alpha_channel():
    fg = np.array([[[10, 20, 30]]], dtype=np.uint8)
    alpha = np.array([[0.5]], dtype=np.float32)

    rgba = make_rgba(fg, alpha)

    assert rgba.shape == (1, 1, 4)
    assert rgba[0, 0, 3] == 127
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_native_runner.py -v
```

Expected: import failure because `compositor.py` does not exist.

- [ ] **Step 3: Implement compositor and ffmpeg path helper**

Create `pipeline/bgremove_native/compositor.py`:

```python
from __future__ import annotations

import numpy as np


def _alpha3(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    return np.clip(a, 0.0, 1.0)


def composite_frame(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = _alpha3(alpha)
    fg = np.asarray(rgb, dtype=np.float32)
    bg = np.asarray(background, dtype=np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_rgba(fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    a8 = (a * 255.0).astype(np.uint8)
    return np.dstack([np.asarray(fg, dtype=np.uint8), a8])
```

Create `pipeline/bgremove_native/ffmpeg.py`:

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_ffmpeg() -> str:
    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and Path(env_ff).is_file():
        return env_ff
    which = shutil.which("ffmpeg")
    if which:
        return which
    return "ffmpeg"
```

- [ ] **Step 4: Port decoder and encoder from external repo**

Copy the current external implementations from `../videoBGremoval/video/decoder.py`, `encoder.py`, and `ffmpeg.py` into `pipeline/bgremove_native/`, adjusting imports to:

```python
from pipeline.bgremove_native.ffmpeg import resolve_ffmpeg
from pipeline.bgremove_native.compositor import make_rgba
```

Keep functions small and preserve existing behavior for audio extraction and muxing:

```python
def extract_audio(input_path: str, output_audio: str, *, ffmpeg_path: str) -> bool: ...
def mux_audio(video_path: str, audio_path: str, output_path: str, *, ffmpeg_path: str) -> None: ...
```

- [ ] **Step 5: Run lightweight tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_native_runner.py -v
```

Expected: compositor tests pass.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/bgremove_native tests/test_bgremove_native_runner.py
git commit -m "feat: add native bgremove video utilities"
```

---

### Task 4: Native Matting Engines

**Files:**
- Create: `pipeline/bgremove_native/engines.py`
- Modify: `pipeline/bgremove_native/__init__.py`
- Modify: `tests/test_bgremove_native_runner.py`

**Interfaces:**
- Produces:
  - `create_engine(model_name: str, *, fp16: bool, infer_long_edge: int | None, alpha_shrink: int, alpha_feather: int)`
  - Engine methods: `load(device: str)`, `predict(rgb_frame)`, `reset()`

- [ ] **Step 1: Add failing model validation test**

Append to `tests/test_bgremove_native_runner.py`:

```python
import pytest

from pipeline.bgremove_native.engines import create_engine


def test_create_engine_rejects_unknown_model_without_torch_import():
    with pytest.raises(ValueError, match="unknown model"):
        create_engine("not a model", fp16=True, infer_long_edge=None, alpha_shrink=0, alpha_feather=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_native_runner.py::test_create_engine_rejects_unknown_model_without_torch_import -v
```

Expected: import failure because `engines.py` does not exist.

- [ ] **Step 3: Port engine code with lazy heavy imports**

Create `pipeline/bgremove_native/engines.py` by porting:

- `../videoBGremoval/matting/base.py`
- `../videoBGremoval/matting/rmbg_engine.py`
- `../videoBGremoval/matting/rvm_engine.py`
- `../videoBGremoval/matting/__init__.py`

Use this public factory:

```python
from __future__ import annotations

from pipeline.bgremove_native import SUPPORTED_MODELS


def create_engine(
    model_name: str,
    *,
    fp16: bool = True,
    infer_long_edge: int | None = None,
    alpha_shrink: int = 0,
    alpha_feather: int = 0,
):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model {model_name!r}; choose from {SUPPORTED_MODELS}")
    if model_name == "RMBG-2.0 HQ":
        from pipeline.bgremove_native.rmbg_engine import RMBGEngine

        return RMBGEngine(
            fp16=fp16,
            infer_long_edge=infer_long_edge,
            alpha_shrink=alpha_shrink,
            alpha_feather=alpha_feather,
        )
    from pipeline.bgremove_native.rvm_engine import RVMEngine

    variant = "mobilenetv3" if model_name == "RVM MobileNetV3" else "resnet50"
    return RVMEngine(
        variant=variant,
        fp16=fp16,
        infer_long_edge=infer_long_edge,
        alpha_shrink=alpha_shrink,
        alpha_feather=alpha_feather,
    )
```

If the external repo's classes are easier to keep in separate files, create:

```text
pipeline/bgremove_native/base.py
pipeline/bgremove_native/rmbg_engine.py
pipeline/bgremove_native/rvm_engine.py
```

Keep torch, torchvision, transformers, and kornia imports inside those engine files, not in `pipeline/bgremove.py`.

- [ ] **Step 4: Run lightweight tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_native_runner.py -v
```

Expected: tests pass without loading torch for the unknown-model path.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/bgremove_native tests/test_bgremove_native_runner.py
git commit -m "feat: add native bgremove matting engines"
```

---

### Task 5: Native Runner End-To-End Wiring With Fakes

**Files:**
- Modify: `pipeline/bgremove_native/runner.py`
- Modify: `pipeline/bgremove.py`
- Modify: `tests/test_bgremove_native_runner.py`
- Modify: `tests/test_bgremove_backend.py`

**Interfaces:**
- Consumes: `create_engine`, `VideoDecoder`, encoder helpers.
- Produces: working `run_bgremove_native(...) -> {"preview": Path, "outputs": list[Path], "log": str}`.

- [ ] **Step 1: Add fake-runner test**

Append to `tests/test_bgremove_native_runner.py`:

```python
from pathlib import Path

import numpy as np

from pipeline.bgremove_native import runner


class FakeEngine:
    def load(self, device="cuda"):
        self.device = device

    def reset(self):
        self.did_reset = True

    def predict(self, frame):
        return np.ones(frame.shape[:2], dtype=np.float32)


class FakeDecoder:
    width = 2
    height = 2
    fps = 24.0
    frame_count = 1

    def __init__(self, path):
        self.path = path

    def frames(self):
        yield np.full((2, 2, 3), 100, dtype=np.uint8)


def test_run_bgremove_native_with_fakes(tmp_path, monkeypatch):
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "create_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(runner, "VideoDecoder", FakeDecoder)

    def fake_write_webm(path, frames, fps):
        Path(path).write_bytes(b"webm")
        return Path(path)

    def fake_write_preview(path, frames, fps):
        Path(path).write_bytes(b"mp4")
        return Path(path)

    monkeypatch.setattr(runner, "write_vp9_alpha", fake_write_webm)
    monkeypatch.setattr(runner, "write_h264_preview", fake_write_preview)

    result = runner.run_bgremove_native(input_video, out_dir, model="RMBG-2.0 HQ")

    assert result["preview"].name == "preview.mp4"
    assert result["outputs"][0].name == "clip.webm"
    assert result["preview"].is_file()
    assert result["outputs"][0].is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_native_runner.py::test_run_bgremove_native_with_fakes -v
```

Expected: FAIL because runner still raises "not implemented".

- [ ] **Step 3: Implement runner orchestration**

Replace `pipeline/bgremove_native/runner.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.bgremove_native.compositor import composite_frame, make_rgba
from pipeline.bgremove_native.decoder import VideoDecoder
from pipeline.bgremove_native.encoder import write_h264_preview, write_vp9_alpha
from pipeline.bgremove_native.engines import create_engine


def run_bgremove_native(
    input_video: Path,
    output_dir: Path,
    *,
    model: str = "RMBG-2.0 HQ",
    formats: str = "webm",
    bg_image: Path | None = None,
    fp16: bool = True,
    infer_size: int = 0,
    alpha_shrink: int = 0,
    alpha_feather: int = 0,
) -> dict:
    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"input video not found: {input_video}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [f.strip().lower() for f in formats.split(",") if f.strip()]
    if not requested:
        requested = ["webm"]

    decoder = VideoDecoder(input_video)
    engine = create_engine(
        model,
        fp16=fp16,
        infer_long_edge=int(infer_size) if infer_size and int(infer_size) > 0 else None,
        alpha_shrink=int(alpha_shrink),
        alpha_feather=int(alpha_feather),
    )
    engine.load(device="cuda")
    engine.reset()

    rgba_frames = []
    preview_frames = []
    gray_cache = None

    for frame in decoder.frames():
        alpha = engine.predict(frame)
        fg = (frame.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        rgba_frames.append(make_rgba(fg, alpha))
        if gray_cache is None:
            gray_cache = np.full_like(frame, 200)
        preview_frames.append(composite_frame(frame, alpha, gray_cache))

    outputs: list[Path] = []
    stem = input_video.stem
    if "webm" in requested:
        outputs.append(write_vp9_alpha(output_dir / f"{stem}.webm", rgba_frames, decoder.fps))

    preview = write_h264_preview(output_dir / "preview.mp4", preview_frames, decoder.fps)
    return {"preview": preview, "outputs": outputs, "log": ""}
```

If encoder function names differ after Task 3, adjust the imports and tests to the exact names implemented there.

- [ ] **Step 4: Wire `pipeline.bgremove.run_bgremove()` to native default**

In `pipeline/bgremove.py`, change `run_bgremove()`:

```python
backend = select_bgremove_backend()
if backend == "native":
    cmd = _build_native_worker_cmd(...)
    cwd = str(Path(__file__).resolve().parent.parent)
else:
    root = resolve_vbg_root(vbg_root)
    worker = root / "worker.py"
    cmd = [...]
    cwd = str(root)
```

Keep the existing subprocess env and result parsing:

```python
r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
log = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
if r.returncode != 0:
    raise RuntimeError(f"background removal worker failed (exit {r.returncode}):\n{log[-2000:]}")
parsed = _parse_worker_results(r.stdout or "", output_dir, input_video)
parsed["log"] = log
return parsed
```

- [ ] **Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py tests/test_bgremove_native_runner.py -v
```

Expected: all lightweight bgremove tests pass.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/bgremove.py pipeline/bgremove_native tests/test_bgremove_backend.py tests/test_bgremove_native_runner.py
git commit -m "feat: wire native bgremove backend"
```

---

### Task 6: Integration Test Gate And Documentation

**Files:**
- Create: `tests/test_bgremove_integration.py`
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: native backend from Task 5.
- Produces: opt-in integration test and setup docs.

- [ ] **Step 1: Add opt-in integration test**

Create `tests/test_bgremove_integration.py`:

```python
import os
import subprocess
from pathlib import Path

import pytest

from pipeline.bgremove import run_bgremove


pytestmark = pytest.mark.skipif(
    os.environ.get("ANIFORGE_RUN_BGREMOVE_INTEGRATION") != "1",
    reason="set ANIFORGE_RUN_BGREMOVE_INTEGRATION=1 to run GPU bgremove integration",
)


def test_native_bgremove_real_short_video(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIFORGE_BGREMOVE_BACKEND", raising=False)
    src = tmp_path / "solid.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=64x64:d=0.2:r=5",
        "-pix_fmt",
        "yuv420p",
        str(src),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    result = run_bgremove(src, tmp_path / "out", model="RMBG-2.0 HQ", formats="webm")

    assert result["preview"] is not None
    assert Path(result["preview"]).is_file()
    assert result["outputs"]
    assert Path(result["outputs"][0]).is_file()
```

- [ ] **Step 2: Run non-integration tests**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py tests/test_bgremove_native_runner.py tests/test_bgremove_integration.py -v
```

Expected: lightweight tests pass; integration test is skipped unless env var is set.

- [ ] **Step 3: Update `.env.example`**

Replace the current videoBGremoval section with:

```text
# Background removal backend.
# Default: native AniForge subprocess backend.
# Set to external only for legacy comparison with a sibling videoBGremoval repo.
# ANIFORGE_BGREMOVE_BACKEND=native
# VIDEO_BG_REMOVAL_ROOT=/path/to/videoBGremoval

# Optional local RMBG model directory. If unset, the native engine uses its bundled
# model directory when present or the Hugging Face cache/download path.
# RMBG_MODEL_DIR=
```

- [ ] **Step 4: Update README prerequisites/configuration**

Change the background-removal prerequisite from external `videoBGremoval` to native:

```markdown
4. Optional legacy comparison backend: **videoBGremoval** sibling folder
```

In the config table, describe:

```markdown
| `ANIFORGE_BGREMOVE_BACKEND` | `native` by default; set `external` only to compare with legacy videoBGremoval |
| `VIDEO_BG_REMOVAL_ROOT` | Optional legacy videoBGremoval repo path |
| `RMBG_MODEL_DIR` | Optional local RMBG-2.0 model directory |
```

- [ ] **Step 5: Commit**

```powershell
git add README.md .env.example tests/test_bgremove_integration.py
git commit -m "docs: document native bgremove backend"
```

---

### Task 7: Real Local Verification

**Files:**
- No required source changes unless verification exposes a bug.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: evidence that native bgremove works on this machine.

- [ ] **Step 1: Run full lightweight test set relevant to this change**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest tests/test_bgremove_backend.py tests/test_bgremove_native_runner.py tests/test_bgremove_integration.py tests/test_server.py tests/test_time_remap.py -v
```

Expected: tests pass, with `test_bgremove_integration.py` skipped unless env var is set.

- [ ] **Step 2: Run real native integration if GPU dependencies are available**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:ANIFORGE_RUN_BGREMOVE_INTEGRATION="1"
pytest tests/test_bgremove_integration.py -v
```

Expected: integration test passes and writes preview/WebM outputs in pytest temp dir.

- [ ] **Step 3: Compare external backend manually if `../videoBGremoval` remains available**

Use one existing short run output:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:ANIFORGE_BGREMOVE_BACKEND="native"
python - <<'PY'
from pathlib import Path
from pipeline.bgremove import run_bgremove
src = next(Path("runs").glob("*/action.mp4"))
print(run_bgremove(src, src.parent / "_native_bgremove_check", model="RMBG-2.0 HQ", formats="webm"))
PY
```

Then:

```powershell
$env:ANIFORGE_BGREMOVE_BACKEND="external"
python - <<'PY'
from pathlib import Path
from pipeline.bgremove import run_bgremove
src = next(Path("runs").glob("*/action.mp4"))
print(run_bgremove(src, src.parent / "_external_bgremove_check", model="RMBG-2.0 HQ", formats="webm"))
PY
```

Expected: both commands produce `preview.mp4` and `action.webm`. Native is accepted if dimensions and frame counts match the input and alpha is present in WebM.

- [ ] **Step 4: Final commit for verification fixes only**

If verification required fixes:

```powershell
git status --short
git add pipeline/bgremove.py pipeline/bgremove_native tests/test_bgremove_backend.py tests/test_bgremove_native_runner.py tests/test_bgremove_integration.py README.md .env.example
git commit -m "fix: stabilize native bgremove verification"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review

- Spec coverage: native default backend, external opt-in backend, subprocess isolation, output compatibility, model names, errors, docs, and integration test gate are covered by Tasks 1-7.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: `run_bgremove_native`, `select_bgremove_backend`, `_parse_worker_results`, and worker result-line contracts are consistent across tasks.
