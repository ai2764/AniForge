"""Video background removal via external videoBGremoval worker (subprocess).

Isolates RVM/RMBG VRAM: process exit unloads models. Does not import torch
into the motion-portrait server process.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Sibling repo by default; override with VIDEO_BG_REMOVAL_ROOT.
DEFAULT_VBG_ROOT = Path(
    os.environ.get("VIDEO_BG_REMOVAL_ROOT", r"C:\Users\AIBOX\dev\videoBGremoval")
)

# Models accepted by videoBGremoval.matting.create_engine
BG_MODELS = (
    "RVM MobileNetV3",
    "RVM ResNet50",
    "RMBG-2.0 HQ",
)


def resolve_vbg_root(root: Path | None = None) -> Path:
    p = Path(root or DEFAULT_VBG_ROOT)
    if not p.is_dir():
        raise FileNotFoundError(
            f"videoBGremoval not found at {p}. "
            "Clone it or set VIDEO_BG_REMOVAL_ROOT."
        )
    return p.resolve()


def resolve_vbg_python(root: Path) -> str:
    """Prefer videoBGremoval .venv (has torch + matting deps)."""
    for cand in (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ):
        if cand.is_file():
            return str(cand)
    # Fallback: same as comfy-scail env if set
    comfy = Path(r"C:\Users\AIBOX\anaconda3\envs\comfy-scail\python.exe")
    if comfy.is_file():
        return str(comfy)
    import sys
    return sys.executable


def run_bgremove(
    input_video: Path,
    output_dir: Path,
    *,
    model: str = "RVM MobileNetV3",
    formats: str = "webm",
    bg_image: Path | None = None,
    fp16: bool = True,
    infer_size: int = 0,
    alpha_shrink: int = 0,
    alpha_feather: int = 0,
    vbg_root: Path | None = None,
) -> dict:
    """Run videoBGremoval worker on one video.

    Returns dict with keys: preview (mp4 path|None), outputs (list[Path]), log (str).
    """
    root = resolve_vbg_root(vbg_root)
    worker = root / "worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"missing worker.py in {root}")

    if model not in BG_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {BG_MODELS}")

    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"input video not found: {input_video}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    py = resolve_vbg_python(root)
    cmd = [
        py,
        str(worker),
        str(input_video.resolve()),
        str(output_dir.resolve()),
        model,
        formats,
    ]
    if fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no-fp16")
    if infer_size and int(infer_size) > 0:
        cmd.extend(["--infer-size", str(int(infer_size))])
    if alpha_shrink:
        cmd.extend(["--alpha-shrink", str(int(alpha_shrink))])
    if alpha_feather:
        cmd.extend(["--alpha-feather", str(int(alpha_feather))])
    if bg_image is not None:
        cmd.extend(["--bg", str(Path(bg_image).resolve())])

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    # Prefer portable ffmpeg from videoBGremoval
    portable_ff = root / "portable" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if portable_ff.is_file():
        env["FFMPEG_PATH"] = str(portable_ff)

    print(f"[bgremove] {' '.join(cmd)}", flush=True)
    r = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    if r.returncode != 0:
        raise RuntimeError(
            f"videoBGremoval worker failed (exit {r.returncode}):\n{log[-2000:]}"
        )

    preview = None
    outputs: list[Path] = []
    for line in (r.stdout or "").splitlines():
        if line.startswith("RESULT:preview:"):
            preview = Path(line.split(":", 2)[2].strip())
        elif line.startswith("RESULT:output:"):
            outputs.append(Path(line.split(":", 2)[2].strip()))

    # Fallbacks if RESULT lines missing
    if preview is None:
        p = output_dir / "preview.mp4"
        if p.is_file():
            preview = p
    if not outputs:
        stem = input_video.stem
        for ext in (".webm", ".mp4", ".mov", ".webp"):
            cand = output_dir / f"{stem}{ext}"
            if cand.is_file():
                outputs.append(cand)

    return {"preview": preview, "outputs": outputs, "log": log}
