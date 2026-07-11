"""Video background removal via external videoBGremoval worker (subprocess).

Isolates RVM/RMBG VRAM: process exit unloads models. Does not import torch
into the AniForge server process.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from pipeline.paths import comfy_python, video_bg_removal_root

# Models accepted by videoBGremoval.matting.create_engine
BG_MODELS = (
    "RMBG-2.0 HQ",
    "RVM MobileNetV3",
    "RVM ResNet50",
)


def resolve_vbg_root(root: Path | None = None) -> Path:
    if root is not None:
        p = Path(root)
        if not p.is_dir():
            raise FileNotFoundError(
                f"videoBGremoval not found at {p}. "
                "Clone it or set VIDEO_BG_REMOVAL_ROOT."
            )
        return p.resolve()
    return video_bg_removal_root().resolve()


def resolve_vbg_python(root: Path) -> str:
    """Prefer videoBGremoval .venv (has torch + matting deps)."""
    for cand in (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ):
        if cand.is_file():
            return str(cand)
    return comfy_python()


def run_bgremove(
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


def resolve_ffmpeg(vbg_root: Path | None = None) -> str:
    """Prefer videoBGremoval portable ffmpeg, then FFMPEG_PATH, then PATH."""
    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and Path(env_ff).is_file():
        return env_ff
    try:
        root = resolve_vbg_root(vbg_root)
        portable = root / "portable" / "ffmpeg" / "bin" / "ffmpeg.exe"
        if portable.is_file():
            return str(portable)
    except FileNotFoundError:
        pass
    which = shutil.which("ffmpeg")
    if which:
        return which
    return "ffmpeg"


def webm_to_prores_alpha(
    webm_path: Path,
    mov_path: Path,
    *,
    vbg_root: Path | None = None,
) -> Path:
    """Convert VP9+alpha WebM to CapCut-friendly ProRes 4444 MOV with real alpha.

    Must decode with libvpx-vp9 — the default VP9 decoder drops the alpha plane
    and yields opaque black-background frames (broken for NLE compositing).
    """
    webm_path = Path(webm_path)
    mov_path = Path(mov_path)
    if not webm_path.is_file():
        raise FileNotFoundError(f"webm not found: {webm_path}")
    mov_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(vbg_root)
    cmd = [
        ffmpeg,
        "-y",
        "-c:v",
        "libvpx-vp9",
        "-i",
        str(webm_path.resolve()),
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4444",
        "-pix_fmt",
        "yuva444p10le",
        "-an",
        str(mov_path.resolve()),
    ]
    print(f"[bgremove] alpha mov: {' '.join(cmd)}", flush=True)
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0 or not mov_path.is_file() or mov_path.stat().st_size < 1000:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-1500:]
        raise RuntimeError(
            f"webm→ProRes alpha failed (exit {r.returncode}):\n{tail}"
        )
    return mov_path
