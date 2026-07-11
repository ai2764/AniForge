"""Portable path resolution for public installs (no machine-specific defaults).

Override any location with environment variables (see ``.env.example``).

Environment
-----------
COMFYUI_SCAIL_ROOT   Root of ComfyUI-scail (contains ``input/``, ``output/``, ``custom_nodes/``)
COMFYUI_INPUT        Override Comfy input folder
COMFYUI_OUTPUT       Override Comfy output folder
COMFYUI_KIMODO       Path to Kimodo package dir (…/ComfyUI-Kimodo/kimodo)
COMFYUI_MOTIONDIFF   Path to ComfyUI-MotionDiff custom node root
COMFY_PYTHON         Python executable with torch/matting deps (optional)
VIDEO_BG_REMOVAL_ROOT  Root of videoBGremoval repo
STANDEE_DIR          Default folder of character standee images (optional)
MANNEQUIN_IMAGE      Default neutral mannequin PNG (optional)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# AniForge repository root (parent of pipeline/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    v = (os.environ.get(name) or "").strip()
    return v or None


def env_path(name: str) -> Path | None:
    v = _env(name)
    return Path(v).expanduser() if v else None


def comfy_scail_root() -> Path | None:
    """ComfyUI-scail install root, if known."""
    p = env_path("COMFYUI_SCAIL_ROOT")
    if p is not None:
        return p
    # Common sibling layout: <parent>/ComfyUI-scail next to AniForge
    sibling = REPO_ROOT.parent / "ComfyUI-scail"
    if sibling.is_dir():
        return sibling
    return None


def comfy_input_dir() -> Path:
    p = env_path("COMFYUI_INPUT")
    if p is not None:
        return p
    root = comfy_scail_root()
    if root is not None:
        return root / "input"
    # Local staging under the repo (created on demand)
    return REPO_ROOT / ".comfy" / "input"


def comfy_output_dir() -> Path:
    p = env_path("COMFYUI_OUTPUT")
    if p is not None:
        return p
    root = comfy_scail_root()
    if root is not None:
        return root / "output"
    return REPO_ROOT / ".comfy" / "output"


def kimodo_package_dir() -> Path:
    """Directory containing the ``kimodo`` Python package (for sys.path)."""
    p = env_path("COMFYUI_KIMODO")
    if p is not None:
        return p
    root = comfy_scail_root()
    if root is not None:
        cand = root / "custom_nodes" / "ComfyUI-Kimodo" / "kimodo"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Kimodo package not found. Set COMFYUI_KIMODO or COMFYUI_SCAIL_ROOT "
        "to your ComfyUI-scail install (custom_nodes/ComfyUI-Kimodo/kimodo)."
    )


def motiondiff_root() -> Path:
    p = env_path("COMFYUI_MOTIONDIFF")
    if p is not None:
        return p
    root = comfy_scail_root()
    if root is not None:
        cand = root / "custom_nodes" / "ComfyUI-MotionDiff"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "ComfyUI-MotionDiff not found. Set COMFYUI_MOTIONDIFF or COMFYUI_SCAIL_ROOT."
    )


def video_bg_removal_root() -> Path:
    p = env_path("VIDEO_BG_REMOVAL_ROOT")
    if p is not None:
        return p
    sibling = REPO_ROOT.parent / "videoBGremoval"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(
        "videoBGremoval not found. Clone it next to AniForge or set "
        "VIDEO_BG_REMOVAL_ROOT to its root directory."
    )


def comfy_python() -> str:
    """Interpreter for heavy deps (Kimodo standalone / matting). Prefer COMFY_PYTHON."""
    v = _env("COMFY_PYTHON")
    if v and Path(v).is_file():
        return v
    try:
        root = video_bg_removal_root()
        for cand in (
            root / ".venv" / "Scripts" / "python.exe",
            root / ".venv" / "bin" / "python",
        ):
            if cand.is_file():
                return str(cand)
    except FileNotFoundError:
        pass
    which = shutil.which("python") or shutil.which("python3")
    return which or sys.executable


def standee_dir() -> Path | None:
    """Optional default folder of standee images (STANDEE_DIR)."""
    return env_path("STANDEE_DIR")


def mannequin_image() -> Path | None:
    p = env_path("MANNEQUIN_IMAGE")
    if p is not None and p.is_file():
        return p
    return None
