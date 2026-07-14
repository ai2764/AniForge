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
