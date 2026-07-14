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
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed; skipping GPU bgremove integration")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this PyTorch environment; skipping GPU bgremove integration")

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
