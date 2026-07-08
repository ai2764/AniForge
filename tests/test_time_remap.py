import json
import subprocess
from pathlib import Path

import numpy as np

from pipeline.spring_time_remap import remap_indices, write_video


def test_chosen_params_small_overshoot():
    idx = remap_indices(105, 24, 0.42, 4.2, 2.4, 1.15)
    back = max((idx[i-1]-idx[i]) for i in range(1, len(idx)))
    assert 1.0 < back < 3.0            # small overshoot, not a full replay
    assert idx[0] == 0 and idx[-1] <= 104


def test_write_video_browser_compatible_h264(tmp_path):
    """Timed action clips must be H.264/yuv420p so the HTML5 player can play them."""
    frames = [np.zeros((64, 48, 3), dtype=np.uint8) for _ in range(8)]
    for i, f in enumerate(frames):
        f[:] = (i * 20, 40, 80)
    out = Path(tmp_path) / "clip.mp4"
    write_video(out, frames, 24.0)
    assert out.is_file() and out.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "json",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
