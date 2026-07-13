import json
import subprocess
from pathlib import Path

import numpy as np

from pipeline.spring_time_remap import remap_indices, write_video
from pipeline.stages import stage_time_overshoot


def test_chosen_params_small_overshoot():
    idx = remap_indices(105, 24, 0.42, 4.2, 2.4, 1.15)
    back = max((idx[i-1]-idx[i]) for i in range(1, len(idx)))
    assert 1.0 < back < 3.0            # small overshoot, not a full replay
    assert idx[0] == 0 and idx[-1] <= 104


def test_stage_time_overshoot_passes_user_strength_and_duration(tmp_path, monkeypatch):
    run_id = "time-controls"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "seed": 7}), encoding="utf-8"
    )
    (run_dir / "action.mp4").write_bytes(b"source")
    received = {}

    def fake_remap(inp, out, **kwargs):
        received.update(kwargs)
        Path(out).write_bytes(b"timed")
        return {"has_alpha": False, "out": out, "out_webm": None}

    monkeypatch.setattr("pipeline.stages.time_remap_file", fake_remap)

    result = stage_time_overshoot(
        run_id, runs_dir=tmp_path, overshoot_b=0.6, overshoot_t=1.5
    )

    assert result["errors"] == {}
    assert received["b"] == 0.6
    assert received["t"] == 1.5
    assert received["d"] == 4.2
    assert received["f"] == 2.4


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
