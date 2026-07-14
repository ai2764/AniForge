import json

import numpy as np

from pipeline import stages


def test_stage_action_pose_selection_becomes_the_run_pose(tmp_path, monkeypatch):
    run_id = "run1"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "input.png").write_bytes(b"not a real image")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pose_mode": "standing",
                "seed": 123,
                "size": [512, 768],
                "extracted": True,
                "idle_done": True,
            }
        ),
        encoding="utf-8",
    )

    base = np.zeros((22, 3), dtype=np.float64)
    np.save(run_dir / "extract_pose.npy", base)

    def fake_run_subprocess(_label, _args):
        posed = np.zeros((4, 22, 3), dtype=np.float64)
        posed[:, 12, 0] = np.linspace(0.0, 0.3, 4)
        np.savez(run_dir / "action_seed123.npz", posed_joints=posed)

    def fake_render(_posed, out, camera=None):
        out.write_bytes(b"mp4")

    monkeypatch.setattr(stages, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(stages, "render_smplx_guide", fake_render)
    monkeypatch.setattr(stages, "_pad_to_aspect", lambda src, dst, w, h: dst.write_bytes(b"guide"))
    monkeypatch.setattr(stages, "_rel_url", lambda path: "/" + path.name if path else None)

    result = stages.stage_action(
        run_id,
        action_prompt="wave",
        pose_mode="lying",
        runs_dir=tmp_path,
    )

    assert result["errors"] == {}
    assert result["pose_mode"] == "lying"
    assert result["action_lock_lower"] is True

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["pose_mode"] == "lying"
    assert "action_pose_mode" not in meta
    assert meta["idle_done"] is True
