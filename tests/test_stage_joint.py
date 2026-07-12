import json
import numpy as np
import pipeline.stages as stages


def _make_run(tmp_path, monkeypatch):
    run_id = "r1"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    seed = 42
    (run_dir / "meta.json").write_text(json.dumps(
        {"seed": seed, "scale": 1.0, "size": [64, 128],
         "pose_mode": "standing", "action_done": True}), encoding="utf-8")
    np.savez(run_dir / f"action_seed{seed}.npz",
             posed_joints=np.zeros((4, 22, 3), dtype=np.float64))
    (run_dir / "action_skel.mp4").write_bytes(b"PLAIN")  # the persistent "before"
    (run_dir / "input.png").write_bytes(b"x")  # _find_image is called before size check

    # Neutralize heavy render/spring: create the output file, identity spring.
    monkeypatch.setattr(stages, "spring_follow",
                        lambda P, fps, **k: P + 1.0)
    monkeypatch.setattr(stages, "render_smplx_guide",
                        lambda P, out, camera=None: out.write_bytes(b"OVERSHOT"))
    monkeypatch.setattr(stages, "_pad_to_aspect",
                        lambda inp, out, w, h, **k: out.write_bytes(inp.read_bytes()))
    monkeypatch.setattr(stages, "_load_action_base_pose",
                        lambda rd, s: (None, None, "extract"))
    return run_dir, seed


def test_preview_is_nondestructive(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    out = stages.stage_joint_overshoot("r1", mode="preview", runs_dir=tmp_path)
    assert out["errors"] == {}
    assert (run_dir / "action_joint_skel.mp4").read_bytes() == b"OVERSHOT"
    assert (run_dir / f"action_joint_seed{seed}.npz").is_file()
    # before-file and SCAIL guide untouched
    assert (run_dir / "action_skel.mp4").read_bytes() == b"PLAIN"
    assert not (run_dir / "action_guide.mp4").is_file()


def test_carry_then_uncarry_toggles_guide_and_meta(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    stages.stage_joint_overshoot("r1", mode="preview", runs_dir=tmp_path)
    c = stages.stage_joint_overshoot("r1", mode="carry", runs_dir=tmp_path)
    assert c["errors"] == {}
    assert c["joint_overshoot"] is True
    assert (run_dir / "action_guide.mp4").read_bytes() == b"OVERSHOT"  # from joint skel
    assert json.loads((run_dir / "meta.json").read_text())["joint_overshoot"] is True

    u = stages.stage_joint_overshoot("r1", mode="uncarry", runs_dir=tmp_path)
    assert u["joint_overshoot"] is False
    assert (run_dir / "action_guide.mp4").read_bytes() == b"PLAIN"  # from action_skel
    assert json.loads((run_dir / "meta.json").read_text())["joint_overshoot"] is False


def test_carry_self_springs_without_preview(tmp_path, monkeypatch):
    run_dir, seed = _make_run(tmp_path, monkeypatch)
    # No preview call first (Run-all one-shot path).
    c = stages.stage_joint_overshoot("r1", mode="carry", runs_dir=tmp_path)
    assert c["errors"] == {}
    assert (run_dir / f"action_joint_seed{seed}.npz").is_file()  # sprang on demand
    assert (run_dir / "action_guide.mp4").read_bytes() == b"OVERSHOT"
