import os
import subprocess
from pathlib import Path
import sys

import pytest

import pipeline.bgremove as bg
from pipeline.bgremove_native import worker


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


def test_run_bgremove_ignores_result_lines_on_stderr(monkeypatch, tmp_path):
    input_video = tmp_path / "action.mp4"
    input_video.write_bytes(b"fake")
    stdout_output = tmp_path / "stdout.webm"
    stderr_output = tmp_path / "stderr.webm"

    monkeypatch.setenv("ANIFORGE_BGREMOVE_BACKEND", "external")
    monkeypatch.setattr(bg, "resolve_vbg_root", lambda root=None: tmp_path)
    monkeypatch.setattr(bg, "resolve_vbg_python", lambda root: "python")
    (tmp_path / "worker.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        bg,
        "_run_worker_command_streams",
        lambda cmd, cwd, env: (
            0,
            f"RESULT:output:{stdout_output}\n",
            f"RESULT:output:{stderr_output}\n",
        ),
        raising=False,
    )

    parsed = bg.run_bgremove(input_video, tmp_path / "out")

    assert parsed["outputs"] == [stdout_output]
    assert f"RESULT:output:{stderr_output}" in parsed["log"]


def test_run_bgremove_uses_native_worker_by_default(monkeypatch, tmp_path):
    input_video = tmp_path / "action.mp4"
    input_video.write_bytes(b"fake")
    output = tmp_path / "out" / "action.webm"
    captured = {}

    monkeypatch.delenv("ANIFORGE_BGREMOVE_BACKEND", raising=False)
    monkeypatch.setattr(
        bg,
        "_build_native_worker_cmd",
        lambda *args, **kwargs: ["native-python", "native-worker"],
    )
    monkeypatch.setattr(
        bg,
        "resolve_vbg_root",
        lambda root=None: pytest.fail("external backend selected"),
    )

    def fake_run(cmd, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return 0, f"RESULT:output:{output}\n", ""

    monkeypatch.setattr(bg, "_run_worker_command_streams", fake_run)

    parsed = bg.run_bgremove(input_video, tmp_path / "out")

    assert captured["cmd"] == ["native-python", "native-worker"]
    assert captured["cwd"] == Path(bg.__file__).resolve().parent.parent
    assert parsed["outputs"] == [output]


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


def test_native_worker_help_runs_from_arbitrary_cwd():
    worker = Path(__file__).resolve().parents[1] / "pipeline" / "bgremove_native" / "worker.py"

    result = subprocess.run(
        [sys.executable, str(worker), "--help"],
        cwd=Path(__file__).resolve().parent,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_native_worker_propagates_runner_log_before_done(monkeypatch, tmp_path, capsys):
    preview = tmp_path / "preview.mp4"
    output = tmp_path / "clip.webm"

    monkeypatch.setattr(
        worker,
        "run_bgremove_native",
        lambda *args, **kwargs: {
            "preview": preview,
            "outputs": [output],
            "log": "Warning: Audio mux failed; using video-only WebM",
        },
    )

    assert worker.main(["input.mp4", str(tmp_path), "RMBG-2.0 HQ", "webm"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        f"RESULT:preview:{preview}",
        f"RESULT:output:{output}",
        "WARN: Warning: Audio mux failed; using video-only WebM",
    ]
    assert len(lines) == 4
    assert lines[-1].startswith("DONE:")
