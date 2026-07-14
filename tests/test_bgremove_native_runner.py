import numpy as np
import pytest
from pathlib import Path

from pipeline.bgremove_native.compositor import composite_frame, make_rgba
from pipeline.bgremove_native.engines import create_engine
from pipeline.bgremove_native.rmbg_engine import resolve_model_source
from pipeline.bgremove_native import runner


def test_composite_frame_blends_alpha():
    fg = np.array([[[100, 50, 0]]], dtype=np.uint8)
    alpha = np.array([[0.25]], dtype=np.float32)
    bg = np.array([[[200, 200, 200]]], dtype=np.uint8)

    out = composite_frame(fg, alpha, bg)

    assert out.shape == (1, 1, 3)
    assert out[0, 0, 0] == 175


def test_make_rgba_adds_alpha_channel():
    fg = np.array([[[10, 20, 30]]], dtype=np.uint8)
    alpha = np.array([[0.5]], dtype=np.float32)

    rgba = make_rgba(fg, alpha)

    assert rgba.shape == (1, 1, 4)
    assert rgba[0, 0, 3] == 127


def test_create_engine_rejects_unknown_model_without_torch_import():
    with pytest.raises(ValueError, match="unknown model"):
        create_engine(
            "not a model",
            fp16=True,
            infer_long_edge=None,
            alpha_shrink=0,
            alpha_feather=0,
        )


def test_resolve_rmbg_model_source_prefers_environment_variable(monkeypatch):
    monkeypatch.setenv("RMBG_MODEL_DIR", "/models/RMBG-2.0")

    assert resolve_model_source() == "/models/RMBG-2.0"


def test_resolve_rmbg_model_source_uses_bundled_directory(monkeypatch, tmp_path):
    bundled_model_dir = tmp_path / "models" / "RMBG-2.0"
    bundled_model_dir.mkdir(parents=True)
    monkeypatch.delenv("RMBG_MODEL_DIR", raising=False)
    monkeypatch.setattr(
        "pipeline.bgremove_native.rmbg_engine._BUNDLED_MODEL_DIR", bundled_model_dir
    )

    assert resolve_model_source() == str(bundled_model_dir)


class FakeEngine:
    def load(self, device="cuda"):
        self.device = device

    def reset(self):
        self.did_reset = True

    def predict(self, frame):
        return np.ones(frame.shape[:2], dtype=np.float32)


class HalfAlphaFakeEngine(FakeEngine):
    def predict(self, frame):
        return np.full(frame.shape[:2], 0.5, dtype=np.float32)


class FakeDecoder:
    width = 2
    height = 2
    fps = 24.0
    frame_count = 1

    def __init__(self, path):
        self.path = path

    def frames(self):
        yield np.full((2, 2, 3), 100, dtype=np.uint8)


def test_run_bgremove_native_with_fakes(tmp_path, monkeypatch):
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "create_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(runner, "VideoDecoder", FakeDecoder)

    def fake_write_webm(path, frames, fps):
        Path(path).write_bytes(b"webm")
        return Path(path)

    def fake_write_preview(path, frames, fps):
        Path(path).write_bytes(b"mp4")
        return Path(path)

    monkeypatch.setattr(runner, "write_vp9_alpha", fake_write_webm)
    monkeypatch.setattr(runner, "write_h264_preview", fake_write_preview)

    result = runner.run_bgremove_native(input_video, out_dir, model="RMBG-2.0 HQ")

    assert result["preview"].name == "preview.mp4"
    assert result["outputs"][0].name == "clip.webm"
    assert result["preview"].is_file()
    assert result["outputs"][0].is_file()


def test_run_bgremove_native_writes_straight_alpha_webm_frames(tmp_path, monkeypatch):
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "create_engine", lambda *a, **k: HalfAlphaFakeEngine())
    monkeypatch.setattr(runner, "VideoDecoder", FakeDecoder)

    def fake_write_webm(path, frames, fps):
        rgba_frame = next(iter(frames))
        np.testing.assert_array_equal(rgba_frame[:, :, :3], np.full((2, 2, 3), 100, dtype=np.uint8))
        np.testing.assert_array_equal(rgba_frame[:, :, 3], np.full((2, 2), 127, dtype=np.uint8))
        Path(path).write_bytes(b"webm")
        return Path(path)

    monkeypatch.setattr(runner, "write_vp9_alpha", fake_write_webm)
    monkeypatch.setattr(runner, "write_h264_preview", lambda path, frames, fps: Path(path))

    runner.run_bgremove_native(input_video, out_dir, model="RMBG-2.0 HQ")


def test_run_bgremove_native_muxes_source_audio_into_webm(tmp_path, monkeypatch):
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    mux_calls = []

    monkeypatch.setattr(runner, "create_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(runner, "VideoDecoder", FakeDecoder)
    monkeypatch.setattr(runner, "extract_audio", lambda *args, **kwargs: True)

    def fake_write_webm(path, frames, fps):
        Path(path).write_bytes(b"video")
        return Path(path)

    def fake_mux_audio(video_path, audio_path, output_path):
        mux_calls.append((Path(video_path), Path(audio_path), Path(output_path)))
        Path(output_path).write_bytes(b"webm-with-audio")

    monkeypatch.setattr(runner, "write_vp9_alpha", fake_write_webm)
    monkeypatch.setattr(runner, "mux_audio", fake_mux_audio)
    monkeypatch.setattr(runner, "write_h264_preview", lambda path, frames, fps: Path(path))

    result = runner.run_bgremove_native(input_video, out_dir, model="RMBG-2.0 HQ")

    assert len(mux_calls) == 1
    assert mux_calls[0][0] == result["outputs"][0]
    assert mux_calls[0][2] == out_dir / "clip.muxed.webm"
    assert result["outputs"][0].read_bytes() == b"webm-with-audio"


def test_run_bgremove_native_keeps_no_audio_webm_when_audio_mux_fails(tmp_path, monkeypatch):
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(runner, "create_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(runner, "VideoDecoder", FakeDecoder)
    monkeypatch.setattr(runner, "extract_audio", lambda *args, **kwargs: True)

    def fake_write_webm(path, frames, fps):
        Path(path).write_bytes(b"video-without-audio")
        return Path(path)

    def failing_mux_audio(video_path, audio_path, output_path):
        raise RuntimeError("AAC cannot be stream-copied into WebM")

    monkeypatch.setattr(runner, "write_vp9_alpha", fake_write_webm)
    monkeypatch.setattr(runner, "mux_audio", failing_mux_audio)
    monkeypatch.setattr(runner, "write_h264_preview", lambda path, frames, fps: Path(path))

    result = runner.run_bgremove_native(input_video, out_dir, model="RMBG-2.0 HQ")

    assert result["outputs"] == [out_dir / "clip.webm"]
    assert result["outputs"][0].read_bytes() == b"video-without-audio"
    assert "Audio mux failed" in result["log"]
