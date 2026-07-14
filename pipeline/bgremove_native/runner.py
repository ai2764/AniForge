from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.bgremove_native.compositor import composite_frame, make_rgba
from pipeline.bgremove_native.decoder import VideoDecoder
from pipeline.bgremove_native.encoder import (
    extract_audio,
    mux_audio,
    write_h264_preview,
    write_vp9_alpha,
)
from pipeline.bgremove_native.engines import create_engine


def run_bgremove_native(
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
) -> dict:
    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"input video not found: {input_video}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [fmt.strip().lower() for fmt in formats.split(",") if fmt.strip()]
    if not requested:
        requested = ["webm"]

    decoder = VideoDecoder(input_video)
    engine = create_engine(
        model,
        fp16=fp16,
        infer_long_edge=int(infer_size) if infer_size and int(infer_size) > 0 else None,
        alpha_shrink=int(alpha_shrink),
        alpha_feather=int(alpha_feather),
    )
    engine.load(device="cuda")
    engine.reset()

    rgba_frames = []
    preview_frames = []
    gray_cache = None

    for frame in decoder.frames():
        alpha = engine.predict(frame)
        rgba_frames.append(make_rgba(frame, alpha))
        if gray_cache is None:
            gray_cache = np.full_like(frame, 200)
        preview_frames.append(composite_frame(frame, alpha, gray_cache))

    outputs: list[Path] = []
    log_messages: list[str] = []
    if "webm" in requested:
        webm_path = output_dir / f"{input_video.stem}.webm"
        audio_path = output_dir / f"{input_video.stem}.audio.mka"
        muxed_path = output_dir / f"{input_video.stem}.muxed.webm"
        write_vp9_alpha(webm_path, rgba_frames, decoder.fps)
        if extract_audio(str(input_video), str(audio_path)):
            try:
                mux_audio(str(webm_path), str(audio_path), str(muxed_path))
                muxed_path.replace(webm_path)
            except Exception as exc:
                log_messages.append(
                    f"Warning: Audio mux failed; using video-only WebM: {exc}"
                )
            finally:
                audio_path.unlink(missing_ok=True)
                muxed_path.unlink(missing_ok=True)
        outputs.append(webm_path)

    preview = write_h264_preview(output_dir / "preview.mp4", preview_frames, decoder.fps)
    return {"preview": preview, "outputs": outputs, "log": "\n".join(log_messages)}
