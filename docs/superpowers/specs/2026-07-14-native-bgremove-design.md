# Native Background Removal Design

## Goal

AniForge will own its background-removal implementation instead of depending on a sibling `videoBGremoval` repository. The user-facing behavior of `stage_bgremove()` remains unchanged: given an `idle`, `action`, or uploaded video, it produces browser preview video, VP9 alpha WebM, and CapCut-friendly ProRes alpha MOV outputs.

Expected outputs keep the existing names:

```text
idle_nobg.mp4
idle_nobg.webm
idle_nobg_alpha.mov
action_nobg.mp4
action_nobg.webm
action_nobg_alpha.mov
upload_nobg.mp4
upload_nobg.webm
upload_nobg_alpha.mov
```

## Scope

This change covers the runtime path used by AniForge for video background removal. It does not redesign the UI, change the Run All pipeline semantics, or add new matting models beyond the currently supported model names.

Supported model names remain:

```text
RMBG-2.0 HQ
RVM MobileNetV3
RVM ResNet50
```

## Architecture

Add a native background-removal package under `pipeline/bgremove_native/`:

```text
pipeline/bgremove_native/
  __init__.py
  engines.py
  decoder.py
  encoder.py
  runner.py
  worker.py
```

`engines.py` owns model selection and alpha prediction. It wraps RMBG-2.0 and RVM behind one interface:

```python
engine = create_engine(model_name, fp16=True, infer_long_edge=None)
engine.load(device="cuda")
alpha = engine.predict(rgb_frame)
engine.reset()
```

`decoder.py` reads frames, fps, dimensions, frame count, and audio metadata from input videos.

`encoder.py` writes:

- H.264 preview MP4 with alpha composited over neutral gray.
- VP9 WebM with alpha.
- ProRes 4444 MOV from the VP9 alpha output, using the same alpha-preserving decode requirement already documented in `pipeline.bgremove.webm_to_prores_alpha`.

`runner.py` exposes `run_bgremove_native(input_video, output_dir, ...)`, returning the same shape as the current `run_bgremove()`:

```python
{"preview": Path | None, "outputs": list[Path], "log": str}
```

`worker.py` is a subprocess entry point. AniForge will invoke this worker through the current `pipeline.bgremove.run_bgremove()` facade so the server process does not import torch or hold matting models in VRAM.

## Data Flow

`stage_bgremove()` continues to choose the source video and output labels. For each label:

1. `pipeline.bgremove.run_bgremove()` starts `pipeline/bgremove_native/worker.py` in a subprocess.
2. The worker loads the selected matting engine.
3. The worker decodes the source video frame by frame.
4. Each frame produces an alpha matte.
5. Encoders write WebM alpha and MP4 preview.
6. AniForge converts or writes ProRes alpha MOV for CapCut.
7. `_bgremove_one()` copies the outputs into the run directory using the existing `*_nobg*` names.

The worker prints structured result lines compatible with the current parser:

```text
RESULT:preview:<path>
RESULT:output:<path>
```

## Compatibility

`pipeline/bgremove.py` remains the public integration point. Its default path changes from external `videoBGremoval/worker.py` to native `pipeline/bgremove_native/worker.py`.

During migration, an explicit fallback can remain available for comparison:

- Native is the default.
- If `ANIFORGE_BGREMOVE_BACKEND=external`, use `VIDEO_BG_REMOVAL_ROOT`.
- If native fails before model load because required files are missing, return a clear setup error instead of silently switching backends.

This keeps behavior predictable while still allowing side-by-side quality checks.

## Dependencies

AniForge already depends on video and image tooling. Native background removal adds or formalizes these runtime dependencies:

```text
torch
torchvision
opencv-python
numpy
Pillow
transformers
kornia
huggingface_hub
```

Because these are heavy GPU dependencies, they should not be imported by `server/app.py` or normal lightweight tests. Imports stay inside the worker and engine modules.

RMBG model assets should be resolved in this order:

1. `RMBG_MODEL_DIR` if set.
2. `pipeline/bgremove_native/models/RMBG-2.0` if present.
3. Hugging Face cache/download path used by the engine.

## Error Handling

Errors should preserve the existing API shape: `stage_bgremove()` returns an `errors` dictionary instead of crashing the server request handler.

Important failure cases:

- Missing input video.
- Unknown model name.
- Missing ffmpeg.
- Missing Python dependency.
- CUDA unavailable or model load failure.
- Encoder failure.
- Alpha MOV conversion failure.

Alpha MOV conversion remains a soft error when WebM and preview MP4 succeeded. The UI can still use the preview, and export users still have the WebM alpha file.

## Testing

Add focused tests before implementation where practical:

- `pipeline.bgremove` chooses native backend by default.
- `ANIFORGE_BGREMOVE_BACKEND=external` routes to the old external worker path.
- Unknown model names fail fast.
- Worker result-line parsing still finds preview and outputs.
- Encoder command construction uses alpha-preserving VP9 decode for MOV conversion.

Add one opt-in integration test for real video processing. It should be skipped unless an environment variable such as `ANIFORGE_RUN_BGREMOVE_INTEGRATION=1` is set, because it needs GPU dependencies and model assets.

## Migration Plan

Implementation should be incremental:

1. Add the native package and subprocess worker.
2. Wire `pipeline.bgremove.run_bgremove()` to default to native.
3. Keep the external backend behind `ANIFORGE_BGREMOVE_BACKEND=external`.
4. Add tests for routing, validation, and output parsing.
5. Run one real short-video comparison against the existing external repo.
6. Update README and `.env.example` to mark `VIDEO_BG_REMOVAL_ROOT` as an optional legacy backend.

## Non-Goals

- No UI redesign.
- No ComfyUI node conversion for background removal in this change.
- No removal of existing `stage_bgremove()` response fields.
- No server-process torch imports.
- No automatic deletion of the external `videoBGremoval` repo.
