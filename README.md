# Motion Portrait

Motion Portrait turns a single character image and an action prompt into a live2d-style animated portrait: a looping idle animation plus a one-shot action triggered by clicking the preview.

A single generate run produces two mp4 files:
- `idle.mp4` — a looping idle animation
- `action.mp4` — a one-shot action clip that plays on click and returns to idle

## Prerequisites

Before running, ensure you have:

1. **ComfyUI backend** running at `http://127.0.0.1:8188` with:
   - `ComfyUI-Kimodo` node pack (SOMA checkpoint) installed
   - SCAIL2 nodes (`WanSCAILToVideo`) installed
   
   See [docs/motion-source.md](docs/motion-source.md) for Kimodo and Llama-3 setup instructions and dependency notes.

2. **ffmpeg** on your system PATH

3. **Python 3.12** (note: Python 3.13+ removed the `cgi` module, which this server uses)

## Installation

```bash
pip install -r requirements.txt
```

## Running

Start the server:

```bash
python server/app.py
```

The server runs on port 8500 by default. Open your browser to:

```
http://127.0.0.1:8500
```

## Usage

1. **Upload a character image** — drag and drop or click to select
2. **Enter an action prompt** — describe the motion you want (required)
3. **Enter an idle prompt** (optional) — describe a relaxed idle animation
   - If left blank, uses a default idle: calm breathing, gentle side-to-side sway, small head/arm movement, feet planted
4. **Overshoot options** (optional, both can be selected and stack):
   - **Joint** — applies damped-spring overshoot to the skeleton in joint-space on the action only
   - **Time** — applies overshoot to the playback timing of the rendered action video
5. **Click Generate** and wait for both clips to render

The preview page loops the idle animation. Click the preview to play the action clip; it will return to idle when finished.

## How It Works

The pipeline orchestrates the following steps:

1. **Text-to-motion**: Kimodo (SOMA 30-joint skeleton) converts your prompts into joint positions
2. **Joint-space processing**: optional damped-spring overshoot applied to the action skeleton
3. **Frame padding**: skeletons padded to 9:16 aspect ratio (portrait mode)
4. **Character driving**: SCAIL2 applies the skeleton guide to your character image
5. **Time-space processing**: optional playback-time overshoot applied to the rendered action video
6. **Output**: `idle.mp4` (loops) and `action.mp4` (plays once on click)

## Status & Known Limits

**MVP (Minimal Viable Product)** — the core pipeline works end-to-end.

Known limitations:
- **Idle-to-action transitions** use a simple loop + click + return (seamless anchor closure is future work)
- **Secondary soft-body motion** (dramatic hair/skirt/chest bounce in live2d) is not produced by the skeleton pipeline and is out of scope for this version
