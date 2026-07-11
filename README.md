# AniForge

**Bring Any Character to Life**

Turn a single character standee (立绘) + action text into a Live2D-style pair of clips: a looping **idle** and a one-shot **action** you can click to play.

Example assets below come from a full **Run all** session  
`runs/b814053601c843cb998d5997c30df8f0` (nurse, 720×1280, seed 42, RMBG-2.0 HQ).  
Frames: idle @ **0.5s**, action @ **1.1s** (hands-on-hips hold).

---

## Pipeline at a glance

| Step | Stage | What it does | Main products |
|:----:|-------|--------------|---------------|
| 0 | **Input** | Character still | `input.png` |
| 1 | **Extract** | HMR pose from image | `extract_skel.png`, `extract_pose.npy` |
| 2 | **Idle skeleton** | Kimodo + Live2D breath shaping | `idle_skel.mp4`, `idle_guide.mp4`, `idle_seed*.npz` |
| 3 | **Action skeleton** | Kimodo from action prompt ± joint spring | `action_skel.mp4`, `action_guide.mp4`, `action_*seed*.npz` |
| 4 | **SCAIL idle** | Drive image with idle guide (cfg≈3) | `idle.mp4` |
| 5 | **SCAIL action** | Drive image with action guide | `action.mp4` |
| 6 | **BG remove** | RMBG-2.0 HQ default; gray preview + alpha | `*_nobg.mp4`, `*_nobg.webm`, `*_nobg_alpha.mov` |
| 7 | **Time overshoot** | Spring remap on **action** only | `action_timed.mp4`, `action_timed.webm` |
| 8 | **Preview** | UI: idle loops; **click** → action once → idle | Combined player |

### Example action prompt

```text
Slowly bring both hands to the waist and rest them firmly on the hips in a
relaxed hands-on-hips pose, then hold still. Smooth, gentle, soft motion,
not snappy. Both arms move together; clipboard can settle at the hip if
needed. Torso upright, hips and feet fixed, no stepping, no twist.
Mouth closed and still, lips sealed, silent, no talking.
```

---

## Step-by-step products (example run)

### 0 · Input

Character image uploaded as the session reference.

![input](docs/readme-assets/thumbs/01_input.png)

→ `input.png`

### 1 · Extract pose

HMR still skeleton for review (root pin by pose mode).

![extract](docs/readme-assets/thumbs/02_extract.png)

→ `extract_skel.png` · `extract_pose.npy`

### 2 · Idle skeleton

Kimodo idle motion + breath shaping; arms/legs locked.

![idle skeleton](docs/readme-assets/thumbs/03_idle_skel.png)

→ `idle_skel.mp4` · `idle_guide.mp4` · `idle_seed*.npz`

### 3 · Action skeleton

Kimodo action (+ optional joint overshoot). Frame at **1.1s** (hips hold).

![action skeleton](docs/readme-assets/thumbs/04_action_skel.png)

→ `action_skel.mp4` · `action_guide.mp4` · `action_joint_seed*.npz`

### 4 · SCAIL idle

Reference image driven by idle guide.

![idle scail](docs/readme-assets/thumbs/05_idle.png)

→ `idle.mp4`

### 5 · SCAIL action

Same character driven by action guide.

![action scail](docs/readme-assets/thumbs/06_action.png)

→ `action.mp4`

### 6 · Background remove

Default **RMBG-2.0 HQ**. Browser preview uses neutral gray; CapCut uses ProRes alpha.

| Idle no-bg | Action no-bg |
|:----------:|:------------:|
| ![idle nobg](docs/readme-assets/thumbs/07_idle_nobg.png) | ![action nobg](docs/readme-assets/thumbs/08_action_nobg.png) |

→ `idle_nobg.mp4` / `.webm` / `_alpha.mov` · `action_nobg.*`

### 7 · Time overshoot

Playback spring on the action (alpha preserved in webm; H.264 preview on gray).

![action timed](docs/readme-assets/thumbs/09_action_timed.png)

→ `action_timed.mp4` · `action_timed.webm`

### 8 · Preview (UI)

Idle loops by default. **Click the video** to play action once, then return to idle.

---

## Typical `runs/<run_id>/` layout

```text
input.png
extract_skel.png / extract_pose.npy
idle_skel.mp4 / idle_guide.mp4 / idle_seed*.npz
action_skel.mp4 / action_guide.mp4 / action_*seed*.npz
idle.mp4 / action.mp4
idle_nobg.mp4 + .webm + _alpha.mov
action_nobg.mp4 + .webm + _alpha.mov
action_timed.mp4 + .webm
meta.json
```

**CapCut:** import `*_nobg_alpha.mov` (ProRes 4444).  
**Browser:** H.264 with transparency flattened onto gray (same as bgremove preview).

Full-res frames also live under [`docs/readme-assets/`](docs/readme-assets/) (README embeds 280px thumbs).

---

## Quick start

### Prerequisites

1. **ComfyUI-scail** with Kimodo (SOMA) + SCAIL2 (`WanSCAILToVideo`), default URL `http://127.0.0.1:8188`
2. **ffmpeg** on PATH
3. **Python 3.10–3.12**
4. Optional: **videoBGremoval** sibling folder

### Configure paths

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `COMFYUI_SCAIL_ROOT` | ComfyUI root (`input/`, `output/`, `custom_nodes/`) |
| `VIDEO_BG_REMOVAL_ROOT` | videoBGremoval repo |
| `COMFY_PYTHON` | Optional Python with torch for subprocesses |
| `STANDEE_DIR` | Optional standee image folder for batch tools |

If unset, AniForge looks for `../ComfyUI-scail` / `../videoBGremoval`, else `./.comfy/…`.

### Install & run

```bash
pip install -r requirements.txt
python server/app.py
```

Open **http://127.0.0.1:8500**

| Tab | Use |
|-----|-----|
| **Run all** | Image + action prompt → full pipeline + shared click preview |
| **Step by step** | Extract → idle → action → SCAIL (editable prompts) → bg → time |

Windows: `start.bat` / `stop.bat` (port 8500).

---

## Models

```text
Image ──HMR──► extract
  idle text ──Kimodo──► idle skel ──SCAIL2──► idle.mp4 ──RMBG──► idle_nobg*
  action text ──Kimodo──► action skel (±joint) ──SCAIL2──► action.mp4 ──RMBG──► action_nobg*
                                                              └──time──► action_timed*
```

| Stage | Engine |
|-------|--------|
| Extract | HMR / seated extract |
| Motion | Kimodo (SOMA), idle ~2s breath |
| Drive | SCAIL-2 + LightX2V distill; default **cfg=3** |
| Matte | **RMBG-2.0 HQ** (RVM optional) |
| Time | Spring remap on action |

SCAIL text describes the **finished video**. Defaults: `/api/scail-defaults`.

---

## Status

MVP end-to-end. Click preview: loop idle / play action / return. Heavy soft-body secondary motion is out of scope for the skeleton path.
