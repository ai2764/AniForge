"""Submit a scail2 character-drive graph and retrieve the output video."""
from __future__ import annotations
import copy
import json
import uuid
from pathlib import Path
from .comfy import ComfyClient, first_output

ASSET_TEMPLATE = Path(__file__).parent / "assets" / "scail2_video.api.json"


def build_scail_graph(template, guide_name, ref_name, width, height, length,
                       pose_strength, seed, steps, prefix, positive,
                       negative=None, cfg=None) -> dict:
    api = copy.deepcopy(template)
    api["5"]["inputs"]["text"] = positive
    if negative is not None:
        api["6"]["inputs"]["text"] = negative
    api["9"]["inputs"]["image"] = ref_name
    api["11"]["inputs"]["file"] = guide_name
    api["13"]["inputs"]["width"] = int(width)
    api["13"]["inputs"]["height"] = int(height)
    api["13"]["inputs"]["length"] = int(length)
    api["13"]["inputs"]["pose_strength"] = float(pose_strength)
    api["14"]["inputs"]["seed"] = int(seed)
    api["14"]["inputs"]["steps"] = int(steps)
    if cfg is not None:
        # cfg>1 re-enables the negative prompt (distill model ships cfg=1, which
        # disables it); needed so anti-lip-motion negatives actually suppress the
        # mouth. cfg~3 seals the mouth at full res with no visible quality loss.
        api["14"]["inputs"]["cfg"] = float(cfg)
    api["17"]["inputs"]["filename_prefix"] = prefix
    return api


def drive_character(client: ComfyClient, guide_mp4: Path, ref_image: Path, out_mp4: Path, *,
                     width=480, height=832, length, pose_strength=0.9, seed=42, steps=10,
                     cfg=3.0, prefix="mp_body", positive, negative=None,
                     comfy_input=Path("C:/Users/AIBOX/dev/ComfyUI-scail/input"),
                     template_path=ASSET_TEMPLATE) -> Path:
    guide_mp4 = Path(guide_mp4)
    ref_image = Path(ref_image)
    guide_name = ComfyClient.stage_input(guide_mp4, guide_mp4.name, comfy_input)
    ref_name = ComfyClient.stage_input(ref_image, ref_image.name, comfy_input)

    if negative is None:
        from pipeline.generate import SCAIL_NEGATIVE
        negative = SCAIL_NEGATIVE

    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    graph = build_scail_graph(template, guide_name, ref_name, width, height, length,
                               pose_strength, seed, steps, prefix, positive,
                               negative=negative, cfg=cfg)

    # Light pre-free only — do not block tens of seconds before submit.
    fre = client.free_vram(
        interrupt=False, clear_queue=False, wait_s=8, min_free_gb=4.0
    )
    print(f"[scail] free_vram before: {fre}", flush=True)

    pid = client.submit(graph, f"mp-scail-{uuid.uuid4().hex[:6]}")
    entry = client.wait(pid)

    if entry["status"]["status_str"] != "success":
        # Reclaim on failure without delaying the error path too long.
        client.free_vram(interrupt=False, clear_queue=False, wait_s=3, min_free_gb=0)
        raise RuntimeError(f"scail2 drive failed: {entry['status'].get('messages')}")

    item = first_output(entry)
    if item is None:
        client.free_vram(interrupt=False, clear_queue=False, wait_s=3, min_free_gb=0)
        raise RuntimeError("scail2 drive produced no output")

    # Fetch video first so the UI can show results; free VRAM after (non-blocking).
    out = client.fetch_output(item, out_mp4)
    client.free_vram(interrupt=False, clear_queue=False, wait_s=3, min_free_gb=0)
    return out
