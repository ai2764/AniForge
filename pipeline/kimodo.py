"""Submit a Kimodo text-to-motion graph and retrieve the NPZ."""
from __future__ import annotations
import uuid
from pathlib import Path
from .comfy import ComfyClient


def build_kimodo_graph(prompt, duration=3.0, seed=42,
                       model="Kimodo-SOMA-RP-v1", steps=50, prefix="mp_motion"):
    return {
        "1": {"class_type": "Kimodo_LoadModel", "inputs": {"model": model}},
        "2": {"class_type": "Kimodo_TextEncode",
              "inputs": {"model": ["1", 0], "prompt": prompt}},
        "3": {"class_type": "Kimodo_Sampler",
              "inputs": {"model": ["1", 0], "conditioning": ["2", 0],
                         "duration": float(duration), "seed": int(seed),
                         "num_samples": 1, "diffusion_steps": int(steps)}},
        "4": {"class_type": "Kimodo_PostProcess", "inputs": {"motion": ["3", 0]}},
        "5": {"class_type": "Kimodo_SaveNPZ",
              "inputs": {"motion": ["4", 0], "filename_prefix": prefix}},
    }


def generate_motion(client: ComfyClient, prompt, out_npz: Path, *,
                    duration=3.0, seed=42, model="Kimodo-SOMA-RP-v1", steps=50,
                    comfy_output: Path | None = None):
    from pipeline.paths import comfy_output_dir

    if comfy_output is None:
        comfy_output = comfy_output_dir()
    comfy_output = Path(comfy_output)
    graph = build_kimodo_graph(prompt, duration, seed, model, steps)
    pid = client.submit(graph, f"mp-kim-{uuid.uuid4().hex[:6]}")
    entry = client.wait(pid)
    if entry["status"]["status_str"] != "success":
        raise RuntimeError(f"Kimodo failed: {entry['status'].get('messages')}")
    # SaveNPZ writes under comfy output; resolve the newest matching file
    node_out = entry["outputs"].get("5", {})
    rel = (node_out.get("file_path") or node_out.get("text") or [None])
    rel = rel[0] if isinstance(rel, list) else rel
    src = (comfy_output / rel) if rel else _newest(comfy_output, "mp_motion")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_npz.write_bytes(Path(src).read_bytes())
    return out_npz


def _newest(root: Path, stem: str) -> Path:
    files = sorted(root.rglob(f"{stem}*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {stem}*.npz under {root}")
    return files[-1]
