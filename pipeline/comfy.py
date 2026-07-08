"""Minimal ComfyUI HTTP client: submit graphs, poll history, fetch outputs."""
from __future__ import annotations
import json, time, shutil, urllib.request, urllib.parse
from pathlib import Path


class ComfyClient:
    def __init__(self, base_url="http://127.0.0.1:8188", opener=urllib.request.urlopen):
        self.base = base_url.rstrip("/")
        self.opener = opener

    def _get(self, path):
        with self.opener(self.base + path, timeout=30) as r:
            return json.load(r)

    def _post(self, path, payload):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.opener(req, timeout=30) as r:
            return json.load(r)

    def object_info(self):
        return self._get("/object_info")

    def submit(self, graph, client_id):
        return self._post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]

    def wait(self, prompt_id, timeout=1500, interval=3):
        t0 = time.time()
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                return hist[prompt_id]
            if time.time() - t0 > timeout:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout}s")
            time.sleep(interval)

    def fetch_output(self, item, dest: Path):
        q = urllib.parse.urlencode({"filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.opener(f"{self.base}/view?{q}", timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest

    @staticmethod
    def stage_input(src: Path, name: str, input_dir: Path) -> str:
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, input_dir / name)
        return name


def first_output(entry, keys=("videos", "gifs", "images")):
    """Return the first output item dict from a history entry, or None."""
    for out in entry.get("outputs", {}).values():
        for k in keys:
            for item in out.get(k, []):
                return item
    return None
