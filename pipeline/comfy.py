"""Minimal ComfyUI HTTP client: submit graphs, poll history, fetch outputs, free VRAM."""
from __future__ import annotations
import json, time, shutil, urllib.request, urllib.parse
from pathlib import Path


class ComfyClient:
    def __init__(self, base_url="http://127.0.0.1:8188", opener=urllib.request.urlopen):
        self.base = base_url.rstrip("/")
        self.opener = opener

    def _get(self, path, timeout=30):
        with self.opener(self.base + path, timeout=timeout) as r:
            return json.load(r)

    def _post(self, path, payload, timeout=30):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload if payload is not None else {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.opener(req, timeout=timeout) as r:
            raw = r.read()
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {"raw": raw.decode("utf-8", errors="replace")}

    def object_info(self):
        return self._get("/object_info")

    def system_stats(self) -> dict:
        try:
            return self._get("/system_stats", timeout=15)
        except Exception as exc:
            return {"error": str(exc)}

    def vram_free_bytes(self) -> int | None:
        """Return torch_vram_free for first CUDA device, or None."""
        stats = self.system_stats()
        devices = stats.get("devices") or []
        if not devices:
            return None
        d0 = devices[0]
        for key in ("torch_vram_free", "vram_free"):
            if key in d0 and d0[key] is not None:
                return int(d0[key])
        return None

    def interrupt(self) -> dict:
        """Stop the currently running ComfyUI prompt (global interrupt)."""
        try:
            return self._post("/interrupt", {}, timeout=15)
        except Exception as exc:
            return {"error": str(exc)}

    def clear_queue(self) -> dict:
        try:
            return self._post("/queue", {"clear": True}, timeout=15)
        except Exception as exc:
            return {"error": str(exc)}

    def free(self, *, unload_models: bool = True, free_memory: bool = True) -> dict:
        """Queue a free request (processed by Comfy worker when idle between jobs).

        Note: ComfyUI only *flags* unload; the worker applies it after the current
        prompt finishes or on the next idle poll (~1s). Prefer ``free_vram()`` when
        you need to wait until memory actually drops.
        """
        try:
            return self._post(
                "/free",
                {"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
                timeout=60,
            )
        except Exception as exc:
            return {"error": str(exc)}

    def queue_running(self) -> bool:
        """True if Comfy has a running or pending prompt."""
        try:
            q = self._get("/queue", timeout=10)
            running = q.get("queue_running") or []
            pending = q.get("queue_pending") or []
            return bool(running or pending)
        except Exception:
            return False

    def free_vram(
        self,
        *,
        interrupt: bool = False,
        clear_queue: bool = False,
        wait_s: float = 45.0,
        min_free_gb: float = 8.0,
    ) -> dict:
        """Aggressively reclaim ComfyUI VRAM and wait until free memory rises.

        Important: Comfy ``POST /free`` only *sets a flag*. The prompt worker
        unloads models **after the current execute() returns**. If SCAIL is mid
        graph, you must ``interrupt`` first, wait for the queue to go idle, then
        free — otherwise VRAM stays full.

        - optional interrupt of running job
        - clear pending queue
        - wait until queue idle
        - POST /free unload_models + free_memory (repeated)
        - poll /system_stats until free >= min_free_gb or timeout
        """
        report: dict = {"steps": []}
        before = self.vram_free_bytes()
        report["vram_free_before"] = before

        if interrupt:
            report["steps"].append({"interrupt": self.interrupt()})
            time.sleep(0.5)
            # second interrupt — first can be ignored mid-node
            report["steps"].append({"interrupt2": self.interrupt()})
        if clear_queue:
            report["steps"].append({"clear_queue": self.clear_queue()})

        # Wait for queue to go idle so free flags can actually run.
        t_idle = time.time()
        idle_deadline = min(30.0, wait_s)
        while time.time() - t_idle < idle_deadline:
            if not self.queue_running():
                break
            if interrupt:
                self.interrupt()
            time.sleep(1.0)
        report["queue_idle"] = not self.queue_running()

        # Flag unload; worker applies on next idle poll (~1s)
        report["steps"].append({"free1": self.free(unload_models=True, free_memory=True)})
        time.sleep(2.0)
        report["steps"].append({"free2": self.free(unload_models=True, free_memory=True)})

        target = int(min_free_gb * (1024 ** 3))
        t0 = time.time()
        last = before
        while time.time() - t0 < wait_s:
            time.sleep(1.5)
            # Re-flag so idle worker keeps unloading
            self.free(unload_models=True, free_memory=True)
            last = self.vram_free_bytes()
            if last is not None and last >= target:
                break

        report["vram_free_after"] = last
        report["ok"] = last is not None and last >= target
        report["waited_s"] = round(time.time() - t0, 1)
        if last is not None:
            report["vram_free_after_gb"] = round(last / (1024 ** 3), 2)
        if before is not None:
            report["vram_free_before_gb"] = round(before / (1024 ** 3), 2)
        return report

    def submit(self, graph, client_id):
        return self._post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]

    def wait(self, prompt_id, timeout=7200, interval=3):
        """Poll history until done. Default timeout 2h (SCAIL under VRAM pressure)."""
        t0 = time.time()
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                return hist[prompt_id]
            if time.time() - t0 > timeout:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout}s")
            time.sleep(interval)

    def fetch_output(self, item, dest: Path):
        q = urllib.parse.urlencode({
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
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
