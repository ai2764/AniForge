"""Backend server: stdlib http.server app exposing POST /generate plus
static serving of the web/ frontend and runs/ generated videos.

Independent of any other project - only pipeline.* and stdlib are imported.
"""
from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pipeline.comfy import ComfyClient
from pipeline.generate import generate

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RUNS_DIR = REPO_ROOT / "runs"

REQUIRED_NODES = ("Kimodo_Sampler", "WanSCAILToVideo")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no I/O)
# ---------------------------------------------------------------------------

def parse_generate_form(fields: dict) -> dict:
    """fields: dict with "action_prompt" (str), "idle_prompt" (str),
    "overshoot" (list[str]). Returns the normalized request dict."""
    action_prompt = fields.get("action_prompt") or ""
    idle_prompt = fields.get("idle_prompt") or None
    if isinstance(idle_prompt, str) and not idle_prompt.strip():
        idle_prompt = None
    overshoot = set(fields.get("overshoot") or [])
    return {
        "action_prompt": action_prompt,
        "idle_prompt": idle_prompt,
        "overshoot": overshoot,
    }


def required_nodes_present(object_info: dict) -> bool:
    """True iff every node class in REQUIRED_NODES is a key of object_info."""
    return all(name in object_info for name in REQUIRED_NODES)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _content_type_for(path: Path) -> str:
    ctype = CONTENT_TYPES.get(path.suffix.lower())
    if ctype:
        return ctype
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


class Handler(BaseHTTPRequestHandler):
    server_version = "MotionPortrait/1.0"

    # -- helpers --------------------------------------------------------
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_json(404, {"error": f"not found: {path.name}"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _content_type_for(path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _resolve_static(self, base_dir: Path, rel: str) -> Path | None:
        """Resolve rel (already stripped of its leading segment) against
        base_dir, refusing to escape it (path traversal guard)."""
        rel = rel.lstrip("/")
        if not rel:
            return None
        candidate = (base_dir / rel).resolve()
        base_resolved = base_dir.resolve()
        try:
            candidate.relative_to(base_resolved)
        except ValueError:
            return None
        return candidate

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_file(WEB_DIR / "index.html")
            return
        if path.startswith("/web/"):
            target = self._resolve_static(WEB_DIR, path[len("/web/"):])
            if target is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_file(target)
            return
        if path.startswith("/runs/"):
            target = self._resolve_static(RUNS_DIR, path[len("/runs/"):])
            if target is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_file(target)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        self._handle_generate()

    def _handle_generate(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._send_json(400, {"error": "expected multipart/form-data"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )

            image_item = form["image"] if "image" in form else None
            if image_item is None or not getattr(image_item, "filename", None):
                self._send_json(400, {"error": "missing 'image' upload"})
                return

            fields = {
                "action_prompt": form.getvalue("action_prompt", ""),
                "idle_prompt": form.getvalue("idle_prompt", ""),
                "overshoot": form.getlist("overshoot"),
            }
            parsed = parse_generate_form(fields)

            client = ComfyClient()
            try:
                object_info = client.object_info()
            except Exception as exc:
                self._send_json(503, {"error": f"ComfyUI unreachable: {exc}"})
                return

            if not required_nodes_present(object_info):
                missing = [n for n in REQUIRED_NODES if n not in object_info]
                self._send_json(503, {
                    "error": "ComfyUI is missing required custom nodes",
                    "missing_nodes": missing,
                })
                return

            run_id = uuid.uuid4().hex
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(image_item.filename).suffix or ".png"
            image_path = run_dir / f"input{ext}"
            image_path.write_bytes(image_item.file.read())

            result = generate(
                image_path,
                parsed["action_prompt"],
                parsed["idle_prompt"],
                parsed["overshoot"],
                run_dir=run_dir,
                client=client,
            )

            def to_url(p):
                if p is None:
                    return None
                rel = Path(p).resolve().relative_to(REPO_ROOT.resolve())
                return "/" + rel.as_posix()

            self._send_json(200, {
                "idle": to_url(result.get("idle")),
                "action": to_url(result.get("action")),
                "errors": result.get("errors", {}),
            })
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    # quieter default logging (still useful, but avoid noisy stderr in tests)
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - %s\n" % (self.address_string(), fmt % args))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Motion Portrait backend server")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args(argv)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = HTTPServer(("", args.port), Handler)
    print(f"Serving on http://127.0.0.1:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
