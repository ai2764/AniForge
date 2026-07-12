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

# Allow `python server/app.py` from the repo root by putting the repo root
# (which contains the `pipeline/` package) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.comfy import ComfyClient  # noqa: E402
from pipeline.generate import generate  # noqa: E402
from pipeline.seated.generate_anchored import generate_anchored  # noqa: E402
from pipeline.stages import (  # noqa: E402
    create_session,
    stage_action,
    stage_bgremove,
    stage_extract,
    stage_idle,
    stage_joint_overshoot,
    stage_scail,
    stage_time_overshoot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RUNS_DIR = REPO_ROOT / "runs"

REQUIRED_NODES = ("Kimodo_Sampler", "WanSCAILToVideo")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no I/O)
# ---------------------------------------------------------------------------

POSE_MODES = ("standing", "sitting", "lying")


def parse_generate_form(fields: dict) -> dict:
    """fields: dict with "action_prompt" (str), "idle_prompt" (str),
    "overshoot" (list[str]), "seed" (str, optional), "pose_mode" (str),
    "scale" (str/float, optional). Returns the normalized request dict.
    Blank seed -> None (random). pose_mode defaults to standing.
    scale is 0.25–1.0 (fraction of max resolution, same aspect as image)."""
    action_prompt = fields.get("action_prompt") or ""
    idle_prompt = fields.get("idle_prompt") or None
    if isinstance(idle_prompt, str) and not idle_prompt.strip():
        idle_prompt = None
    overshoot = set(fields.get("overshoot") or [])
    raw_seed = fields.get("seed")
    try:
        seed = int(raw_seed) if raw_seed not in (None, "") else None
    except (TypeError, ValueError):
        seed = None
    pose_mode = (fields.get("pose_mode") or "standing").strip().lower()
    if pose_mode not in POSE_MODES:
        pose_mode = "standing"
    try:
        scale = float(fields.get("scale") if fields.get("scale") not in (None, "") else 1.0)
    except (TypeError, ValueError):
        scale = 1.0
    scale = max(0.25, min(1.0, scale))
    return {
        "action_prompt": action_prompt,
        "idle_prompt": idle_prompt,
        "overshoot": overshoot,
        "seed": seed,
        "pose_mode": pose_mode,
        "scale": scale,
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
    server_version = "AniForge/1.0"

    # -- helpers --------------------------------------------------------
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        """Serve a file; support HTTP Range for HTML5 video seek/play."""
        if not path.is_file():
            self._send_json(404, {"error": f"not found: {path.name}"})
            return
        data = path.read_bytes()
        ctype = _content_type_for(path)
        total = len(data)
        range_hdr = self.headers.get("Range") or self.headers.get("range")
        # Browser <video> often sends Range; without 206 some clients mark MEDIA_ERR.
        if range_hdr and range_hdr.startswith("bytes=") and total > 0:
            try:
                spec = range_hdr[len("bytes="):].strip()
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else (total - 1)
                start = max(0, min(start, total - 1))
                end = max(start, min(end, total - 1))
                chunk = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(chunk)
                return
            except (ValueError, TypeError):
                pass
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "no-cache")
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
        if path == "/api/scail-defaults":
            self._handle_scail_defaults()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/generate":
            self._handle_generate()
            return
        if path == "/session":
            self._handle_session_create()
            return
        if path == "/session/extract":
            self._handle_session_extract()
            return
        if path == "/session/idle":
            self._handle_session_idle()
            return
        if path == "/session/action":
            self._handle_session_action()
            return
        if path == "/session/scail":
            self._handle_session_scail()
            return
        if path == "/session/joint-overshoot":
            self._handle_session_joint_overshoot()
            return
        if path == "/session/time-overshoot":
            self._handle_session_time_overshoot()
            return
        if path == "/session/bgremove":
            self._handle_session_bgremove()
            return
        # legacy alias
        if path == "/session/overshoot":
            self._handle_session_time_overshoot()
            return
        self._send_json(404, {"error": "not found"})

    def _read_multipart(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return None, "expected multipart/form-data"
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
        )
        return form, None

    def _handle_session_create(self):
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            image_item = form["image"] if "image" in form else None
            if image_item is None or not getattr(image_item, "filename", None):
                self._send_json(400, {"error": "missing 'image' upload"})
                return
            fields = {
                "pose_mode": form.getvalue("pose_mode", "standing"),
                "seed": form.getvalue("seed", ""),
                "scale": form.getvalue("scale", "1"),
                "action_prompt": "x",  # parse_generate_form requires key
            }
            parsed = parse_generate_form(fields)
            data = create_session(
                image_item.file.read(),
                image_item.filename,
                pose_mode=parsed["pose_mode"],
                seed=parsed["seed"],
                scale=parsed["scale"],
                runs_dir=RUNS_DIR,
            )
            self._send_json(200, data)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_extract(self):
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip()
            if not run_id or not (RUNS_DIR / run_id).is_dir():
                self._send_json(400, {"error": "invalid run_id"})
                return
            result = stage_extract(
                run_id,
                pose_mode=form.getvalue("pose_mode", "") or None,
                runs_dir=RUNS_DIR,
            )
            status = 200 if not result.get("errors") else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_idle(self):
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip()
            if not run_id or not (RUNS_DIR / run_id).is_dir():
                self._send_json(400, {"error": "invalid run_id"})
                return
            # idle = Kimodo skeleton only (no Comfy required)
            raw_keep = form.getvalue("idle_motion_keep", "")
            try:
                idle_keep = float(raw_keep) if raw_keep not in (None, "") else None
            except (TypeError, ValueError):
                idle_keep = None
            result = stage_idle(
                run_id,
                idle_prompt=form.getvalue("idle_prompt", "") or None,
                idle_motion_keep=idle_keep,
                runs_dir=RUNS_DIR,
            )
            status = 200 if not result.get("errors") else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_action(self):
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip()
            if not run_id or not (RUNS_DIR / run_id).is_dir():
                self._send_json(400, {"error": "invalid run_id"})
                return
            # action = Kimodo skeleton only (no Comfy, no overshoot)
            raw_keep = form.getvalue("action_motion_keep", "")
            try:
                action_keep = float(raw_keep) if raw_keep not in (None, "") else None
            except (TypeError, ValueError):
                action_keep = None
            raw_dur = form.getvalue("action_duration", "")
            try:
                action_dur = float(raw_dur) if raw_dur not in (None, "") else None
            except (TypeError, ValueError):
                action_dur = None
            result = stage_action(
                run_id,
                action_prompt=form.getvalue("action_prompt", ""),
                action_motion_keep=action_keep,
                action_duration=action_dur,
                runs_dir=RUNS_DIR,
            )
            status = 200 if not result.get("errors") else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_scail_defaults(self):
        """Product defaults for SCAIL2 positive/negative text fields."""
        try:
            from urllib.parse import parse_qs, urlparse

            from pipeline.generate import (
                SCAIL_ACTION_POSITIVE,
                SCAIL_IDLE_POSITIVE,
                SCAIL_NEGATIVE,
                build_scail_positive,
            )

            qs = parse_qs(urlparse(self.path).query)
            action_prompt = (qs.get("action_prompt") or [""])[0]
            self._send_json(
                200,
                {
                    "idle_positive": SCAIL_IDLE_POSITIVE,
                    "action_positive": build_scail_positive(
                        "action", action_prompt
                    ),
                    "action_positive_base": SCAIL_ACTION_POSITIVE,
                    "negative": SCAIL_NEGATIVE,
                },
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_scail(self):
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip()
            if not run_id or not (RUNS_DIR / run_id).is_dir():
                self._send_json(400, {"error": "invalid run_id"})
                return
            client = ComfyClient()
            try:
                client.object_info()
            except Exception as exc:
                self._send_json(503, {"error": f"ComfyUI unreachable: {exc}"})
                return
            which = (form.getvalue("which") or "both").strip().lower()
            result = stage_scail(
                run_id,
                which=which,
                runs_dir=RUNS_DIR,
                client=client,
                scale=form.getvalue("scale", "") or None,
                positive_idle=form.getvalue("scail_idle_positive") or None,
                positive_action=form.getvalue("scail_action_positive") or None,
                negative=form.getvalue("scail_negative") or None,
            )
            status = 200 if not result.get("errors") else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_joint_overshoot(self):
        """Spring overshoot on action skeleton (before SCAIL). No Comfy."""
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip()
            if not run_id or not (RUNS_DIR / run_id).is_dir():
                self._send_json(400, {"error": "invalid run_id"})
                return
            # mode: preview (render overshot preview) | carry | uncarry (guide only)
            mode = (str(form.getvalue("mode", "preview")).strip().lower() or "preview")

            def _optfloat(name):
                raw = form.getvalue(name, "")
                try:
                    return float(raw) if str(raw).strip() != "" else None
                except (TypeError, ValueError):
                    return None

            result = stage_joint_overshoot(
                run_id,
                mode=mode,
                omega=_optfloat("joint_omega"),
                zeta=_optfloat("joint_zeta"),
                soft=_optfloat("joint_soft"),
                runs_dir=RUNS_DIR,
            )
            status = 200 if not result.get("errors") else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_time_overshoot(self):
        """Time-remap action video: session action/nobg and/or uploaded video."""
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip() or None
            upload_bytes = None
            upload_filename = "upload.mp4"
            if "video" in form:
                item = form["video"]
                if getattr(item, "filename", None):
                    upload_filename = item.filename or upload_filename
                    upload_bytes = item.file.read()
            if not run_id and upload_bytes is None:
                self._send_json(
                    400,
                    {"error": "upload a video or create a session first"},
                )
                return
            result = stage_time_overshoot(
                run_id,
                runs_dir=RUNS_DIR,
                upload_bytes=upload_bytes,
                upload_filename=upload_filename,
            )
            # Success if any timed/action artifact exists; soft warnings stay in payload.
            has_out = any(
                result.get(k)
                for k in (
                    "action",
                    "action_timed",
                    "action_nobg",
                    "action_timed_webm",
                    "action_nobg_webm",
                    "action_nobg_alpha",
                )
            )
            hard = result.get("errors") or {}
            # Soft keys that must not fail the HTTP status alone
            soft_keys = {"time_alpha"}
            hard_only = {k: v for k, v in hard.items() if k not in soft_keys}
            if has_out and hard:
                # Move soft / residual messages to warnings for the client.
                result["warnings"] = dict(hard)
                if not hard_only:
                    result["errors"] = {}
            status = 200 if has_out or not hard_only else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_session_bgremove(self):
        """Video BG removal — optional session videos and/or uploaded video (no prereqs)."""
        try:
            form, err = self._read_multipart()
            if err:
                self._send_json(400, {"error": err})
                return
            run_id = (form.getvalue("run_id") or "").strip() or None
            which = (form.getvalue("which") or "both").strip().lower()
            model = (form.getvalue("model") or "RMBG-2.0 HQ").strip()
            upload_bytes = None
            upload_filename = "upload.mp4"
            if "video" in form:
                item = form["video"]
                if getattr(item, "filename", None):
                    upload_filename = item.filename or upload_filename
                    upload_bytes = item.file.read()
            result = stage_bgremove(
                run_id,
                which=which,
                model=model,
                upload_bytes=upload_bytes,
                upload_filename=upload_filename,
                runs_dir=RUNS_DIR,
            )
            has_out = any(
                result.get(k)
                for k in (
                    "idle_nobg",
                    "action_nobg",
                    "upload_nobg",
                    "idle_nobg_webm",
                    "action_nobg_webm",
                    "upload_nobg_webm",
                    "idle_nobg_alpha",
                    "action_nobg_alpha",
                    "upload_nobg_alpha",
                )
            )
            errs = result.get("errors") or {}
            status = 200 if has_out or not errs else 500
            self._send_json(status, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

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
                "seed": form.getvalue("seed", ""),
                "pose_mode": form.getvalue("pose_mode", "standing"),
                "scale": form.getvalue("scale", "1"),
            }
            parsed = parse_generate_form(fields)

            client = ComfyClient()
            try:
                object_info = client.object_info()
            except Exception as exc:
                self._send_json(503, {"error": f"ComfyUI unreachable: {exc}"})
                return

            # Standing path needs Kimodo Comfy nodes; sitting/lying use standalone Kimodo
            # but still need SCAIL in ComfyUI.
            if parsed["pose_mode"] == "standing":
                if not required_nodes_present(object_info):
                    missing = [n for n in REQUIRED_NODES if n not in object_info]
                    self._send_json(503, {
                        "error": "ComfyUI is missing required custom nodes",
                        "missing_nodes": missing,
                    })
                    return
            elif "WanSCAILToVideo" not in object_info:
                self._send_json(503, {
                    "error": "ComfyUI is missing SCAIL nodes required for sitting/lying",
                    "missing_nodes": ["WanSCAILToVideo"],
                })
                return

            run_id = uuid.uuid4().hex
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(image_item.filename).suffix or ".png"
            image_path = run_dir / f"input{ext}"
            image_path.write_bytes(image_item.file.read())

            if parsed["pose_mode"] == "standing":
                result = generate(
                    image_path,
                    parsed["action_prompt"],
                    parsed["idle_prompt"],
                    parsed["overshoot"],
                    run_dir=run_dir,
                    client=client,
                    seed=parsed["seed"],
                    scale=parsed["scale"],
                )
            else:
                result = generate_anchored(
                    image_path,
                    parsed["action_prompt"],
                    parsed["idle_prompt"],
                    parsed["overshoot"],
                    run_dir=run_dir,
                    client=client,
                    pose_mode=parsed["pose_mode"],
                    seed=parsed["seed"],
                    scale=parsed["scale"],
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
                "seed": result.get("seed"),
                "pose_mode": parsed["pose_mode"],
                "scale": result.get("scale", parsed["scale"]),
                "size": result.get("size"),
            })
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    # quieter default logging (still useful, but avoid noisy stderr in tests)
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - %s\n" % (self.address_string(), fmt % args))


def main(argv=None):
    parser = argparse.ArgumentParser(description="AniForge backend server")
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
