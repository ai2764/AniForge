"""Staged generation: extract skeleton → idle → action, with previews between steps.

State lives in run_dir/meta.json + artifacts. Used by the stepwise UI.
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import numpy as np

from pipeline.comfy import ComfyClient
from pipeline.generate import (
    ACTION_DURATION_SEC,
    ACTION_MOTION_KEEP,
    DEFAULT_IDLE_PROMPT,
    FPS,
    IDLE_DURATION_SEC,
    IDLE_MOTION_KEEP,
    JOINT_SPRING,
    SCAIL_NEGATIVE,
    build_scail_positive,
    TIME_SPRING,
    _output_size,
    _pad_to_aspect,
    align_4k1,
    align_motion_to_base_pose,
    close_lower_body_action_requested,
    dampen_idle_joints,
    prepare_idle_source_motion,
    plan_steps,
    lower_body_action_requested,
    sanitize_action,
    shape_live2d_idle,
)
from pipeline.scail import drive_character
from pipeline.seated.generate_anchored import (
    HERE as SEATED_HERE,
    _run_subprocess,
    render_smplx_guide,
    skeleton_camera_from_joints,
)
from pipeline.skeleton_spring import spring_follow
from pipeline.spring_time_remap import time_remap_file

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
META_NAME = "meta.json"
POSE_MODES = ("standing", "sitting", "lying")


def _load_meta(run_dir: Path) -> dict:
    p = run_dir / META_NAME
    if not p.is_file():
        raise FileNotFoundError(f"missing session meta: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _normalize_pose_mode(value: str | None, default: str = "standing") -> str:
    mode = (value or default or "standing").strip().lower()
    if mode not in POSE_MODES:
        fallback = (default or "standing").strip().lower()
        return fallback if fallback in POSE_MODES else "standing"
    return mode


def _rel_url(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return str(resolved)
    return "/" + rel.as_posix()


def _find_image(run_dir: Path) -> Path:
    for p in sorted(run_dir.glob("input.*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            return p
    raise FileNotFoundError(f"no input image in {run_dir}")


def create_session(
    image_bytes: bytes,
    filename: str,
    *,
    pose_mode: str = "standing",
    seed: int | None = None,
    scale: float = 1.0,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    pose_mode = _normalize_pose_mode(pose_mode)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    try:
        scale = max(0.25, min(1.0, float(scale)))
    except (TypeError, ValueError):
        scale = 1.0

    run_id = uuid.uuid4().hex
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix or ".png"
    image_path = run_dir / f"input{ext}"
    image_path.write_bytes(image_bytes)

    out_w, out_h = _output_size(image_path, scale=scale)
    meta = {
        "run_id": run_id,
        "pose_mode": pose_mode,
        "seed": seed,
        "scale": scale,
        "size": [out_w, out_h],
        "image": image_path.name,
        "step": "created",
        "extracted": False,
        "idle_done": False,      # idle skeleton motion ready
        "action_done": False,    # action skeleton motion ready
        "idle_scail_done": False,
        "action_scail_done": False,
        "scail_done": False,     # both SCAIL character videos ready
    }
    _save_meta(run_dir, meta)
    return {
        "run_id": run_id,
        "pose_mode": pose_mode,
        "seed": seed,
        "scale": scale,
        "size": [out_w, out_h],
        "image": _rel_url(image_path),
    }


def stage_extract(
    run_id: str,
    *,
    runs_dir: Path = RUNS_DIR,
    n_frames: int = 90,
    pose_mode: str | None = None,
) -> dict:
    """All pose modes: HMR → hips-only constraint → static skeleton preview."""
    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)
    image = _find_image(run_dir)
    old_pose_mode = meta.get("pose_mode", "standing")
    requested_pose_mode = (pose_mode or old_pose_mode or "standing").strip().lower()
    if requested_pose_mode not in ("standing", "sitting", "lying"):
        requested_pose_mode = (
            old_pose_mode
            if old_pose_mode in ("standing", "sitting", "lying")
            else "standing"
        )
    pose_changed = requested_pose_mode != old_pose_mode
    if pose_changed:
        meta["pose_mode"] = requested_pose_mode
        meta["extracted"] = False
        meta["idle_done"] = False
        meta["action_done"] = False
        meta["idle_scail_done"] = False
        meta["action_scail_done"] = False
        meta["scail_done"] = False
    pose_mode = requested_pose_mode
    out: dict = {
        "run_id": run_id,
        "pose_mode": pose_mode,
        "pose_changed": pose_changed,
        "errors": {},
    }

    constraint_path = run_dir / "constraint.json"
    try:
        _run_subprocess("phase1_extract", [
            str(SEATED_HERE / "phase1_extract.py"),
            str(image),
            str(constraint_path),
            str(n_frames),
            pose_mode,  # standing|sitting|lying → hips-only pin
        ])
    except Exception as exc:
        out["errors"]["extract"] = str(exc)
        return out

    # FK preview via comfy-scail env (has kimodo + torch)
    try:
        cons = json.loads(constraint_path.read_text(encoding="utf-8"))
        c0 = cons[0]
        # Still only (1 frame → png + short h264 for compatibility)
        skel = run_dir / "extract_skel.mp4"
        _run_subprocess("render_constraint_skel", [
            str(SEATED_HERE / "render_constraint_skel.py"),
            str(constraint_path),
            str(skel),
            "1",
        ])
        meta["extracted"] = True
        meta["step"] = "extracted"
        meta["constraint_joints"] = c0.get("joint_names")
        _save_meta(run_dir, meta)
        png = skel.with_suffix(".png")
        out.update({
            "skipped": False,
            # UI uses PNG still for extract; mp4 kept for SCAIL tooling if needed
            "skeleton": None,
            "skeleton_png": _rel_url(png) if png.is_file() else _rel_url(skel),
            "constraint_joints": c0.get("joint_names"),
            "seed": meta["seed"],
        })
    except Exception as exc:
        if constraint_path.is_file():
            meta["extracted"] = True
            meta["step"] = "extracted"
            _save_meta(run_dir, meta)
            out["constraint"] = _rel_url(constraint_path)
            out["warning"] = f"preview render failed: {exc}"
            # still allow proceeding to idle
        else:
            out["errors"]["extract_preview"] = str(exc)
    return out


def _parse_duration(value, default: float, lo: float = 1.0, hi: float = 5.0) -> float:
    try:
        d = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        d = float(default)
    return max(lo, min(hi, d))


def _kimodo_job(
    run_dir: Path,
    *,
    seed: int,
    name: str,
    prompt: str,
    pose_mode: str,
    duration: float | None = None,
) -> dict:
    """Build a Kimodo job dict.

    Standing: free text-to-motion (no pose pin).
    Sitting/lying **idle**: hips-only pin (keeps seat/lie root while calm).
    Sitting/lying **action**: free Kimodo (so limbs can swing large); post-process
    re-sticks start pose + locks lower body. Pinning during action killed amplitude.
    """
    if duration is None:
        duration = IDLE_DURATION_SEC if name == "idle" else ACTION_DURATION_SEC
    duration = _parse_duration(duration, ACTION_DURATION_SEC if name == "action" else IDLE_DURATION_SEC)
    job = {
        "outdir": str(run_dir),
        "seed": seed,
        "duration": float(duration),
        "steps": 100,
        "jobs": [{"name": name, "prompt": prompt}],
    }
    pose_mode = (pose_mode or "standing").strip().lower()
    use_pin = pose_mode in ("sitting", "lying") and name == "idle"
    if use_pin:
        constraint = run_dir / "constraint.json"
        if not constraint.is_file():
            raise FileNotFoundError("missing constraint.json — run extract first")
        job["constraint_json"] = str(constraint)
    else:
        job["constraint_json"] = ""
    return job


def _parse_keep(value, default: float) -> float:
    try:
        k = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        k = float(default)
    return max(0.0, min(1.0, k))


def _parse_idle_motion_keep(value, default: float = IDLE_MOTION_KEEP) -> float:
    return _parse_keep(value, default)


def _parse_action_motion_keep(value, default: float = ACTION_MOTION_KEEP) -> float:
    return _parse_keep(value, default)


def _parse_output_scale(value, default: float = 1.0) -> float:
    try:
        scale = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        scale = float(default)
    return max(0.25, min(1.0, scale))


def _parse_pose_strength(value, default: float = 1.0) -> float:
    """SCAIL pose-guide strength (how strongly the skeleton constrains the video).

    Default 1.0 matches the official SCAIL2 workflow / Camera Lab (fully follow
    the skeleton, so a single raised hand stays single).
    """
    try:
        v = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        v = float(default)
    return max(0.0, min(1.0, v))


def _parse_cfg(value, default: float = 3.0) -> float:
    """SCAIL classifier-free guidance (how strongly the prompts steer the video)."""
    try:
        v = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        v = float(default)
    return max(1.0, min(10.0, v))


def resolve_scail_size(image: Path, meta: dict, scale=None) -> tuple[int, int]:
    """Output size for SCAIL video.

    Explicit SCAIL scale wins over session size so skeleton motion can be
    generated once and rendered at different final video resolutions later.
    """
    if scale not in (None, ""):
        return _output_size(image, scale=_parse_output_scale(scale))
    if meta.get("size"):
        w, h = meta["size"]
        return int(w), int(h)
    return _output_size(image, scale=meta.get("scale", 1.0))


def _load_extract_pose_and_camera(run_dir: Path):
    """Return (base_pose[J,3]|None, camera_dict|None) for session-consistent framing."""
    extract_pose = run_dir / "extract_pose.npy"
    if not extract_pose.is_file():
        return None, None
    base = np.load(extract_pose)
    cam = skeleton_camera_from_joints(base)
    return base, cam


def _load_action_base_pose(run_dir: Path, seed: int):
    """Base start pose for action: extract_pose first; idle frame-0 if present.

    Action does not require idle. Prefer extract_pose.npy (from Extract).
    If idle was already run, its frame-0 is also extract-anchored and may be used
    as a fallback when extract_pose is missing.
    """
    base, cam = _load_extract_pose_and_camera(run_dir)
    if base is not None:
        return np.asarray(base, dtype=np.float64).copy(), cam, "extract_pose"
    idle_npz = run_dir / f"idle_seed{seed}.npz"
    if idle_npz.is_file():
        try:
            with np.load(idle_npz) as z:
                Pi = z["posed_joints"]
            if Pi.ndim == 3 and Pi.shape[0] >= 1:
                base = np.asarray(Pi[0], dtype=np.float64).copy()
                cam = skeleton_camera_from_joints(base)
                return base, cam, "idle_frame0"
        except Exception:
            pass
    return None, None, None


def stage_idle(
    run_id: str,
    *,
    idle_prompt: str | None = None,
    idle_motion_keep: float | None = None,
    runs_dir: Path = RUNS_DIR,
    client: ComfyClient | None = None,
) -> dict:
    """Kimodo idle motion only → skeleton + SCAIL guide (no character video yet)."""
    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)
    if not meta.get("extracted"):
        return {"run_id": run_id, "errors": {"idle": "run extract first"}}
    image = _find_image(run_dir)
    seed = meta["seed"]
    scale = meta.get("scale", 1.0)
    pose_mode = meta.get("pose_mode", "standing")
    keep = _parse_idle_motion_keep(
        idle_motion_keep if idle_motion_keep is not None else meta.get("idle_motion_keep"),
        default=IDLE_MOTION_KEEP,
    )
    out_w, out_h = meta.get("size") or _output_size(image, scale=scale)
    # No ensure_mouth_still: idle drives a jawless skeleton; mouth control is SCAIL's.
    idle_text = (idle_prompt or "").strip() or DEFAULT_IDLE_PROMPT
    out: dict = {
        "run_id": run_id,
        "errors": {},
        "seed": seed,
        "pose_mode": pose_mode,
        "idle_motion_keep": keep,
    }

    try:
        job = _kimodo_job(
            run_dir,
            seed=seed,
            name="idle",
            prompt=idle_text,
            pose_mode=pose_mode,
            duration=IDLE_DURATION_SEC,
        )
        job_path = run_dir / "kimodo_job_idle.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        _run_subprocess("gen_kimodo_idle", [
            str(SEATED_HERE / "gen_kimodo_standalone.py"),
            str(job_path),
        ])
        npz = run_dir / f"idle_seed{seed}.npz"
        if not npz.is_file():
            out["errors"]["idle"] = "idle npz missing"
            return out
        # Close NpzFile before overwrite (Windows locks open mmap files).
        with np.load(npz) as raw:
            save_kw = {k: raw[k].copy() for k in raw.files}
        P_raw = np.asarray(save_kw["posed_joints"], dtype=np.float64)
        std_raw = float(P_raw.std(axis=0).mean())
        base, cam = _load_extract_pose_and_camera(run_dir)
        # Live2D-product idle: head/chest micro, arms/legs locked, ~2s seamless loop.
        P = shape_live2d_idle(
            P_raw,
            base_pose=base,
            keep=keep,
            fps=float(FPS),
        )
        std_source = float(P.std(axis=0).mean())
        save_kw["posed_joints"] = P
        np.savez(npz, **save_kw)
        skel = run_dir / "idle_skel.mp4"
        guide = run_dir / "idle_guide.mp4"
        # Same camera as extract so idle frame-0 matches extract still.
        render_smplx_guide(P, skel, camera=cam)
        _pad_to_aspect(skel, guide, out_w, out_h)
        out["skeleton"] = _rel_url(skel)
        png = skel.with_suffix(".png")
        out["skeleton_png"] = _rel_url(png) if png.is_file() else None
        out["n_frames"] = int(P.shape[0])
        out["motion_std_before"] = std_raw
        out["motion_std"] = std_source
        out["idle_motion_keep"] = keep
        out["idle_duration"] = IDLE_DURATION_SEC
        out["idle_style"] = "live2d"
        out["idle_anchored_to_extract"] = base is not None
        out["extract_pose_strength"] = 1.0  # always fully stuck
        out["kimodo_constraint"] = bool(job.get("constraint_json"))
    except Exception as exc:
        out["errors"]["idle"] = str(exc)
        return out

    meta["idle_done"] = True
    meta["step"] = "idle_skel"
    meta["idle_prompt"] = idle_text
    meta["idle_motion_keep"] = keep
    meta["idle_scail_done"] = False
    meta["scail_done"] = False  # invalidate final videos if re-run
    _save_meta(run_dir, meta)
    out["size"] = [out_w, out_h]
    return out


def stage_action(
    run_id: str,
    *,
    action_prompt: str,
    pose_mode: str | None = None,
    action_motion_keep: float | None = None,
    action_duration: float | None = None,
    runs_dir: Path = RUNS_DIR,
    client: ComfyClient | None = None,
) -> dict:
    """Kimodo action motion only → skeleton + guide (no SCAIL, no overshoot)."""
    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)
    if not meta.get("extracted"):
        return {"run_id": run_id, "errors": {"action": "run extract first"}}
    if not (action_prompt or "").strip():
        return {"run_id": run_id, "errors": {"action": "action_prompt is required"}}
    action_text = sanitize_action(action_prompt)
    lower_pose = lower_body_action_requested(action_text)
    close_lower_pose = close_lower_body_action_requested(action_text)
    keep = _parse_action_motion_keep(
        action_motion_keep if action_motion_keep is not None else meta.get("action_motion_keep"),
        default=ACTION_MOTION_KEEP,
    )
    duration = _parse_duration(
        action_duration if action_duration is not None else meta.get("action_duration"),
        ACTION_DURATION_SEC,
        lo=1.0,
        hi=5.0,
    )

    image = _find_image(run_dir)
    seed = meta["seed"]
    scale = meta.get("scale", 1.0)
    session_pose_mode = _normalize_pose_mode(meta.get("pose_mode"))
    action_pose_mode = _normalize_pose_mode(pose_mode, default=session_pose_mode)
    out_w, out_h = meta.get("size") or _output_size(image, scale=scale)
    out: dict = {
        "run_id": run_id,
        "errors": {},
        "seed": seed,
        "pose_mode": action_pose_mode,
        "session_pose_mode": session_pose_mode,
        "action_motion_keep": keep,
        "action_duration": duration,
        "preserve_lower_pose": action_pose_mode == "standing" and lower_pose,
        "close_lower_pose": action_pose_mode == "standing" and close_lower_pose,
        "extract_pose_strength": 1.0,
    }

    try:
        job = _kimodo_job(
            run_dir,
            seed=seed,
            name="action",
            prompt=action_text,
            pose_mode=action_pose_mode,
            duration=duration,
        )
        job_path = run_dir / "kimodo_job_action.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        _run_subprocess("gen_kimodo_action", [
            str(SEATED_HERE / "gen_kimodo_standalone.py"),
            str(job_path),
        ])
        npz = run_dir / f"action_seed{seed}.npz"
        if not npz.is_file():
            out["errors"]["action"] = "action npz missing"
            return out
        with np.load(npz) as raw:
            save_kw = {k: raw[k].copy() for k in raw.files}
        P_raw = np.asarray(save_kw["posed_joints"], dtype=np.float64)
        base, cam, base_src = _load_action_base_pose(run_dir, seed)
        if base is None:
            out["errors"]["action"] = (
                "missing extract_pose.npy — re-run Extract before Action"
            )
            return out
        # Sitting/lying: free Kimodo then lock lower body; boost weak upper motion
        # so keep=100% is clearly different from idle micro-motion.
        lock_lower = action_pose_mode in ("sitting", "lying")
        f0_err_before = float(np.linalg.norm(P_raw[0] - base, axis=-1).mean())
        P = align_motion_to_base_pose(
            P_raw,
            base,
            keep=keep,
            lock_lower_body=lock_lower,
            preserve_lower_pose=action_pose_mode == "standing" and lower_pose,
            close_lower_body=action_pose_mode == "standing" and close_lower_pose,
            boost_upper=True,
        )
        f0_err = float(np.linalg.norm(P[0] - base, axis=-1).mean())
        if f0_err > 1e-4:
            # Hard fail rather than silently shipping standing default start.
            out["errors"]["action"] = (
                f"start-pose align failed (f0_err={f0_err:.5f}, before={f0_err_before:.5f})"
            )
            return out
        save_kw["posed_joints"] = P.astype(np.float64)
        # Rewrite aligned joints. np.savez appends ".npz" if missing — use a real .npz name.
        # On Windows, Path.replace can fail if target is open; write then replace.
        tmp_npz = run_dir / f"action_seed{seed}_aligned.npz"
        np.savez(tmp_npz, **save_kw)
        try:
            tmp_npz.replace(npz)
        except OSError:
            # Fallback: overwrite in place if replace fails (file lock).
            np.savez(npz, **save_kw)
            try:
                tmp_npz.unlink(missing_ok=True)
            except OSError:
                pass
        # Verify what is on disk
        with np.load(npz) as chk:
            f0_disk = float(np.linalg.norm(chk["posed_joints"][0] - base, axis=-1).mean())
        if f0_disk > 1e-4:
            out["errors"]["action"] = f"npz rewrite verify failed (f0_disk={f0_disk:.5f})"
            return out
        skel = run_dir / "action_skel.mp4"
        guide = run_dir / "action_guide.mp4"
        # Same camera as extract/idle so frame0 matches the still the user approved.
        render_smplx_guide(P, skel, camera=cam)
        _pad_to_aspect(skel, guide, out_w, out_h)
        out["skeleton"] = _rel_url(skel)
        png = skel.with_suffix(".png")
        out["skeleton_png"] = _rel_url(png) if png.is_file() else None
        out["n_frames"] = int(P.shape[0])
        out["motion_std"] = float(np.asarray(P).std(axis=0).mean())
        out["motion_std_raw"] = float(np.asarray(P_raw).std(axis=0).mean())
        # Upper-body (arms/spine) residual magnitude — should be >> idle when keep=1
        upper_idx = [j for j in range(12, min(22, P.shape[1]))]
        if upper_idx:
            up = P[:, upper_idx, :] - P[0:1, upper_idx, :]
            out["upper_motion_std"] = float(np.sqrt(np.mean(up * up)))
        out["action_anchored_to_extract"] = True
        out["action_base_source"] = base_src
        out["action_f0_err"] = f0_err
        out["action_f0_err_before"] = f0_err_before
        out["action_f0_err_disk"] = f0_disk
        out["action_lock_lower"] = lock_lower
        out["action_close_lower_pose"] = action_pose_mode == "standing" and close_lower_pose
        out["action_motion_keep"] = keep
        out["kimodo_constraint"] = bool(job.get("constraint_json"))
    except Exception as exc:
        out["errors"]["action"] = str(exc)
        return out

    meta["action_done"] = True
    meta["step"] = "action_skel"
    meta["action_prompt"] = action_text
    # Step-by-step chooses the run pose at Action time; Run All already set the
    # same field at session creation. Keep one pose mode for all later stages.
    meta["pose_mode"] = action_pose_mode
    meta.pop("action_pose_mode", None)
    meta["action_preserve_lower_pose"] = action_pose_mode == "standing" and lower_pose
    meta["action_close_lower_pose"] = action_pose_mode == "standing" and close_lower_pose
    meta["action_motion_keep"] = keep
    meta["action_duration"] = duration
    meta["action_anchored_to_extract"] = True
    meta["joint_overshoot"] = False
    meta["action_scail_done"] = False
    meta["scail_done"] = False
    meta["time_overshoot"] = False
    # Drop previous joint npz so SCAIL won't use a stale springed motion.
    joint_npz = run_dir / f"action_joint_seed{seed}.npz"
    if joint_npz.is_file():
        try:
            joint_npz.unlink()
        except OSError:
            pass
    _save_meta(run_dir, meta)
    out["size"] = [out_w, out_h]
    return out


def stage_scail(
    run_id: str,
    *,
    which: str = "both",
    runs_dir: Path = RUNS_DIR,
    client: ComfyClient | None = None,
    scale=None,
    pose_strength=None,
    cfg=None,
    positive_idle: str | None = None,
    positive_action: str | None = None,
    negative: str | None = None,
) -> dict:
    """SCAIL2: drive character image with skeleton guide(s).

    which:
      - \"idle\"   — only idle.mp4 (needs idle skeleton motion done)
      - \"action\" — only action.mp4 (needs action skeleton motion done)
      - \"both\"   — idle then action (legacy / Run all)

    Optional positive_idle / positive_action / negative override the product
    defaults (empty or None → build_scail_positive / SCAIL_NEGATIVE).
    """
    which = (which or "both").strip().lower()
    if which not in ("idle", "action", "both"):
        return {"run_id": run_id, "errors": {"scail": "which must be idle|action|both"}}

    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)

    neg = (negative if negative is not None else "").strip() or SCAIL_NEGATIVE
    pos_idle_override = (positive_idle or "").strip()
    pos_action_override = (positive_action or "").strip()

    labels: list[tuple[str, str]] = []
    if which in ("idle", "both"):
        if not meta.get("idle_done"):
            return {
                "run_id": run_id,
                "errors": {"idle": "run idle skeleton motion first"},
            }
        labels.append(
            (
                "idle",
                pos_idle_override or build_scail_positive("idle"),
            )
        )
    if which in ("action", "both"):
        if not meta.get("action_done"):
            return {
                "run_id": run_id,
                "errors": {"action": "run action skeleton motion first"},
            }
        labels.append(
            (
                "action",
                pos_action_override
                or build_scail_positive("action", meta.get("action_prompt")),
            )
        )

    image = _find_image(run_dir)
    seed = meta["seed"]
    out_w, out_h = resolve_scail_size(image, meta, scale=scale)
    pose_mode = meta["pose_mode"]
    out: dict = {
        "run_id": run_id,
        "errors": {},
        "seed": seed,
        "size": [out_w, out_h],
        "scale": _parse_output_scale(scale, meta.get("scale", 1.0)),
        "which": which,
    }
    ps = _parse_pose_strength(pose_strength)
    cf = _parse_cfg(cfg)
    out["pose_strength"] = ps
    out["cfg"] = cf

    if client is None:
        client = ComfyClient()

    fre0 = client.free_vram(interrupt=False, clear_queue=True, wait_s=12, min_free_gb=4.0)
    out["comfy_free_start"] = fre0
    print(f"[stage_scail] which={which} free_vram start: {fre0}", flush=True)

    for label, positive in labels:
        guide = run_dir / f"{label}_guide.mp4"
        skel = run_dir / f"{label}_skel.mp4"
        if not guide.is_file():
            if not skel.is_file():
                out["errors"][label] = f"missing {label}_skel / guide — re-run {label} motion"
                continue
            try:
                _pad_to_aspect(skel, guide, out_w, out_h)
            except Exception as exc:
                out["errors"][label] = f"guide pad failed: {exc}"
                continue
        npz = run_dir / f"{label}_seed{seed}.npz"
        if label == "action":
            joint_npz = run_dir / f"action_joint_seed{seed}.npz"
            if joint_npz.is_file() and meta.get("joint_overshoot"):
                npz = joint_npz
        n_frames = int(np.load(npz)["posed_joints"].shape[0]) if npz.is_file() else 89
        try:
            out_mp4 = drive_character(
                client, guide, image, run_dir / f"{label}.mp4",
                length=align_4k1(n_frames), width=out_w, height=out_h,
                prefix=f"mp_{pose_mode}_{label}", seed=seed,
                pose_strength=ps, cfg=cf,
                positive=positive,
                negative=neg,
            )
            # Mouth: rely on SCAIL positive/negative prompts (pixel paste face_lock
            # disabled — quality was unacceptable on production standees).
            meta[f"{label}_mouth_locked"] = False
            meta[f"{label}_scail_positive"] = positive
            meta[f"{label}_scail_negative"] = neg
            out[label] = _rel_url(out_mp4)
            meta[f"{label}_scail_done"] = True
        except Exception as exc:
            out["errors"][label] = str(exc)
            client.free_vram(interrupt=True, clear_queue=True, wait_s=5, min_free_gb=0)

    # Non-blocking reclaim — do not hold the HTTP response for 30s+ after Comfy is done.
    fre1 = client.free_vram(interrupt=False, clear_queue=False, wait_s=3, min_free_gb=0)
    out["comfy_free_end"] = fre1
    print(f"[stage_scail] free_vram end: {fre1}", flush=True)

    # Include already-done sibling URLs so UI can refresh player
    for label in ("idle", "action"):
        if out.get(label):
            continue
        p = run_dir / f"{label}.mp4"
        if p.is_file():
            out[label] = _rel_url(p)

    idle_ok = bool(meta.get("idle_scail_done") or (run_dir / "idle.mp4").is_file())
    action_ok = bool(meta.get("action_scail_done") or (run_dir / "action.mp4").is_file())
    # Refresh flags from disk if meta lagging
    if (run_dir / "idle.mp4").is_file():
        meta["idle_scail_done"] = True
        idle_ok = True
    if (run_dir / "action.mp4").is_file():
        meta["action_scail_done"] = True
        action_ok = True

    out["idle_scail_done"] = idle_ok
    out["action_scail_done"] = action_ok
    if idle_ok and action_ok:
        meta["scail_done"] = True
        meta["step"] = "scail"
    elif out.get("idle") and not out.get("errors", {}).get("idle"):
        meta["step"] = "scail_idle"
    elif out.get("action") and not out.get("errors", {}).get("action"):
        meta["step"] = "scail_action"
    _save_meta(run_dir, meta)
    return out


def stage_joint_overshoot(
    run_id: str,
    *,
    mode: str = "preview",
    omega: float | None = None,
    zeta: float | None = None,
    soft: float | None = None,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Non-destructive joint-overshoot on the action skeleton.

    mode="preview": spring raw ``action_seed`` with the given params, save
        ``action_joint_seed`` + render ``action_joint_skel.mp4``. Never touches
        ``action_skel.mp4`` / ``action_guide.mp4``.
    mode="carry": (re)pad the overshot skeleton into ``action_guide.mp4`` so SCAIL
        uses it. Self-springs a preview first if none exists (Run-all one-shot).
    mode="uncarry": re-pad the plain ``action_skel.mp4`` into ``action_guide.mp4``.
    """
    run_dir = Path(runs_dir) / run_id
    meta = _load_meta(run_dir)
    if not meta.get("action_done"):
        return {"run_id": run_id, "errors": {"joint": "run action skeleton first"}, "mode": mode}

    seed = meta["seed"]
    scale = meta.get("scale", 1.0)
    image = _find_image(run_dir)
    out_w, out_h = meta.get("size") or _output_size(image, scale=scale)
    out: dict = {"run_id": run_id, "errors": {}, "mode": mode, "seed": seed, "size": [out_w, out_h]}

    raw = run_dir / f"action_seed{seed}.npz"
    joint_npz = run_dir / f"action_joint_seed{seed}.npz"
    joint_skel = run_dir / "action_joint_skel.mp4"
    action_skel = run_dir / "action_skel.mp4"
    guide = run_dir / "action_guide.mp4"

    def _spring_preview():
        P = np.asarray(np.load(raw)["posed_joints"], dtype=np.float64)
        P = spring_follow(
            P, FPS,
            omega=JOINT_SPRING["omega"] if omega is None else float(omega),
            zeta=JOINT_SPRING["zeta"] if zeta is None else float(zeta),
            soft_scale=JOINT_SPRING["soft"] if soft is None else float(soft),
        )
        pose_mode = _normalize_pose_mode(
            meta.get("action_pose_mode"),
            default=meta.get("pose_mode", "standing"),
        )
        base, cam, _src = _load_action_base_pose(run_dir, seed)
        if base is not None:
            P = align_motion_to_base_pose(
                P, base, keep=1.0, lock_lower_body=pose_mode in ("sitting", "lying"),
            )
        np.savez(joint_npz, posed_joints=P)
        render_smplx_guide(P, joint_skel, camera=cam)
        return P

    try:
        if not raw.is_file():
            out["errors"]["joint"] = "missing action_seed npz — re-run action motion"
            return out

        if mode == "preview":
            P = _spring_preview()
            out["skeleton"] = _rel_url(joint_skel)
            out["n_frames"] = int(P.shape[0])
            out["motion_std"] = float(np.asarray(P).std(axis=0).mean())
            return out

        if mode == "carry":
            if not joint_skel.is_file():
                _spring_preview()  # self-spring for the Run-all one-shot path
            _pad_to_aspect(joint_skel, guide, out_w, out_h)
            meta["joint_overshoot"] = True
        elif mode == "uncarry":
            _pad_to_aspect(action_skel, guide, out_w, out_h)
            meta["joint_overshoot"] = False
        else:
            out["errors"]["joint"] = f"unknown mode {mode!r}"
            return out

        meta["action_scail_done"] = False
        meta["scail_done"] = False
        meta["step"] = "action_joint" if mode == "carry" else "action"
        _save_meta(run_dir, meta)
        out["joint_overshoot"] = bool(meta["joint_overshoot"])
        out["guide"] = _rel_url(guide)
        return out
    except Exception as exc:
        out["errors"]["joint"] = str(exc)
        return out


def stage_time_overshoot(
    run_id: str | None = None,
    *,
    runs_dir: Path = RUNS_DIR,
    upload_bytes: bytes | None = None,
    upload_filename: str = "upload.mp4",
    overshoot_b: float | None = None,
    overshoot_t: float | None = None,
) -> dict:
    """Time-space spring remap on final action character video only.

    Source priority:
      1. Optional uploaded video (saved as ``time_upload.*`` then used as src)
      2. ``action_nobg.mp4`` if present (after bg remove)
      3. ``action.mp4`` (SCAIL)

    Idle is unchanged. No Comfy / no joint edit. Creates a run folder when
    only an upload is provided (standalone time overshoot).
    """
    import shutil

    spring = dict(TIME_SPRING)
    if overshoot_b is not None:
        spring["b"] = max(0.0, min(0.7, float(overshoot_b)))
    if overshoot_t is not None:
        spring["t"] = max(0.5, min(1.8, float(overshoot_t)))

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Treat missing/stale run folders as "no session" (upload can still mint a new one).
    if run_id and not (runs_dir / run_id).is_dir():
        run_id = None

    if not run_id:
        if upload_bytes is None:
            return {
                "errors": {
                    "time": "create a session or upload a video first"
                }
            }
        run_id = uuid.uuid4().hex
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": run_id,
            "pose_mode": "standing",
            "seed": 0,
            "step": "time_overshoot_only",
            "standalone_time_overshoot": True,
        }
        _save_meta(run_dir, meta)
    else:
        run_dir = runs_dir / run_id
        try:
            meta = _load_meta(run_dir)
        except FileNotFoundError:
            meta = {"run_id": run_id}

    action_p = run_dir / "action.mp4"
    nobg_p = run_dir / "action_nobg.mp4"
    nobg_webm_p = run_dir / "action_nobg.webm"
    nobg_alpha_p = run_dir / "action_nobg_alpha.mov"
    upload_src: Path | None = None

    if upload_bytes is not None:
        ext = Path(upload_filename or "upload.mp4").suffix or ".mp4"
        if ext.lower() not in (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"):
            ext = ".mp4"
        upload_src = run_dir / f"time_upload{ext}"
        upload_src.write_bytes(upload_bytes)

    # Prefer real-alpha sources so time remap does not bake a solid background.
    # Priority: upload → action_nobg.webm → action_nobg_alpha.mov → action_nobg.mp4 → action.mp4
    if upload_src is not None and upload_src.is_file():
        src = upload_src
        time_source = "upload"
    elif nobg_webm_p.is_file():
        src = nobg_webm_p
        time_source = "action_nobg_webm"
    elif nobg_alpha_p.is_file():
        src = nobg_alpha_p
        time_source = "action_nobg_alpha"
    elif nobg_p.is_file():
        src = nobg_p
        time_source = "action_nobg"
    else:
        src = action_p
        time_source = "action"

    if not src.is_file():
        return {
            "run_id": run_id,
            "errors": {
                "time": (
                    "missing video — upload a file, or generate action.mp4 / "
                    "action_nobg.* in this session"
                )
            },
        }

    out: dict = {
        "run_id": run_id,
        "errors": {},
        "seed": meta.get("seed"),
        "time_source": time_source,
    }
    if upload_src is not None:
        out["time_upload"] = _rel_url(upload_src)

    try:
        timed = run_dir / "action_timed.mp4"
        timed_webm = run_dir / "action_timed.webm"
        remap = time_remap_file(
            src,
            timed,
            out_webm=timed_webm,
            prefer_alpha=True,
            **spring,
        )
        # Backup original SCAIL (with bg) once when remapping session action.
        raw_backup = run_dir / "action_scail.mp4"
        if (
            time_source not in ("upload",)
            and not raw_backup.is_file()
            and action_p.is_file()
        ):
            try:
                shutil.copy2(action_p, raw_backup)
            except OSError:
                pass
        # Browser player uses action.mp4 (H.264; black under transparent regions).
        try:
            shutil.copy2(timed, action_p)
        except OSError:
            pass
        # Keep nobg preview mp4 in sync with timed black-composite preview.
        if time_source.startswith("action_nobg") or remap.get("has_alpha"):
            try:
                shutil.copy2(timed, nobg_p)
            except OSError:
                pass
            out["action_nobg"] = _rel_url(nobg_p)

        # Preserve / refresh true-alpha exports for CapCut after time remap.
        webm_out = remap.get("out_webm")
        if webm_out and Path(webm_out).is_file():
            try:
                shutil.copy2(webm_out, nobg_webm_p)
            except OSError:
                pass
            out["action_nobg_webm"] = _rel_url(nobg_webm_p)
            try:
                from pipeline.bgremove import webm_to_prores_alpha

                webm_to_prores_alpha(nobg_webm_p, nobg_alpha_p)
                out["action_nobg_alpha"] = _rel_url(nobg_alpha_p)
            except Exception as alpha_exc:
                # Soft warning only — timed mp4/webm already succeeded.
                out.setdefault("warnings", {})["time_alpha"] = str(alpha_exc)
                print(f"[time] alpha mov after remap failed: {alpha_exc}", flush=True)

        out["action"] = _rel_url(action_p) or _rel_url(timed)
        out["action_timed"] = _rel_url(timed)
        if timed_webm.is_file():
            out["action_timed_webm"] = _rel_url(timed_webm)
        out["has_alpha"] = bool(remap.get("has_alpha"))
        out["step"] = "time_overshoot"
    except Exception as exc:
        out["errors"]["time"] = str(exc)
        return out

    # Clear hard errors on success path (warnings may still be present).
    if out.get("action") or out.get("action_timed"):
        out["errors"] = {
            k: v
            for k, v in (out.get("errors") or {}).items()
            if k not in ("time_alpha",)
        }

    meta["time_overshoot"] = True
    meta["time_overshoot_b"] = spring["b"]
    meta["time_overshoot_t"] = spring["t"]
    meta["time_overshoot_source"] = out.get("time_source")
    meta["time_overshoot_has_alpha"] = bool(out.get("has_alpha"))
    meta["step"] = "time_overshoot"
    _save_meta(run_dir, meta)
    idle_nobg = run_dir / "idle_nobg.mp4"
    idle_p = run_dir / "idle.mp4"
    if idle_nobg.is_file():
        out["idle"] = _rel_url(idle_nobg)
        out["idle_nobg"] = _rel_url(idle_nobg)
    elif idle_p.is_file():
        out["idle"] = _rel_url(idle_p)
    return out


def _bgremove_one(
    run_dir: Path,
    src: Path,
    label: str,
    *,
    model: str,
) -> dict:
    """Run worker on one video; return dict of URL fields / error.

    Also writes CapCut-friendly ``{label}_nobg_alpha.mov`` (ProRes 4444) from
    the VP9 alpha WebM via libvpx-vp9 decode.
    """
    import shutil

    from pipeline.bgremove import run_bgremove, webm_to_prores_alpha

    piece: dict = {}
    if not src.is_file():
        piece["error"] = f"missing {src.name}"
        return piece
    work = run_dir / f"_bgremove_{label}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        result = run_bgremove(src, work, model=model, formats="webm", fp16=True)
        preview = result.get("preview")
        if preview and Path(preview).is_file():
            dest_prev = run_dir / f"{label}_nobg.mp4"
            shutil.copy2(preview, dest_prev)
            piece["nobg"] = _rel_url(dest_prev)
        webm_path: Path | None = None
        for p in result.get("outputs") or []:
            p = Path(p)
            if p.suffix.lower() == ".webm" and p.is_file():
                dest_w = run_dir / f"{label}_nobg.webm"
                shutil.copy2(p, dest_w)
                piece["nobg_webm"] = _rel_url(dest_w)
                webm_path = dest_w
        if webm_path is not None:
            try:
                dest_mov = run_dir / f"{label}_nobg_alpha.mov"
                webm_to_prores_alpha(webm_path, dest_mov)
                piece["nobg_alpha"] = _rel_url(dest_mov)
            except Exception as alpha_exc:
                # WebM/preview still usable; surface alpha failure as soft error.
                piece["alpha_error"] = str(alpha_exc)
                print(f"[bgremove] alpha mov skip ({label}): {alpha_exc}", flush=True)
        if not piece.get("nobg") and not piece.get("nobg_webm"):
            piece["error"] = "worker produced no output files"
    except Exception as exc:
        piece["error"] = str(exc)
    return piece


def stage_bgremove(
    run_id: str | None = None,
    *,
    which: str = "both",
    model: str = "RMBG-2.0 HQ",
    upload_bytes: bytes | None = None,
    upload_filename: str = "upload.mp4",
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Video background removal via videoBGremoval worker.

    No hard pipeline prerequisites:
      - which idle|action|both: process session SCAIL mp4s if present
      - which upload (or upload_bytes set): process an uploaded video only
    Creates a fresh run_id when none is provided (standalone upload mode).
    """
    from pipeline.bgremove import BG_MODELS

    which = (which or "both").strip().lower()
    if which not in ("idle", "action", "both", "upload"):
        return {"run_id": run_id, "errors": {"bgremove": "which must be idle|action|both|upload"}}
    if model not in BG_MODELS:
        model = BG_MODELS[0]

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Standalone upload without session → new run folder
    if not run_id or not (runs_dir / run_id).is_dir():
        if upload_bytes is None and which == "upload":
            return {"errors": {"bgremove": "upload a video or create a session first"}}
        if upload_bytes is not None or which == "upload":
            run_id = uuid.uuid4().hex
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "run_id": run_id,
                "pose_mode": "standing",
                "seed": 0,
                "step": "bgremove_only",
                "standalone_bgremove": True,
            }
            _save_meta(run_dir, meta)
        else:
            return {"errors": {"bgremove": "invalid run_id — create session or upload a video"}}
    else:
        run_dir = runs_dir / run_id
        try:
            meta = _load_meta(run_dir)
        except FileNotFoundError:
            meta = {"run_id": run_id}

    out: dict = {
        "run_id": run_id,
        "errors": {},
        "which": which,
        "model": model,
    }

    # Free Comfy VRAM before matting (best-effort; may be offline).
    try:
        fre = ComfyClient().free_vram(
            interrupt=False, clear_queue=False, wait_s=15, min_free_gb=4.0
        )
        out["comfy_free"] = fre
    except Exception as exc:
        out["comfy_free"] = {"error": str(exc)}

    jobs: list[tuple[str, Path]] = []  # (label, src)

    # Optional arbitrary video (no pipeline prerequisites).
    if upload_bytes is not None:
        ext = Path(upload_filename or "upload.mp4").suffix or ".mp4"
        if ext.lower() not in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
            ext = ".mp4"
        up_path = run_dir / f"upload{ext}"
        up_path.write_bytes(upload_bytes)
        out["upload"] = _rel_url(up_path)
        jobs.append(("upload", up_path))

    # Optional session SCAIL outputs (skip quietly if missing).
    if which in ("idle", "both") and (run_dir / "idle.mp4").is_file():
        jobs.append(("idle", run_dir / "idle.mp4"))
    if which in ("action", "both") and (run_dir / "action.mp4").is_file():
        jobs.append(("action", run_dir / "action.mp4"))

    # which=upload with only file already handled; which=both with no files + no upload:
    if not jobs:
        out["errors"]["bgremove"] = (
            "nothing to process — choose a video file, or generate idle/action.mp4 in this session"
        )
        return out

    for label, src in jobs:
        piece = _bgremove_one(run_dir, src, label, model=model)
        if piece.get("error"):
            out["errors"][label] = piece["error"]
            continue
        if piece.get("nobg"):
            out[f"{label}_nobg"] = piece["nobg"]
        if piece.get("nobg_webm"):
            out[f"{label}_nobg_webm"] = piece["nobg_webm"]
        if piece.get("nobg_alpha"):
            out[f"{label}_nobg_alpha"] = piece["nobg_alpha"]
        if piece.get("alpha_error"):
            out["errors"][f"{label}_alpha"] = piece["alpha_error"]
        meta[f"{label}_bgremove_done"] = True
        if piece.get("nobg_alpha"):
            meta[f"{label}_nobg_alpha"] = True

    has_out = any(
        out.get(k)
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
    if has_out:
        meta["bgremove_done"] = True
        meta["bgremove_model"] = model
        meta["step"] = "bgremove"
        _save_meta(run_dir, meta)

    # Player convenience
    if out.get("idle_nobg"):
        out["idle"] = out["idle_nobg"]
    elif (run_dir / "idle.mp4").is_file():
        out["idle"] = _rel_url(run_dir / "idle.mp4")
    if out.get("action_nobg"):
        out["action"] = out["action_nobg"]
    elif out.get("upload_nobg"):
        out["action"] = out["upload_nobg"]
    elif (run_dir / "action.mp4").is_file():
        out["action"] = _rel_url(run_dir / "action.mp4")

    # Drop soft "missing" errors when we still produced something from other sources
    if has_out:
        for k in list(out["errors"].keys()):
            if "no " in str(out["errors"][k]) and "mp4" in str(out["errors"][k]):
                # keep only real process errors
                if "optional" in str(out["errors"][k]):
                    del out["errors"][k]

    return out


def stage_overshoot(
    run_id: str,
    *,
    overshoot: set,
    runs_dir: Path = RUNS_DIR,
    client: ComfyClient | None = None,
) -> dict:
    """Legacy dispatcher: joint → skeleton spring; time → final video remap.

    Prefer ``stage_joint_overshoot`` / ``stage_time_overshoot`` from the UI.
    ``client`` is unused (joint no longer re-SCAILs).
    """
    del client  # joint no longer needs Comfy
    plan = plan_steps(overshoot or set())
    out: dict = {"run_id": run_id, "errors": {}}
    if not plan["joint"] and not plan["time"]:
        out["errors"]["overshoot"] = "select joint and/or time overshoot"
        return out
    if plan["joint"]:
        # mode="carry" reproduces the old apply-then-SCAIL behaviour (default is now "preview").
        j = stage_joint_overshoot(run_id, mode="carry", runs_dir=runs_dir)
        out.update({k: v for k, v in j.items() if k != "errors"})
        if j.get("errors"):
            out.setdefault("errors", {}).update(j["errors"])
    if plan["time"]:
        t = stage_time_overshoot(run_id, runs_dir=runs_dir)
        out.update({k: v for k, v in t.items() if k != "errors"})
        if t.get("errors"):
            out.setdefault("errors", {}).update(t["errors"])
    return out
