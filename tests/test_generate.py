from pathlib import Path

from PIL import Image

import numpy as np

from pipeline.generate import (
    plan_steps,
    sanitize_action,
    ensure_mouth_still,
    dampen_idle_joints,
    prepare_idle_source_motion,
    align_motion_to_base_pose,
    DEFAULT_IDLE_PROMPT,
    MOUTH_STILL_CLAUSE,
    _output_size,
)


def test_plan_steps_selects_overshoot():
    assert plan_steps(set()) == {"joint": False, "time": False}
    assert plan_steps({"joint"}) == {"joint": True, "time": False}
    assert plan_steps({"joint", "time"}) == {"joint": True, "time": True}


def test_sanitize_drops_turning():
    assert "turn" not in sanitize_action("she turns and waves").lower()


def test_sanitize_blocks_talking_and_keeps_mouth_still():
    p = sanitize_action("she waves and talks loudly")
    # user speech verbs stripped; anti-speech clause remains
    assert "talks" not in p.lower()
    assert "waves" in p.lower()
    assert "mouth closed" in p.lower()
    assert "no talking" in p.lower()


def test_ensure_mouth_still_idempotent():
    a = ensure_mouth_still("wave hello")
    b = ensure_mouth_still(a)
    assert a.count("Mouth closed") == 1
    assert b == a
    assert "mouth closed" in DEFAULT_IDLE_PROMPT.lower()
    assert "no talking" in MOUTH_STILL_CLAUSE.lower()


def test_default_idle_keeps_pose_and_avoids_large_motion():
    p = DEFAULT_IDLE_PROMPT.lower()
    assert "current pose" in p or "relaxed idle" in p
    assert "do not stand up" in p
    assert "stands nearly" not in p


def test_dampen_idle_joints_shrinks_motion():
    P = np.zeros((10, 22, 3), dtype=np.float64)
    P[:, 0, 0] = np.linspace(0, 1, 10)  # large drift on joint 0
    out = dampen_idle_joints(P, keep=0.12)
    assert out.shape == P.shape
    assert np.allclose(out[0], P[0])
    assert out[:, 0, 0].std() < P[:, 0, 0].std() * 0.2
    frozen = dampen_idle_joints(P, keep=0.0)
    assert np.allclose(frozen, np.repeat(P[0:1], 10, axis=0))
    full = dampen_idle_joints(P, keep=1.0)
    assert np.allclose(full, P)


def test_dampen_idle_joints_anchors_to_extract_pose():
    P = np.zeros((10, 22, 3), dtype=np.float64)
    P[:, 16, 1] = np.linspace(0.5, 1.5, 10)  # big arm swing
    base = np.zeros((22, 3), dtype=np.float64)
    base[16, 1] = 0.1  # extract: arm down
    out = dampen_idle_joints(P, keep=0.06, base_pose=base)
    assert np.allclose(out[0], base)
    # deltas scaled
    assert np.allclose(out[1:] - out[0:1], 0.06 * (P[1:] - P[0:1]))
    full = dampen_idle_joints(P, keep=1.0, base_pose=base)
    assert np.allclose(full[0], base)
    assert np.allclose(full - full[0:1], P - P[0:1])


def test_prepare_idle_source_boosts_near_static():
    P = np.zeros((20, 22, 3), dtype=np.float64)
    P[:, 0, 0] = 0.0001 * np.sin(np.linspace(0, 6.28, 20))
    out = prepare_idle_source_motion(P, ref_std=0.012)
    assert float(out.std()) > float(P.std())
    assert float(np.sqrt(np.mean((out - out.mean(0)) ** 2))) >= 0.01


def test_align_motion_to_base_pose_preserves_deltas():
    P = np.zeros((5, 22, 3), dtype=np.float64)
    P[:, 0, 0] = np.linspace(1.0, 2.0, 5)  # root drifts
    P[0, 5, 1] = 0.8  # wrong start arm height
    P[1:, 5, 1] = 0.8 + np.linspace(0.1, 0.4, 4)
    base = np.zeros((22, 3), dtype=np.float64)
    base[5, 1] = 0.2  # extract arm
    base[0, 1] = 0.9  # seated pelvis
    out = align_motion_to_base_pose(P, base, keep=1.0)
    assert np.allclose(out[0], base)
    # relative motion preserved on free joints
    assert np.allclose(out[1:, 5] - out[0, 5], P[1:, 5] - P[0, 5])
    half = align_motion_to_base_pose(P, base, keep=0.5)
    assert np.allclose(half[0], base)
    assert np.allclose(half[1:, 5] - half[0, 5], 0.5 * (P[1:, 5] - P[0, 5]))
    locked = align_motion_to_base_pose(P, base, keep=1.0, lock_lower_body=True)
    assert np.allclose(locked[0], base)
    # pelvis stays at base every frame when locked
    assert np.allclose(locked[:, 0, :], base[0])


def test_align_boost_upper_increases_weak_arm_motion():
    P = np.zeros((10, 22, 3), dtype=np.float64)
    base = np.zeros((22, 3), dtype=np.float64)
    base[0, 1] = 0.9
    # tiny arm wiggle on left wrist (20)
    P[:, 20, 1] = np.linspace(0, 0.01, 10)
    out = align_motion_to_base_pose(
        P, base, keep=1.0, lock_lower_body=True, boost_upper=True, upper_ref_std=0.035
    )
    assert np.allclose(out[0], base)
    assert np.allclose(out[:, 0, :], base[0])  # pelvis locked
    # wrist motion amplified well above the tiny raw wiggle
    assert float(out[:, 20, 1].ptp()) > 0.05


def test_output_size_keeps_aspect_and_scale(tmp_path):
    img = Path(tmp_path) / "p.png"
    Image.new("RGB", (1000, 2000), (0, 0, 0)).save(img)  # portrait 1:2
    w1, h1 = _output_size(img, scale=1.0)
    w2, h2 = _output_size(img, scale=0.5)
    assert w1 % 16 == 0 and h1 % 16 == 0
    assert abs(w1 / h1 - 0.5) < 0.05
    assert w2 <= w1 and h2 <= h1
    assert abs(w2 / h2 - w1 / h1) < 0.05
