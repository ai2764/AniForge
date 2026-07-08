import numpy as np
from pipeline.skeleton_spring import spring_follow
from pipeline import skeletons


def test_integrator_stable_at_high_omega():
    k = np.random.RandomState(0).randn(40, skeletons.N_JOINTS, 3).astype(np.float32) * 0.1
    out = spring_follow(k, 30, 200.0, 1.0, 0.0)
    assert np.isfinite(out).all()
    # near-rigid: high omega tracks the target closely
    assert np.linalg.norm(out - k, axis=2).mean() < 0.05


def test_bones_indices_in_range():
    for a, b in skeletons.BONES:
        assert 0 <= a < skeletons.N_JOINTS
        assert 0 <= b < skeletons.N_JOINTS
