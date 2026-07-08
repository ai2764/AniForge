from pipeline.spring_time_remap import remap_indices


def test_chosen_params_small_overshoot():
    idx = remap_indices(105, 24, 0.42, 4.2, 2.4, 1.15)
    back = max((idx[i-1]-idx[i]) for i in range(1, len(idx)))
    assert 1.0 < back < 3.0            # small overshoot, not a full replay
    assert idx[0] == 0 and idx[-1] <= 104
