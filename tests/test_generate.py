from pipeline.generate import plan_steps, sanitize_action, DEFAULT_IDLE_PROMPT


def test_plan_steps_selects_overshoot():
    assert plan_steps(set()) == {"joint": False, "time": False}
    assert plan_steps({"joint"}) == {"joint": True, "time": False}
    assert plan_steps({"joint", "time"}) == {"joint": True, "time": True}


def test_sanitize_drops_turning():
    assert "turn" not in sanitize_action("she turns and waves").lower()


def test_default_idle_is_in_place():
    assert "in place" in DEFAULT_IDLE_PROMPT.lower()
