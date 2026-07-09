from server.app import parse_generate_form, required_nodes_present


def test_parse_form_overshoot_multi():
    d = parse_generate_form({"action_prompt": "wave", "idle_prompt": "",
                             "overshoot": ["joint", "time"]})
    assert d["action_prompt"] == "wave"
    assert d["idle_prompt"] is None
    assert d["overshoot"] == {"joint", "time"}
    assert d["pose_mode"] == "standing"


def test_parse_form_pose_mode():
    d = parse_generate_form({"action_prompt": "wave", "pose_mode": "sitting"})
    assert d["pose_mode"] == "sitting"
    d = parse_generate_form({"action_prompt": "wave", "pose_mode": "lying"})
    assert d["pose_mode"] == "lying"
    d = parse_generate_form({"action_prompt": "wave", "pose_mode": "nope"})
    assert d["pose_mode"] == "standing"


def test_parse_form_scale():
    d = parse_generate_form({"action_prompt": "wave"})
    assert d["scale"] == 1.0
    d = parse_generate_form({"action_prompt": "wave", "scale": "0.5"})
    assert d["scale"] == 0.5
    d = parse_generate_form({"action_prompt": "wave", "scale": "2"})
    assert d["scale"] == 1.0
    d = parse_generate_form({"action_prompt": "wave", "scale": "0.1"})
    assert d["scale"] == 0.25


def test_required_nodes_check():
    assert required_nodes_present({"Kimodo_Sampler": {}, "WanSCAILToVideo": {}})
    assert not required_nodes_present({"Kimodo_Sampler": {}})
