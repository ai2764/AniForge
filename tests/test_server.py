from server.app import parse_generate_form, required_nodes_present


def test_parse_form_overshoot_multi():
    d = parse_generate_form({"action_prompt": "wave", "idle_prompt": "",
                             "overshoot": ["joint", "time"]})
    assert d["action_prompt"] == "wave"
    assert d["idle_prompt"] is None
    assert d["overshoot"] == {"joint", "time"}


def test_required_nodes_check():
    assert required_nodes_present({"Kimodo_Sampler": {}, "WanSCAILToVideo": {}})
    assert not required_nodes_present({"Kimodo_Sampler": {}})
