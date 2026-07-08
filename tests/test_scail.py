import json
from pathlib import Path
from pipeline.scail import build_scail_graph

TPL = json.loads(Path("pipeline/assets/scail2_video.api.json").read_text(encoding="utf-8"))


def test_scail_graph_sets_guide_ref_and_size():
    g = build_scail_graph(TPL, "guide.mp4", "ref.png", 480, 832, 105, 0.9, 42, 6, "mp_body", "a person")
    assert g["9"]["inputs"]["image"] == "ref.png"
    assert g["11"]["inputs"]["file"] == "guide.mp4"
    assert g["13"]["inputs"]["width"] == 480 and g["13"]["inputs"]["length"] == 105
    assert g["14"]["inputs"]["seed"] == 42
    assert g["17"]["inputs"]["filename_prefix"] == "mp_body"


def test_build_scail_graph_does_not_mutate_template():
    before = json.dumps(TPL)
    build_scail_graph(TPL, "guide.mp4", "ref.png", 480, 832, 105, 0.9, 42, 6, "mp_body", "a person")
    after = json.dumps(TPL)
    assert before == after
