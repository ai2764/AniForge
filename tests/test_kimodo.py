from pipeline.kimodo import build_kimodo_graph

def test_graph_wires_text_and_sampler():
    g = build_kimodo_graph("A person waves.", duration=3.0, seed=42,
                            model="Kimodo-SOMA-RP-v1", steps=50)
    assert g["1"]["class_type"] == "Kimodo_LoadModel"
    assert g["1"]["inputs"]["model"] == "Kimodo-SOMA-RP-v1"
    assert g["2"]["inputs"]["prompt"] == "A person waves."
    assert g["2"]["inputs"]["model"] == ["1", 0]
    assert g["3"]["inputs"]["conditioning"] == ["2", 0]
    assert g["3"]["inputs"]["duration"] == 3.0
    assert g["5"]["class_type"] == "Kimodo_SaveNPZ"
