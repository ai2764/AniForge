import json, io
from pipeline.comfy import ComfyClient

class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def make_opener(script):
    calls = []
    def opener(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        return FakeResp(json.dumps(script[len(calls)-1]).encode())
    opener.calls = calls
    return opener

def test_submit_returns_prompt_id():
    opener = make_opener([{"prompt_id": "abc"}])
    c = ComfyClient(opener=opener)
    assert c.submit({"1": {}}, "cid") == "abc"

def test_wait_returns_history_entry():
    opener = make_opener([{"pid": {"status": {"status_str": "success"}, "outputs": {}}}])
    c = ComfyClient(opener=opener)
    entry = c.wait("pid", timeout=5)
    assert entry["status"]["status_str"] == "success"
