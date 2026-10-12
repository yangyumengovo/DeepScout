from deepscout.llm import DiskCacheClient, LLMResponse


class CountingClient:
    name = "counting"

    def __init__(self):
        self.model = "m"
        self.n = 0

    def complete(self, messages, tools=None, temperature=0.3):
        self.n += 1
        return LLMResponse(
            content=f"r{self.n}", model="m", usage={"prompt_tokens": 1, "completion_tokens": 1}
        )


def test_cache_hits_on_same_request(tmp_path):
    inner = CountingClient()
    client = DiskCacheClient(inner, tmp_path / "c")
    msgs = [{"role": "user", "content": "hi"}]
    r1 = client.complete(msgs, temperature=0.0)
    r2 = client.complete(msgs, temperature=0.0)
    assert r1.content == "r1"
    assert r2.content == "r1"
    assert inner.n == 1


def test_cache_misses_on_different_request(tmp_path):
    inner = CountingClient()
    client = DiskCacheClient(inner, tmp_path / "c")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0)
    client.complete([{"role": "user", "content": "different"}], temperature=0.0)
    assert inner.n == 2


def test_cache_can_be_disabled(tmp_path):
    inner = CountingClient()
    client = DiskCacheClient(inner, tmp_path / "c", enabled=False)
    msgs = [{"role": "user", "content": "hi"}]
    client.complete(msgs, temperature=0.0)
    client.complete(msgs, temperature=0.0)
    assert inner.n == 2
