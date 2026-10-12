import json

from deepscout.search import MockSearchProvider
from deepscout.tools import ToolBox


def make_box() -> ToolBox:
    return ToolBox(MockSearchProvider())


def test_schema_generation():
    schemas = make_box().schemas(["web_search", "fetch_page"])
    assert [s["function"]["name"] for s in schemas] == ["web_search", "fetch_page"]
    params = schemas[0]["function"]["parameters"]
    assert params["properties"]["query"]["type"] == "string"
    assert "required" in params


def test_web_search_mock_returns_results():
    out = json.loads(make_box().call("web_search", query="长上下文 扩展 技术"))
    assert isinstance(out, list) and len(out) >= 1
    assert "url" in out[0] and "title" in out[0]


def test_args_validation_returns_error_not_exception():
    out = json.loads(make_box().call("web_search", query="x", max_results=99))  # 超过 le=10
    assert "error" in out


def test_fetch_demo_page():
    out = json.loads(make_box().call("fetch_page", url="demo://llm-context/rope"))
    assert out.get("title") and out.get("text") and "RoPE" in out["text"]


def test_fetch_unknown_demo_page():
    out = json.loads(make_box().call("fetch_page", url="demo://llm-context/nope"))
    assert "error" in out


def test_unknown_tool():
    out = json.loads(make_box().call("not_a_tool"))
    assert "error" in out
