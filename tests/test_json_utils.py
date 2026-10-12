from deepscout.json_utils import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_json_wrapped_in_prose():
    assert extract_json('好的，结果如下：{"a": 1} 希望有帮助') == {"a": 1}


def test_first_of_multiple():
    assert extract_json('{"a": 1} 然后又 {"b": 2}') == {"a": 1}


def test_invalid_returns_none():
    assert extract_json("完全没有 JSON") is None
    assert extract_json("") is None
    assert extract_json(None) is None
