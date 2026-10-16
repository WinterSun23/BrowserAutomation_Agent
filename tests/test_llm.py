import pytest
from sitegraph.llm import FakeLLM, parse_json, complete_json


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure! Here it is: {"a": 3} hope that helps') == {"a": 3}
    assert parse_json('["x", "y"]') == ["x", "y"]
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_complete_json_falls_back_to_default():
    llm = FakeLLM(lambda s, u: "I cannot comply")
    assert complete_json(llm, "sys", "user", default={"ok": False}) == {"ok": False}
    assert llm.calls and "JSON" in llm.calls[0][0]
