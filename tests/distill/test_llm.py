from __future__ import annotations

import base64
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from taste.distill import llm
from taste.distill.schemas import WebCardDraft
from taste.interfaces import FakeVLM
from taste.models import ClusterCard

VALID = {"name": "Ink grids", "one_liner": "o", "principles": ["p"], "do": ["d"],
         "avoid": ["a"], "contrast_note": "c"}


def test_get_vlm_fake(settings):
    assert isinstance(llm.get_vlm(settings), FakeVLM)


def test_get_vlm_anthropic_needs_key(settings):
    with pytest.raises(llm.VLMError, match="ANTHROPIC_API_KEY"):
        llm.get_vlm(replace(settings, vlm="anthropic", anthropic_api_key=None))
    vlm = llm.get_vlm(replace(settings, vlm="anthropic", anthropic_api_key="k",
                              vlm_model="some-model"))
    assert isinstance(vlm, llm.AnthropicVLM) and vlm.model == vlm.name == "some-model"
    with pytest.raises(llm.VLMError, match="Unknown TASTE_VLM"):
        llm.get_vlm(replace(settings, vlm="gpt"))


def test_build_request_structure(tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGBA", (3000, 1500), (10, 20, 30, 128)).save(big)
    small = tmp_path / "small.jpg"
    Image.new("RGB", (64, 48), (200, 0, 0)).save(small)

    req = llm.build_request("some-model", "PROMPT", WebCardDraft, [big, small], system="SYS")
    assert req["model"] == "some-model" and req["system"] == "SYS" and req["max_tokens"] > 0
    assert len(req["messages"]) == 1 and req["messages"][0]["role"] == "user"
    content = req["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == "Image 1:" and content[2]["text"] == "Image 2:"
    assert content[-1]["text"] == "PROMPT"
    src = content[1]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(src["data"]))) as im:
        assert im.format == "JPEG" and im.size == (1024, 512)
    with Image.open(io.BytesIO(base64.b64decode(content[3]["source"]["data"]))) as im:
        assert im.size == (64, 48)  # never upscaled

    assert len(req["tools"]) == 1
    tool = req["tools"][0]
    assert tool["name"] == "record_web_card_draft"
    assert tool["input_schema"] == WebCardDraft.model_json_schema()
    assert req["tool_choice"] == {"type": "tool", "name": tool["name"]}

    text_only = llm.build_request("m", "P", ClusterCard)
    assert "system" not in text_only
    assert text_only["messages"][0]["content"] == [{"type": "text", "text": "P"}]


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def tool_response(data, stop_reason="tool_use"):
    return SimpleNamespace(stop_reason=stop_reason, content=[
        SimpleNamespace(type="tool_use", id="tu_1", name="record_web_card_draft", input=data)])


def test_generate_validates(tmp_path):
    client = StubClient([tool_response(VALID)])
    out = llm.AnthropicVLM("m", client=client).generate("P", WebCardDraft)
    assert isinstance(out, WebCardDraft) and out.name == "Ink grids" and out.tokens is None
    assert len(client.requests) == 1


def test_generate_retries_once_with_the_validation_error():
    bad = {k: v for k, v in VALID.items() if k != "avoid"}
    client = StubClient([tool_response(bad), tool_response(VALID)])
    out = llm.AnthropicVLM("m", client=client).generate("P", WebCardDraft, system="S")
    assert out.avoid == ["a"] and len(client.requests) == 2
    retry = client.requests[1]
    assert retry["tools"] == client.requests[0]["tools"] and retry["system"] == "S"
    assert [m["role"] for m in retry["messages"]] == ["user", "assistant", "user"]
    assert retry["messages"][1]["content"] == [
        {"type": "tool_use", "id": "tu_1", "name": "record_web_card_draft", "input": bad}]
    result = retry["messages"][2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1"
    assert result["is_error"] is True and "avoid" in result["content"]


def test_generate_gives_up_after_one_retry():
    no_tool = SimpleNamespace(stop_reason="end_turn",
                              content=[SimpleNamespace(type="text", text="sorry")])
    client = StubClient([no_tool, tool_response({"name": 1})])
    with pytest.raises(llm.VLMError, match="after one retry"):
        llm.AnthropicVLM("m", client=client).generate("P", WebCardDraft)
    assert len(client.requests) == 2
    assert client.requests[1]["messages"][2]["content"][0]["type"] == "text"


def test_truncated_answer_counts_as_invalid():
    client = StubClient([tool_response(VALID, stop_reason="max_tokens"), tool_response(VALID)])
    llm.AnthropicVLM("m", client=client).generate("P", WebCardDraft)
    assert "max_tokens" in client.requests[1]["messages"][2]["content"][0]["content"]
