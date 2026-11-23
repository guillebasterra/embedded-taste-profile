"""VLM factory and the Anthropic implementation.

`get_vlm(settings)` is the only entry point other modules use. The Anthropic SDK is imported
lazily, and `build_request` is a pure function so the request shape is testable offline.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from taste.config import Settings
from taste.interfaces import VLM, FakeVLM

T = TypeVar("T", bound=BaseModel)

MAX_IMAGE_EDGE = 1024
JPEG_QUALITY = 85
DEFAULT_MAX_TOKENS = 4096


class VLMError(RuntimeError):
    """Configuration or generation failure with a message fit to show the user."""


def get_vlm(settings: Settings) -> VLM:
    kind = (settings.vlm or "").lower()
    if kind == "fake":
        return FakeVLM()
    if kind == "anthropic":
        if not settings.anthropic_api_key:
            raise VLMError(
                "ANTHROPIC_API_KEY is not set. Export it or add it to "
                f"{settings.home / '.env'}, or set TASTE_VLM=fake for a dry run."
            )
        return AnthropicVLM(model=settings.vlm_model, api_key=settings.anthropic_api_key)
    raise VLMError(f"Unknown TASTE_VLM {settings.vlm!r}: expected 'anthropic' or 'fake'.")


# ---------------------------------------------------------------------------
# Request building (pure, no network)
# ---------------------------------------------------------------------------

def encode_image(path: Path, max_edge: int = MAX_IMAGE_EDGE) -> str:
    """Base64 JPEG, long edge at most `max_edge`, EXIF rotation applied, alpha on white."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def tool_name_for(schema: type[BaseModel]) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", schema.__name__).lower()
    return f"record_{snake}"[:64]


def build_request(model: str, prompt: str, schema: type[BaseModel],
                  images: list[Path] | None = None, system: str | None = None,
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    """Keyword arguments for `client.messages.create`.

    Images come first, each preceded by a text block "Image N:" (1-based, in list order), so
    prompts can refer to images by number. The answer comes back through one tool call whose
    input schema is the pydantic JSON schema. tool_choice is "auto" plus an instruction naming
    the tool, because Sonnet 5.5 and Opus 5.5 reject a forced tool_choice with a 400.
    """
    content: list[dict[str, Any]] = []
    for i, path in enumerate(images or [], start=1):
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": encode_image(Path(path))},
        })
    name = tool_name_for(schema)
    content.append({"type": "text", "text": prompt})
    content.append({"type": "text", "text": f"Answer only by calling the {name} tool."})

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "tools": [{
            "name": name,
            "description": (
                f"Record the final {schema.__name__}. Call exactly once with every required "
                "field filled in."
            ),
            "input_schema": schema.model_json_schema(),
        }],
        "tool_choice": {"type": "auto"},
    }
    if system:
        request["system"] = system
    return request


def _block_to_dict(block: Any) -> dict[str, Any] | None:
    """Turn a response content block into the dict form accepted back by the API."""
    btype = getattr(block, "type", None)
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if btype == "text" and getattr(block, "text", ""):
        return {"type": "text", "text": block.text}
    return None


def build_retry_messages(messages: list[dict[str, Any]], response: Any,
                         error: str) -> list[dict[str, Any]]:
    """Original messages + the failed assistant turn + the validation error fed back."""
    blocks = [d for b in getattr(response, "content", []) or [] if (d := _block_to_dict(b))]
    tool_use = next((b for b in blocks if b["type"] == "tool_use"), None)
    fix = ("The tool input failed validation. Call the tool again with a corrected, complete "
           f"input.\n\nValidation error:\n{error}")
    if tool_use is None:
        follow_up: list[dict[str, Any]] = [{"type": "text", "text": fix}]
        blocks = blocks or [{"type": "text", "text": "(no tool call)"}]
    else:
        follow_up = [{"type": "tool_result", "tool_use_id": tool_use["id"],
                      "is_error": True, "content": fix}]
    return [*messages, {"role": "assistant", "content": blocks},
            {"role": "user", "content": follow_up}]


def parse_response(response: Any, schema: type[T]) -> T:
    """Validate the forced tool call. Raises ValueError/ValidationError on a bad answer."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise ValueError("the answer was cut off by max_tokens; write a shorter one")
            return schema.model_validate(block.input)
    raise ValueError("the response contained no tool call")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AnthropicVLM:
    """`VLM` backed by the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - it is a hard dependency
                raise VLMError("The 'anthropic' package is not installed.") from e
            self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=4)
        return self._client

    def generate(self, prompt: str, schema: type[T], images: list[Path] | None = None,
                 system: str | None = None) -> T:
        request = build_request(self.model, prompt, schema, images, system, self.max_tokens)
        response = self.client.messages.create(**request)
        try:
            return parse_response(response, schema)
        except (ValidationError, ValueError) as first:
            retry = dict(request)
            retry["messages"] = build_retry_messages(request["messages"], response, str(first))
            response = self.client.messages.create(**retry)
            try:
                return parse_response(response, schema)
            except (ValidationError, ValueError) as second:
                raise VLMError(
                    f"{self.model} did not return a valid {schema.__name__} after one retry: "
                    f"{second}"
                ) from second
