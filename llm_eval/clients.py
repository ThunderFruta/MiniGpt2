from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str | None = None
    provider: str = "openai_compatible"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int | None = None
    price_per_1k_prompt_tokens: float | None = None
    price_per_1k_completion_tokens: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    model_name: str
    output_text: str
    latency_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: dict[str, Any] | None = None


class ClientError(RuntimeError):
    pass


def load_model_configs(path: str | Path) -> list[ModelConfig]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model config must contain a non-empty 'models' list")

    configs: list[ModelConfig] = []
    for index, item in enumerate(models, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be an object")
        try:
            configs.append(
                ModelConfig(
                    name=str(item["name"]),
                    model=str(item["model"]),
                    base_url=str(item["base_url"]),
                    api_key_env=item.get("api_key_env"),
                    provider=str(item.get("provider", "openai_compatible")),
                    timeout_sec=float(item.get("timeout_sec", 60.0)),
                    temperature=float(item.get("temperature", 0.0)),
                    max_tokens=item.get("max_tokens"),
                    price_per_1k_prompt_tokens=item.get("price_per_1k_prompt_tokens"),
                    price_per_1k_completion_tokens=item.get("price_per_1k_completion_tokens"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"models[{index}] missing required field: {exc.args[0]}") from exc
    return configs


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        if config.provider != "openai_compatible":
            raise ValueError(f"unsupported provider '{config.provider}'")
        self.config = config

    def generate(self, prompt: str) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        data = json.dumps(payload).encode("utf-8")
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ClientError(f"missing API key env var '{self.config.api_key_env}'")
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.config.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClientError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise ClientError(str(exc.reason)) from exc
        except TimeoutError as exc:
            raise ClientError("request timed out") from exc

        latency = time.perf_counter() - started
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ClientError("response was not valid JSON") from exc

        return _parse_openai_compatible_response(self.config.name, raw, latency)


def _parse_openai_compatible_response(
    model_name: str,
    raw: dict[str, Any],
    latency_sec: float,
) -> ModelResponse:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ClientError("response missing choices")

    first = choices[0]
    output_text = ""
    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            output_text = content if isinstance(content, str) else ""
        if not output_text and isinstance(first.get("text"), str):
            output_text = first["text"]
    if not output_text:
        raise ClientError("response missing output text")

    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return ModelResponse(
        model_name=model_name,
        output_text=output_text,
        latency_sec=latency_sec,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_response=raw,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

