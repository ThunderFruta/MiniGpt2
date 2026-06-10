import json
from io import BytesIO
from urllib import error

import pytest

from llm_eval.clients import (
    ClientError,
    ModelConfig,
    OpenAICompatibleClient,
    _parse_openai_compatible_response,
    load_model_configs,
)


def test_load_model_configs(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "local",
                        "model": "tiny",
                        "base_url": "http://127.0.0.1:8000/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    configs = load_model_configs(path)

    assert configs[0].name == "local"
    assert configs[0].provider == "openai_compatible"


def test_parse_openai_compatible_response_with_usage():
    response = _parse_openai_compatible_response(
        "m1",
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        0.25,
    )

    assert response.output_text == "hello"
    assert response.total_tokens == 5


def test_parse_openai_compatible_response_computes_total_tokens_when_missing():
    response = _parse_openai_compatible_response(
        "m1",
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        0.25,
    )

    assert response.total_tokens == 5


def test_parse_openai_compatible_response_rejects_missing_choices():
    with pytest.raises(ClientError, match="choices"):
        _parse_openai_compatible_response("m1", {"choices": []}, 0.1)


def test_client_errors_when_api_key_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        ModelConfig(
            name="api",
            model="remote",
            base_url="https://example.invalid/v1",
            api_key_env="MISSING_API_KEY",
        )
    )

    with pytest.raises(ClientError, match="MISSING_API_KEY"):
        client.generate("hello")


def test_client_wraps_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise error.HTTPError(
            url="http://test/v1/chat/completions",
            code=500,
            msg="server error",
            hdrs=None,
            fp=BytesIO(b'{"error":"boom"}'),
        )

    monkeypatch.setattr("llm_eval.clients.request.urlopen", raise_http_error)
    client = OpenAICompatibleClient(ModelConfig(name="local", model="m", base_url="http://test/v1"))

    with pytest.raises(ClientError, match="HTTP 500"):
        client.generate("hello")

