"""Tests for openkb.okf.llm - model normalization, JSON resilience, evidence.

Covers (#10): ``base_url`` + a bare model -> ``openai/<model>``.
(#11): a model already prefixed with ``openai/`` is left unchanged.
Plus: evidence-less extracts are dropped (with a warning), and malformed /
fenced JSON is repaired (the json_repair resilience pattern).

LLM mocking uses Strategy B: monkeypatch ``LLMClient.json_completion`` (the
thin wrapper) rather than ``litellm.completion`` so the tests never depend on
litellm's internal retry/wrapper behavior. ``json_completion`` returns the
raw content string the caller would otherwise parse.
"""

from __future__ import annotations

from unittest.mock import patch

from openkb.okf.llm import (
    LLMClient,
    LLMConfig,
    _parse_json,
    extract,
    normalize_model,
    resolve_config,
)
from openkb.okf.markdown import split_sections

_MD = "# T\n\n## Intro\n\nHello world of transformers.\n"


def _extract_with_json(client: LLMClient, md: str, sections, *, responses):
    """Run ``extract`` with ``json_completion`` returning ``responses`` in order.

    Strategy B: monkeypatch the thin wrapper so the test never touches litellm
    (no real call, no retry behavior). Each entry in ``responses`` is either a
    ``str`` (returned as the raw content) or an Exception (raised by the call).
    """
    it = iter(responses)

    def fake_json(system: str, user: str) -> str:
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(LLMClient, "json_completion", side_effect=fake_json):
        return extract(
            client,
            md,
            sections,
            language="en",
            max_concepts=12,
            max_entities=12,
        )


# --- model normalization (#10, #11) ---


def test_normalize_adds_openai_prefix_when_base_url_set():
    assert normalize_model("xopkimik26", "https://api.example.com/v1") == "openai/xopkimik26"


def test_normalize_keeps_existing_prefix():
    assert normalize_model("openai/xopkimik26", "https://api.example.com/v1") == "openai/xopkimik26"
    assert (
        normalize_model("anthropic/claude-x", "https://api.example.com/v1") == "anthropic/claude-x"
    )


def test_normalize_no_prefix_when_no_base_url():
    assert normalize_model("gpt-4o", None) == "gpt-4o"
    assert normalize_model("gpt-4o", "") == "gpt-4o"


def test_normalize_none_passthrough():
    assert normalize_model(None, "https://api.example.com/v1") is None


def test_client_records_normalized_model():
    cfg = LLMConfig(base_url="https://api.example.com/v1", model="xopkimik26", api_key="sk-x")
    client = LLMClient(cfg)
    assert client.model == "openai/xopkimik26"
    # an already-prefixed model is unchanged at the client level
    cfg2 = LLMConfig(
        base_url="https://api.example.com/v1",
        model="openai/xopkimik26",
        api_key="sk-x",
    )
    assert LLMClient(cfg2).model == "openai/xopkimik26"


# --- config resolution: CLI > env > .env > default ---


def test_resolve_config_cli_wins():
    env = {"OPENKB_LLM_MODEL": "env-model"}
    dotenv = {"OPENKB_LLM_MODEL": "dotenv-model"}
    cfg = resolve_config(
        base_url="cli-url",
        model="cli-model",
        api_key="cli-key",
        timeout=30,
        env=env,
        dotenv_values=dotenv,
    )
    assert cfg.model == "cli-model"
    assert cfg.base_url == "cli-url"
    assert cfg.api_key == "cli-key"
    assert cfg.timeout == 30


def test_resolve_config_env_over_dotenv():
    env = {"OPENKB_LLM_MODEL": "env-model", "OPENKB_LLM_TIMEOUT": "90"}
    dotenv = {"OPENKB_LLM_MODEL": "dotenv-model", "OPENKB_LLM_BASE_URL": "https://dotenv/v1"}
    cfg = resolve_config(
        base_url=None, model=None, api_key=None, timeout=None, env=env, dotenv_values=dotenv
    )
    assert cfg.model == "env-model"
    assert cfg.timeout == 90
    assert cfg.base_url == "https://dotenv/v1"  # only dotenv has it


def test_resolve_config_defaults():
    cfg = resolve_config(
        base_url=None, model=None, api_key=None, timeout=None, env={}, dotenv_values={}
    )
    assert cfg.model is None
    assert cfg.api_key is None
    assert cfg.timeout == 120.0  # default


# --- JSON resilience (json_repair) ---


def test_parse_json_plain():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_prose_wrapped():
    # json_repair salvages JSON embedded in prose
    result = _parse_json('Here is the result: {"a": 1, "b": [1,2]} done')
    assert result == {"a": 1, "b": [1, 2]}


def test_parse_json_trailing_comma_repaired():
    # json_repair handles trailing commas (common LLM artifact)
    result = _parse_json('{"concepts": [{"name": "x"},]}')
    assert isinstance(result, dict)


def test_parse_json_rejects_scalar():
    import pytest

    with pytest.raises(ValueError):
        _parse_json('"just a string"')
    with pytest.raises(ValueError):
        _parse_json("42")


# --- evidence dropping + extract orchestration ---


def test_evidence_less_concept_dropped_with_warning():
    sections = split_sections(_MD)
    responses = [
        '{"summary": "a summary"}',
        '{"concepts": ['
        '{"name": "Good", "description": "d", "evidence": '
        '{"heading_path": "Intro", "line_start": 3, "line_end": 5}},'
        '{"name": "NoEv", "description": "d"}'
        "]}",
        '{"entities": []}',
        '{"relations": []}',
    ]
    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key="sk-x"))
    out = _extract_with_json(client, _MD, sections, responses=responses)
    assert len(out.concepts) == 1
    assert out.concepts[0].name == "Good"
    assert any("dropped 1" in w and "concept" in w for w in out.warnings)


def test_extract_invalid_evidence_dropped():
    sections = split_sections(_MD)
    # evidence with line_start > line_end is invalid
    responses = [
        '{"summary": "s"}',
        '{"concepts": [{"name": "Bad", "description": "d", "evidence": '
        '{"heading_path": "Intro", "line_start": 10, "line_end": 5}}]}',
        '{"entities": []}',
        '{"relations": []}',
    ]
    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key="sk-x"))
    out = _extract_with_json(client, _MD, sections, responses=responses)
    assert out.concepts == []  # invalid evidence dropped
    assert any("dropped" in w for w in out.warnings)


def test_extract_one_call_failure_does_not_abort_others():
    """A broken concepts call records a warning but entities/relations still run."""
    sections = split_sections(_MD)
    responses = [  # summary ok, concepts BOOM, entities ok, relations ok
        '{"summary": "s"}',
        RuntimeError("concepts endpoint exploded"),
        '{"entities": [{"name": "E", "type": "org", "description": "d", '
        '"evidence": {"heading_path": "Intro", "line_start": 3, "line_end": 5}}]}',
        '{"relations": []}',
    ]
    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key="sk-x"))
    out = _extract_with_json(client, _MD, sections, responses=responses)
    assert out.summary == "s"
    assert len(out.entities) == 1  # entities still extracted
    assert any("concepts" in w and "failed" in w for w in out.warnings)


def test_api_key_not_leaked_in_extracts():
    """The api_key is never carried into Extracts (it's transient, LLM-only).

    We verify the extract output contains no api_key string even though the
    client holds one. ``json_completion`` is monkeypatched so the key never
    reaches a real call; the assertion is that the Extracts dataclasses don't
    carry it.
    """
    sections = split_sections(_MD)
    secret = "sk-SECRET-999"
    responses = [
        '{"summary": "s"}',
        '{"concepts": []}',
        '{"entities": []}',
        '{"relations": []}',
    ]
    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key=secret))
    out = _extract_with_json(client, _MD, sections, responses=responses)
    blob = out.summary + "".join(out.warnings)
    for c in out.concepts:
        blob += c.name + c.description
    for e in out.entities:
        blob += e.name + e.description
    assert secret not in blob


def test_json_completion_passes_openai_compat_kwargs():
    """json_completion passes model/api_base/api_key/temperature/response_format.

    Patches ``litellm.completion`` (no exceptions -> no retry interference) and
    captures the kwargs the wrapper builds, so we assert the OpenAI-compatible
    call shape is exactly what the spec requires.
    """
    from unittest.mock import MagicMock

    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"summary": "ok"}'))]
        return resp

    client = LLMClient(
        LLMConfig(
            base_url="https://api.example.com/v1", model="mymodel", api_key="sk-x", timeout=45
        )
    )
    with patch("litellm.completion", side_effect=fake_completion):
        content = client.json_completion("sys", "user")
    assert content == '{"summary": "ok"}'
    assert captured["model"] == "openai/mymodel"  # normalized
    assert captured["api_base"] == "https://api.example.com/v1"
    assert captured["api_key"] == "sk-x"
    assert captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 45


def test_llmclient_requires_model():
    import pytest

    cfg = LLMConfig(base_url="https://x/v1", model=None, api_key="sk-x")
    with pytest.raises(ValueError):
        LLMClient(cfg)


def test_relations_prompt_receives_prior_concepts_and_entities():
    """The relations call must be passed the prior concepts + entities so it
    picks subject/object from those names rather than re-inferring them.
    """
    sections = split_sections(_MD)
    captured: list[str] = []

    responses = [
        '{"summary": "s"}',
        '{"concepts": [{"name": "Attention", "description": "d", "evidence": '
        '{"heading_path": "Intro", "line_start": 3, "line_end": 5}}]}',
        '{"entities": [{"name": "Google", "type": "organization", "description": "d", '
        '"evidence": {"heading_path": "Intro", "line_start": 3, "line_end": 5}}]}',
        '{"relations": []}',
    ]
    it = iter(responses)

    def fake_json(system: str, user: str) -> str:
        captured.append(user)
        return next(it)

    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key="sk-x"))
    with patch.object(LLMClient, "json_completion", side_effect=fake_json):
        extract(client, _MD, sections, language="en", max_concepts=12, max_entities=12)

    # 4 calls: summary, concepts, entities, relations
    assert len(captured) == 4
    concepts_call = captured[1]
    entities_call = captured[2]
    relations_call = captured[3]
    # concepts call sees the prior summary
    assert "Prior summary" in concepts_call
    # entities call sees prior summary + prior concepts
    assert "Attention" in entities_call
    assert "Prior summary" in entities_call
    # relations call sees prior concepts AND entities
    assert "Attention" in relations_call
    assert "Google" in relations_call
    assert "MUST be exact names from the prior" in relations_call


def test_relation_with_unknown_subject_or_object_is_dropped():
    sections = split_sections(_MD)
    responses = [
        '{"summary": "s"}',
        '{"concepts": [{"name": "Attention", "description": "d", "evidence": '
        '{"heading_path": "Intro", "line_start": 3, "line_end": 5}}]}',
        '{"entities": []}',
        '{"relations": [{"subject": "Unknown", "relation": "mentions", '
        '"object": "Attention", "evidence": '
        '{"heading_path": "Intro", "line_start": 3, "line_end": 5}}]}',
    ]
    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key="sk-x"))

    out = _extract_with_json(client, _MD, sections, responses=responses)

    assert out.relations == []
    assert any("subject/object" in w for w in out.warnings)


def test_api_key_redacted_from_exception_warning():
    """A thrown exception whose message echoes the api_key is redacted in warnings."""
    sections = split_sections(_MD)
    secret = "sk-SECRET-REDACT-ME-1234567890"

    class _Boom(Exception):
        pass

    responses = [
        _Boom(f"request failed with key {secret}"),
        '{"concepts": []}',
        '{"entities": []}',
        '{"relations": []}',
    ]
    it = iter(responses)

    def fake_json(system: str, user: str) -> str:
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    client = LLMClient(LLMConfig(base_url="https://x/v1", model="m", api_key=secret))
    with patch.object(LLMClient, "json_completion", side_effect=fake_json):
        out = extract(client, _MD, sections, language="en", max_concepts=12, max_entities=12)

    # the summary failed -> a warning was recorded; the secret must be gone
    joined = "\n".join(out.warnings)
    assert secret not in joined, f"api_key leaked into warnings: {joined}"
    assert "[REDACTED]" in joined
