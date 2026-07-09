"""LLM client for the OKF compiler: a self-contained OpenAI-compatible 3-param
client (base_url / model / api_key / timeout), independent of the OpenKB KB
config (``openkb.config``) and of ``cli._setup_llm_key``.

Resolution order for each parameter: explicit CLI arg > ``OPENKB_LLM_*`` env
var > ``.env`` file > default. ``--no-llm`` callers never construct this
client, so config validation is only enforced when an LLM call is about to
happen.

The ``api_key`` is passed straight to LiteLLM and is NEVER written to any
output file, manifest, log, or zip. Only the model name is recorded (in the
manifest ``compiler.model`` field).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openkb.okf.prompts import (
    concepts_messages,
    entities_messages,
    relations_messages,
    summary_messages,
)
from openkb.okf.schema import (
    ConceptExtract,
    EntityExtract,
    Evidence,
    Extracts,
    ProposedEdge,
    SectionSpec,
    validate_evidence,
)

logger = logging.getLogger(__name__)

_JSON_RESPONSE_FORMAT = {"type": "json_object"}

# Env var names - the OKF LLM config namespace, deliberately separate from the
# OpenKB KB-level ``LLM_API_KEY`` so an OKF compile doesn't require a KB.
ENV_BASE_URL = "OPENKB_LLM_BASE_URL"
ENV_MODEL = "OPENKB_LLM_MODEL"
ENV_API_KEY = "OPENKB_LLM_API_KEY"
ENV_TIMEOUT = "OPENKB_LLM_TIMEOUT"

_DEFAULT_TIMEOUT = 120.0


# Redaction patterns. ``redact_secrets`` is applied to every string that might
# reach a persisted surface (manifest warnings, batch_report errors, log.md,
# CLI output) so an exception carrying a key/token never leaks out.
_SK_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_BEARER_PATTERN = re.compile(
    r"(Authorization|Bearer)\s*[:=]?\s*[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE
)
_REDACTED = "[REDACTED]"


def redact_secrets(text: str, api_key: str | None = None) -> str:
    """Strip secret-looking substrings from ``text``.

    Redacts, in order: the exact ``api_key`` value (when given), ``sk-...``
    token patterns, and ``Authorization``/``Bearer`` header values. Never
    raises - returns the input unchanged on any error. Idempotent on already
    redacted text (``[REDACTED]`` contains no ``sk-`` prefix).
    """
    if not isinstance(text, str):
        return text
    try:
        out = text
        if api_key and api_key.strip():
            out = out.replace(api_key, _REDACTED)
        out = _SK_PATTERN.sub(_REDACTED, out)
        out = _BEARER_PATTERN.sub(lambda m: m.group(1) + ": " + _REDACTED, out)
        return out
    except Exception:  # noqa: BLE001 - redaction must never raise
        return out


@dataclass
class LLMConfig:
    """Resolved 3-param LLM config. ``api_key`` is transient - never persisted."""

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = _DEFAULT_TIMEOUT

    def is_configured(self) -> bool:
        """True when enough is set to make a call (a model is required)."""
        return bool(self.model)


def resolve_config(
    *,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
    env: dict[str, str],
    dotenv_values: dict[str, str],
) -> LLMConfig:
    """Resolve a :class:`LLMConfig` by CLI > env > .env > default precedence.

    ``env`` is typically ``os.environ``; ``dotenv_values`` is the parsed
    ``.env`` mapping. Both are passed in (rather than read here) so the
    function is pure and testable without monkeypatching the environment.
    """

    # CLI > env > .env > default, first non-empty wins.
    def pick(cli: str | None, env_key: str) -> str | None:
        if cli and cli.strip():
            return cli.strip()
        env_val = env.get(env_key) or dotenv_values.get(env_key)
        if env_val and env_val.strip():
            return env_val.strip()
        return None

    resolved_base = pick(base_url, ENV_BASE_URL)
    resolved_model = pick(model, ENV_MODEL)
    resolved_key = pick(api_key, ENV_API_KEY)

    # Timeout: CLI float > env float > .env float > default.
    resolved_timeout = _DEFAULT_TIMEOUT
    if timeout is not None:
        resolved_timeout = float(timeout)
    else:
        raw_timeout = env.get(ENV_TIMEOUT) or dotenv_values.get(ENV_TIMEOUT)
        if raw_timeout:
            try:
                resolved_timeout = float(raw_timeout)
            except (TypeError, ValueError):
                logger.warning(
                    "invalid %s=%r; using default %s",
                    ENV_TIMEOUT,
                    raw_timeout,
                    _DEFAULT_TIMEOUT,
                )

    return LLMConfig(
        base_url=resolved_base,
        model=resolved_model,
        api_key=resolved_key,
        timeout=resolved_timeout,
    )


def normalize_model(model: str | None, base_url: str | None) -> str | None:
    """Add the ``openai/`` LiteLLM provider prefix when a custom ``base_url``
    is set and the model has no provider prefix.

    With ``base_url`` the target is an OpenAI-compatible endpoint, so LiteLLM
    needs the ``openai/`` prefix to route to the OpenAI client against that
    base. A model that already carries a provider prefix (``openai/xopkimik26``,
    ``anthropic/claude-...``) is left untouched so explicit routing wins.
    """
    if not model:
        return model
    model = model.strip()
    if "/" in model:
        return model
    if base_url and base_url.strip():
        return f"openai/{model}"
    return model


def _parse_json(text: str) -> list | dict:
    """Parse JSON from an LLM response, tolerating fences/prose/malformed JSON.

    Mirrors :func:`openkb.agent.compiler._parse_json` (PR d1d3f4d): strip a
    leading ``` ``` fence, repair with ``json_repair``, then require a
    dict/list shape so a JSON scalar is rejected.
    """
    from json_repair import repair_json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1 :] if first_nl != -1 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    result = json.loads(repair_json(cleaned.strip()))
    if not isinstance(result, (dict, list)):
        raise ValueError(f"Expected JSON object or array, got {type(result).__name__}")
    return result


def _evidence_from(obj: dict) -> Evidence | None:
    """Build an :class:`Evidence` from a raw dict; ``None`` if malformed."""
    ev = obj.get("evidence")
    if not isinstance(ev, dict):
        return None
    heading = ev.get("heading_path")
    ls = ev.get("line_start")
    le = ev.get("line_end")
    if not isinstance(heading, str) or not isinstance(ls, int) or not isinstance(le, int):
        return None
    return Evidence(heading_path=heading, line_start=ls, line_end=le)


class LLMClient:
    """Thin wrapper over ``litellm.completion`` for OpenAI-compatible calls.

    Constructed only when an LLM run is actually requested (``--no-llm``
    callers never reach here). The model is normalized at construction so the
    recorded manifest ``model`` reflects the normalized form.
    """

    def __init__(self, config: LLMConfig):
        if not config.is_configured():
            raise ValueError("LLMConfig is not configured: a model is required")
        self._config = config
        # Normalize once; record the normalized model for the manifest.
        self.model = normalize_model(config.model, config.base_url)

    @property
    def api_key(self) -> str | None:
        """The configured api_key (read-only; used only for redaction in warnings)."""
        return self._config.api_key

    def json_completion(self, system: str, user: str) -> str:
        """Make one JSON-mode completion; return the raw content string.

        Raises on transport/API errors so the caller can mark the extract
        failed. Never raises on shape problems - the caller parses via
        :func:`_parse_json` and drops malformed output.
        """
        import litellm

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": _JSON_RESPONSE_FORMAT,
        }
        if self._config.base_url:
            kwargs["api_base"] = self._config.base_url
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        if self._config.timeout:
            kwargs["timeout"] = self._config.timeout

        logger.debug("OKF LLM request: model=%s base=%s", self.model, self._config.base_url)
        response = litellm.completion(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def test(self) -> str:
        """Round-trip a trivial JSON request; returns the raw content.

        Used by ``openkb okf test-llm``. Raises on any error so the CLI can
        report failure.
        """
        return self.json_completion(
            "You are a connectivity test. Reply with valid JSON.",
            'Reply with the JSON object: {"ok": true}',
        )


def extract(
    client: LLMClient,
    markdown: str,
    sections: list[SectionSpec],
    *,
    language: str,
    max_concepts: int,
    max_entities: int,
) -> Extracts:
    """Run the four extracts sequentially and return validated :class:`Extracts`.

    Order: summary -> concepts -> entities -> relations, with each step's
    output threaded forward (concepts see the summary; entities see summary +
    concepts; relations see summary + concepts + entities and must pick
    subject/object from those names rather than re-inferring them).

    Each call is awaited before the next (single in-flight call). Any call that
    fails or yields no valid items records a warning and the extract continues -
    a broken concept call must not prevent the summary from being written.

    Evidence is validated against the section map (heading_path must match a
    known section and the line range must fall within it); invalid extracts are
    dropped with a warning. Secrets are redacted from every warning string so a
    thrown exception can never leak the api_key into the manifest.
    """
    out = Extracts()
    warnings = out.warnings
    total_lines = max(markdown.count("\n") + 1, 1)
    api_key = client.api_key if hasattr(client, "api_key") else None

    def _record_warning(msg: str) -> None:
        warnings.append(redact_secrets(msg, api_key))

    # 1. Summary (whole-article, no evidence required).
    try:
        sys_msg, user_msg = summary_messages(markdown, sections, language)
        raw = client.json_completion(sys_msg, user_msg)
        summary = _parse_summary(raw)
        out.summary = summary or ""
        if not out.summary:
            _record_warning("summary: model returned empty summary")
    except Exception as exc:  # noqa: BLE001 - surface as a warning, keep going
        _record_warning(f"summary: extraction failed ({_err(exc)})")

    # 2. Concepts (local, evidence-required) - receive the summary.
    try:
        sys_msg, user_msg = concepts_messages(markdown, sections, max_concepts, summary=out.summary)
        raw = client.json_completion(sys_msg, user_msg)
        out.concepts = _parse_concepts(raw, warnings, sections, total_lines)
    except Exception as exc:  # noqa: BLE001
        _record_warning(f"concepts: extraction failed ({_err(exc)})")

    # 3. Entities (local, evidence-required) - receive summary + concepts.
    try:
        sys_msg, user_msg = entities_messages(
            markdown,
            sections,
            max_entities,
            summary=out.summary,
            concepts=out.concepts,
        )
        raw = client.json_completion(sys_msg, user_msg)
        out.entities = _parse_entities(raw, warnings, sections, total_lines)
    except Exception as exc:  # noqa: BLE001
        _record_warning(f"entities: extraction failed ({_err(exc)})")

    # 4. Relations (proposed-only, evidence-required) - receive summary +
    #    concepts + entities; subject/object must come from those names.
    try:
        sys_msg, user_msg = relations_messages(
            markdown,
            sections,
            summary=out.summary,
            concepts=out.concepts,
            entities=out.entities,
        )
        raw = client.json_completion(sys_msg, user_msg)
        out.relations = _parse_relations(raw, warnings, sections, total_lines)
    except Exception as exc:  # noqa: BLE001
        _record_warning(f"relations: extraction failed ({_err(exc)})")

    return out


def _parse_summary(raw: str) -> str:
    """Pull ``summary`` from a JSON object; tolerant of fenced/malformed JSON."""
    obj = _parse_json(raw)
    if isinstance(obj, dict):
        s = obj.get("summary")
        if isinstance(s, str):
            return s.strip()
    return ""


def _parse_concepts(raw, warnings, sections, total_lines):
    items = _extract_items(raw, "concepts")
    return _filter_with_evidence(items, "concept", warnings, _to_concept, sections, total_lines)


def _parse_entities(raw, warnings, sections, total_lines):
    items = _extract_items(raw, "entities")
    return _filter_with_evidence(items, "entity", warnings, _to_entity, sections, total_lines)


def _parse_relations(raw, warnings, sections, total_lines):
    items = _extract_items(raw, "relations")
    return _filter_with_evidence(items, "relation", warnings, _to_edge, sections, total_lines)


def _extract_items(raw: str, key: str) -> list[dict]:
    """Parse the response and return the list under ``key`` (empty on miss)."""
    obj = _parse_json(raw)
    if isinstance(obj, dict):
        items = obj.get(key)
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
    return []


def _filter_with_evidence(items, label, warnings, converter, sections, total_lines):
    """Keep items whose evidence validates against the section map.

    Drops the rest with a single aggregated warning so a noisy model doesn't
    flood the manifest. ``converter`` returns ``None`` for a structurally bad
    item; an item with structurally-valid but section-mismatched evidence is
    also dropped (counted in the same warning).
    """
    kept = []
    dropped = 0
    for it in items:
        ext = converter(it)
        if ext is None or not validate_evidence(ext.evidence, sections, total_lines):
            dropped += 1
            continue
        kept.append(ext)
    if dropped:
        warnings.append(
            f"{label}: dropped {dropped} item(s) with missing/invalid/out-of-range evidence"
        )
    return kept


def _confidence(obj: dict) -> float | None:
    """Coerce an LLM ``confidence`` value to a 0-1 float, or ``None``."""
    c = obj.get("confidence")
    if isinstance(c, bool):  # bool is an int subclass; reject it
        return None
    if isinstance(c, (int, float)):
        try:
            v = float(c)
        except (TypeError, ValueError):
            return None
        if 0.0 <= v <= 1.0:
            return v
    return None


def _to_concept(obj: dict) -> ConceptExtract | None:
    name = obj.get("name")
    desc = obj.get("description")
    if not isinstance(name, str) or not name.strip():
        return None
    return ConceptExtract(
        name=name.strip(),
        description=desc.strip() if isinstance(desc, str) else "",
        evidence=_evidence_from(obj),
        confidence=_confidence(obj),
    )


def _to_entity(obj: dict) -> EntityExtract | None:
    name = obj.get("name")
    etype = obj.get("type")
    desc = obj.get("description")
    aliases = obj.get("aliases")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(etype, str) or not etype.strip():
        return None
    if isinstance(aliases, list):
        al = [str(a).strip() for a in aliases if isinstance(a, str) and a.strip()]
    else:
        al = []
    return EntityExtract(
        name=name.strip(),
        entity_type=etype.strip(),
        description=desc.strip() if isinstance(desc, str) else "",
        aliases=al,
        evidence=_evidence_from(obj),
        confidence=_confidence(obj),
    )


def _to_edge(obj: dict) -> ProposedEdge | None:
    subj = obj.get("subject")
    rel = obj.get("relation")
    obj_name = obj.get("object")
    note = obj.get("note")
    if not (isinstance(subj, str) and isinstance(rel, str) and isinstance(obj_name, str)):
        return None
    if not (subj.strip() and rel.strip() and obj_name.strip()):
        return None
    return ProposedEdge(
        subject=subj.strip(),
        relation=rel.strip(),
        object=obj_name.strip(),
        evidence=_evidence_from(obj),
        note=note.strip() if isinstance(note, str) else "",
    )


def _err(exc: Exception) -> str:
    """Compact error string for a warning line (no api_key leakage)."""
    msg = str(exc)
    if not msg:
        return type(exc).__name__
    return f"{type(exc).__name__}: {msg}"


def load_dotenv_values(dotenv_path: Path | None = None) -> dict[str, str]:
    """Load a ``.env`` file into a dict without touching ``os.environ``.

    Keeps the resolver pure: the CLI reads ``.env`` once and passes the dict
    in, so env-stash leakage tests stay deterministic. Returns ``{}`` when the
    file is absent or unreadable.
    """
    if dotenv_path is None or not dotenv_path.exists():
        return {}
    try:
        from dotenv import dotenv_values

        vals = dotenv_values(dotenv_path)
        return {k: v for k, v in vals.items() if isinstance(v, str)}
    except Exception:  # noqa: BLE001 - .env is best-effort, never fatal
        return {}
