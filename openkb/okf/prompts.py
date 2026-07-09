"""Prompt builders for the four OKF LLM extracts.

Each builder returns a ``(system_prompt, user_message)`` pair. The user message
always carries the *full* Markdown body plus a section map (heading path +
line range) so the model can cite evidence coordinates that match
``sources/article.md`` exactly. Every extract type is asked to return a JSON
object (DeepSeek/Qwen require the word "json" in the prompt when the
``response_format=json_object`` kwarg is set - same convention as
``openkb/agent/compiler.py:49-51``).

Extracts are *local* to the article - they are not global wiki pages and must
not invent cross-document links. Relations are proposed-only (consumers decide
whether to accept them); nothing here materializes a backlink in the body.
"""

from __future__ import annotations

from openkb.okf.schema import ConceptExtract, EntityExtract, SectionSpec

_SYSTEM = """\
You are the extraction agent for an OKF Bundle: a self-contained, per-article \
knowledge extract. You receive ONE Markdown article and return ONLY a JSON \
object - no prose, no code fences, no commentary outside the JSON.

Critical rules:
- You see only this one article. Treat every concept/entity as LOCAL to it. \
Do NOT assume a global wiki exists; do NOT emit links to other documents.
- Every concept, entity, and relation MUST include an `evidence` object with \
`section_id`, `heading_path`, `line_start`, and `line_end`. These must point \
at real lines in the article (use the provided section map). An extract with \
no evidence or out-of-range evidence is invalid and will be dropped.
- `line_start`/`line_end` are 1-indexed and inclusive, matching the section \
map below exactly.
- Be concrete and faithful. Do not speculate beyond what the article says.
"""


def _section_map(sections: list[SectionSpec]) -> str:
    """Render section ids, headings, line ranges, and anchors for citation."""
    if not sections:
        return "(no sections)"
    lines = ["Section map (use these coordinates for evidence):"]
    for s in sections:
        sid = s.section_id or f"s{s.index + 1:04d}"
        lines.append(f"- {sid} | {s.heading_path} | lines {s.line_start}-{s.line_end}")
        if s.anchors:
            lines.append("  anchors:")
            for a in s.anchors:
                lines.append(f"  - {a.anchor_id} | {a.title} | line {a.line_no}")
    return "\n".join(lines)


def _user_body(markdown: str, sections: list[SectionSpec], task: str) -> str:
    return (
        f"{task}\n\n"
        f"{_section_map(sections)}\n\n"
        f"--- ARTICLE START ---\n{markdown}\n--- ARTICLE END ---\n"
    )


def _compact_concepts(concepts: list[ConceptExtract]) -> str:
    """One-line-per-concept compact view for downstream prompts.

    Carries the name and a trimmed description so the relations prompt can pick
    subject/object from the actual extracted names instead of re-inferring them.
    """
    if not concepts:
        return "(no concepts extracted yet)"
    lines = ["Prior extracted concepts (reuse these names, do not invent new ones):"]
    for c in concepts:
        desc = (c.description or "").strip().replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"- {c.name}: {desc}")
    return "\n".join(lines)


def _compact_entities(entities: list[EntityExtract]) -> str:
    """One-line-per-entity compact view for downstream prompts."""
    if not entities:
        return "(no entities extracted yet)"
    lines = ["Prior extracted entities (reuse these names, do not invent new ones):"]
    for e in entities:
        desc = (e.description or "").strip().replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        al = f" (aliases: {', '.join(e.aliases)})" if e.aliases else ""
        lines.append(f"- {e.name} [{e.entity_type}]: {desc}{al}")
    return "\n".join(lines)


def _prior_summary(summary: str) -> str:
    s = (summary or "").strip()
    return f"Prior summary of this article (use it, do not contradict it):\n{s}" if s else ""


def summary_messages(markdown: str, sections: list[SectionSpec], language: str) -> tuple[str, str]:
    """Build the (system, user) messages for the summary extract."""
    lang = language.strip() or "en"
    task = (
        "Write a concise summary of this article as a JSON object with the shape "
        '{\n  "summary": "<markdown summary, a few sentences>"\n}. '
        f"Write the summary in language: {lang}. "
        "Do not include evidence (summary is whole-article). Return only the JSON object."
    )
    return _SYSTEM, _user_body(markdown, sections, task)


def concepts_messages(
    markdown: str,
    sections: list[SectionSpec],
    max_concepts: int,
    *,
    summary: str = "",
) -> tuple[str, str]:
    """Build the (system, user) messages for the local-concepts extract.

    Receives the prior ``summary`` so concept extraction stays consistent with
    it rather than re-reading the article blind.
    """
    task = (
        f"Extract up to {max_concepts} key CONCEPTS from this article. Return a JSON object:\n"
        '{\n  "concepts": [\n'
        "    {\n"
        '      "name": "<concept name>",\n'
        '      "description": "<one or two sentences>",\n'
        '      "confidence": <0.0-1.0>,\n'
        '      "evidence": {"section_id": "<section id>", "heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>}\n'
        "    }\n  ]\n"
        "}\n"
        "These are LOCAL concepts for this article only - not global wiki pages. "
        "Each concept MUST have a valid evidence object. Return only the JSON object."
    )
    prior = _prior_summary(summary)
    full = f"{task}\n\n{prior}" if prior else task
    return _SYSTEM, _user_body(markdown, sections, full)


def entities_messages(
    markdown: str,
    sections: list[SectionSpec],
    max_entities: int,
    *,
    summary: str = "",
    concepts: list[ConceptExtract] | None = None,
) -> tuple[str, str]:
    """Build the (system, user) messages for the local-entities extract.

    Receives the prior ``summary`` and the compact ``concepts`` list so entity
    extraction aligns with what was already extracted.
    """
    task = (
        f"Extract up to {max_entities} named ENTITIES from this article. Return a JSON object:\n"
        '{\n  "entities": [\n'
        "    {\n"
        '      "name": "<entity name>",\n'
        '      "type": "person|organization|place|product|work|event|other",\n'
        '      "description": "<one sentence>",\n'
        '      "aliases": ["<alt name>", ...],\n'
        '      "confidence": <0.0-1.0>,\n'
        '      "evidence": {"section_id": "<section id>", "heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>}\n'
        "    }\n  ]\n"
        "}\n"
        "These are LOCAL entities for this article only. Each entity MUST have a valid "
        "evidence object. Return only the JSON object."
    )
    parts = [task]
    prior = _prior_summary(summary)
    if prior:
        parts.append(prior)
    if concepts:
        parts.append(_compact_concepts(concepts))
    full = "\n\n".join(parts)
    return _SYSTEM, _user_body(markdown, sections, full)


def relations_messages(
    markdown: str,
    sections: list[SectionSpec],
    *,
    summary: str = "",
    concepts: list[ConceptExtract] | None = None,
    entities: list[EntityExtract] | None = None,
) -> tuple[str, str]:
    """Build the (system, user) messages for the proposed-relations extract.

    Receives the prior ``summary`` + compact ``concepts`` + compact ``entities``.
    The prompt forbids inventing new node names: subject/object MUST be chosen
    from the supplied concept/entity names.
    """
    task = (
        "Propose relations BETWEEN the concepts and entities extracted above. "
        "Return a JSON object:\n"
        '{\n  "relations": [\n'
        "    {\n"
        '      "subject": "<one of the names listed above>",\n'
        '      "relation": "<relation type>",\n'
        '      "object": "<one of the names listed above>",\n'
        '      "evidence": {"section_id": "<section id>", "heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>},\n'
        '      "note": "<optional short note>"\n'
        "    }\n  ]\n"
        "}\n"
        "CRITICAL: subject and object MUST be exact names from the prior "
        "concepts/entities lists. Do NOT invent new names. If no relations "
        "exist, return an empty list. These are PROPOSED edges only - do not "
        "modify the article body. Each relation MUST have a valid evidence "
        "object. Return only the JSON object."
    )
    parts = [task]
    prior = _prior_summary(summary)
    if prior:
        parts.append(prior)
    if concepts:
        parts.append(_compact_concepts(concepts))
    if entities:
        parts.append(_compact_entities(entities))
    full = "\n\n".join(parts)
    return _SYSTEM, _user_body(markdown, sections, full)
