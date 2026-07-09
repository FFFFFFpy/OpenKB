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

from openkb.okf.schema import SectionSpec

_SYSTEM = """\
You are the extraction agent for an OKF Bundle: a self-contained, per-article \
knowledge extract. You receive ONE Markdown article and return ONLY a JSON \
object - no prose, no code fences, no commentary outside the JSON.

Critical rules:
- You see only this one article. Treat every concept/entity as LOCAL to it. \
Do NOT assume a global wiki exists; do NOT emit links to other documents.
- Every concept, entity, and relation MUST include an `evidence` object with \
`heading_path`, `line_start`, and `line_end`. These must point at real lines \
in the article (use the provided section map). An extract with no evidence or \
out-of-range evidence is invalid and will be dropped.
- `line_start`/`line_end` are 1-indexed and inclusive, matching the section \
map below exactly.
- Be concrete and faithful. Do not speculate beyond what the article says.
"""


def _section_map(sections: list[SectionSpec]) -> str:
    """Render the H2 section list with heading path + line range for citation."""
    if not sections:
        return "(no sections)"
    lines = ["Section map (use these coordinates for evidence):"]
    for s in sections:
        lines.append(f"- {s.heading_path} | lines {s.line_start}-{s.line_end}")
    return "\n".join(lines)


def _user_body(markdown: str, sections: list[SectionSpec], task: str) -> str:
    return (
        f"{task}\n\n"
        f"{_section_map(sections)}\n\n"
        f"--- ARTICLE START ---\n{markdown}\n--- ARTICLE END ---\n"
    )


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
    markdown: str, sections: list[SectionSpec], max_concepts: int
) -> tuple[str, str]:
    """Build the (system, user) messages for the local-concepts extract."""
    task = (
        f"Extract up to {max_concepts} key CONCEPTS from this article. Return a JSON object:\n"
        '{\n  "concepts": [\n'
        "    {\n"
        '      "name": "<concept name>",\n'
        '      "description": "<one or two sentences>",\n'
        '      "evidence": {"heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>}\n'
        "    }\n  ]\n"
        "}\n"
        "These are LOCAL concepts for this article only - not global wiki pages. "
        "Each concept MUST have a valid evidence object. Return only the JSON object."
    )
    return _SYSTEM, _user_body(markdown, sections, task)


def entities_messages(
    markdown: str, sections: list[SectionSpec], max_entities: int
) -> tuple[str, str]:
    """Build the (system, user) messages for the local-entities extract."""
    task = (
        f"Extract up to {max_entities} named ENTITIES from this article. Return a JSON object:\n"
        '{\n  "entities": [\n'
        "    {\n"
        '      "name": "<entity name>",\n'
        '      "type": "person|organization|place|product|work|event|other",\n'
        '      "description": "<one sentence>",\n'
        '      "aliases": ["<alt name>", ...],\n'
        '      "evidence": {"heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>}\n'
        "    }\n  ]\n"
        "}\n"
        "These are LOCAL entities for this article only. Each entity MUST have a valid "
        "evidence object. Return only the JSON object."
    )
    return _SYSTEM, _user_body(markdown, sections, task)


def relations_messages(markdown: str, sections: list[SectionSpec]) -> tuple[str, str]:
    """Build the (system, user) messages for the proposed-relations extract."""
    task = (
        "Propose relations BETWEEN the concepts and entities of THIS article. "
        "Return a JSON object:\n"
        '{\n  "relations": [\n'
        "    {\n"
        '      "subject": "<name>",\n'
        '      "relation": "<relation type>",\n'
        '      "object": "<name>",\n'
        '      "evidence": {"heading_path": "<section>", '
        '"line_start": <int>, "line_end": <int>},\n'
        '      "note": "<optional short note>"\n'
        "    }\n  ]\n"
        "}\n"
        "Subject/object must name concepts or entities that appear in this article. "
        "These are PROPOSED edges only - do not modify the article body. "
        "Each relation MUST have a valid evidence object. Return only the JSON object."
    )
    return _SYSTEM, _user_body(markdown, sections, task)
