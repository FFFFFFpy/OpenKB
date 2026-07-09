"""OKF - a subtraction compiler for fresh per-article Markdown -> OKF Bundle.

Unlike ``openkb add`` (which builds a long-lived, globally-merged wiki KB),
this package compiles a single Markdown article into a self-contained OKF
Bundle (``.okf.zip``) with no reads of, or writes to, the long-term KB state:
no PageIndex, no markitdown, no global concept merge, no backlink materialization.

Each article is compiled in isolation. ``compile-dir`` is a thin orchestrator
that loops ``compile_one`` per file with no shared context between files.
"""

from __future__ import annotations
