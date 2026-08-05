"""
Load individual functions out of app.py without importing app.py.

WHY THIS EXISTS: app.py cannot be imported in this environment. Two
independent reasons, both pre-existing and unrelated to these tests:

  1. Import chain — app.py pulls in flask, openai, chromadb, pandas, pymongo
     and executes module-level setup (env-var checks, CSV/Chroma loading) at
     import time. Importing it to unit-test five pure functions would mean
     standing up the whole app.

  2. Syntax — the shared venv is Python 3.9.6, and app.py uses 3.10+ union
     annotations (`str | None` at app.py:1830, `bool | None` at app.py:1842).
     A plain import fails at parse time regardless of the dependencies.

So instead of importing, we parse app.py's source with `ast`, pull out just
the FunctionDef nodes we need, and exec them in an isolated namespace. The
`from __future__ import annotations` prefix (PEP 563, available since 3.7)
makes the 3.10-only annotations lazy strings so they never get evaluated,
which is what lets 3.9 run them.

This tests the REAL source text of those functions — not a copy. If someone
edits a threshold in app.py, these tests see the edit on the next run.
"""

import ast
import logging
from pathlib import Path

# evals/LE8/le8_loader.py -> evals/LE8 -> evals -> back-end/app.py
APP_PY = Path(__file__).resolve().parent.parent.parent / "app.py"


def load_functions(*names):
    """
    Extract the named top-level functions from app.py and return them in a
    dict. Raises LookupError if any requested name is missing, so a renamed
    or deleted function fails loudly here instead of silently skipping tests.
    """
    source = APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wanted = set(names)
    segments = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segments[node.name] = ast.get_source_segment(source, node)

    missing = wanted - set(segments)
    if missing:
        raise LookupError(
            f"Not found as top-level functions in {APP_PY}: {sorted(missing)}. "
            "They were probably renamed or moved — update the test suite."
        )

    namespace = {
        # _pa_score logs a warning on malformed payloads. Give it a real logger
        # aimed at nowhere so the guard path runs without polluting test output.
        "logger": logging.getLogger("le8_tests"),
    }
    ordered = "\n\n".join(segments[n] for n in names)
    exec("from __future__ import annotations\n" + ordered, namespace)

    return {name: namespace[name] for name in names}


def read_app_source():
    """Raw app.py text, for the system-prompt threshold-table assertions."""
    return APP_PY.read_text(encoding="utf-8")
