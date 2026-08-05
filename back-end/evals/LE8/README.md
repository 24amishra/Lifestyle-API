# LE8 Deterministic Scoring Tests

Plain `pytest` unit tests for the Life's Essential 8 scoring logic in
`app.py`. **No LLM judge, no API calls, no RAGAS / TruLens / DeepEval, no
network, no cost.** Runs in well under a second.

## Why this is not an LLM eval

LE8 scoring is fixed-threshold arithmetic. A non-HDL cholesterol of 145
mg/dL scores 60 and lands in the Intermediate tier — that is either right or
wrong, with no interpretation involved. A semantic judge cannot add
confidence to a comparison the interpreter already settles.

The RAG-grounding evaluators are worse than merely unnecessary here: they are
structurally unable to score this correctly. LE8 scores come from the system
prompt and the `le8_data` profile payload, never from retrieved chunks, so a
faithfulness/groundedness metric marks a *correct* score as unsupported
because it cannot find it in the retrieved context. Testing it that way
manufactures false negatives.

`evals/RAGAS/` and `evals/TruLens/` remain the right tools for what they
cover — retrieval quality and answer groundedness. This suite is a separate
concern and shares nothing with them.

## Files

| File | Covers |
|---|---|
| `test_le8_scoring.py` | The three LE8 metrics scored in Python: HbA1c (both scales), fasting glucose, non-HDL cholesterol. Plus `_le8_tier` cutoffs, the `_non_hdl_range_for_score` reverse map, and a prompt-text regression guard for the five metrics that have no Python implementation. |
| `test_pa_difficulty_mapping.py` | Downstream exercise-difficulty inference: `_pa_score`, `_infer_difficulty_from_le8` (70/40 cutoffs), `_resolve_exercise_difficulty`, and `_build_difficulty_note` for the `difficulty_pa` leak described under bug #3. |
| `le8_loader.py` | Shared helper that extracts functions from `app.py` without importing it (see below). |

The two test files are kept separate on purpose. Difficulty inference
*consumes* an LE8 score but is not an LE8 metric, and its 70/40 cutoffs are
deliberately different from the LE8 tier cutoffs of 80/50. Merging them
invites someone to "harmonize" two threshold sets that are supposed to
diverge.

## What is covered

For each scoring function:

- a value squarely inside each tier
- **every exact boundary value**, checked on both sides — 130/159/160/189/190/219/220 for non-HDL, 100/126 for fasting glucose, 5.7/6.5 non-diabetic and 7/8/9/10 diabetic for HbA1c, 40/70 for difficulty
- out-of-range input (negative, zero, and 10× the top threshold)
- cross-checks: the reverse map covers every score the forward function can emit; every emittable score maps to a valid tier

Thresholds are taken from the `LE8 SCORING REFERENCE` block in the system
prompt, `app.py:3109-3176`. Nothing is invented — each test cites the prompt
line it came from.

## How to run

From `back-end/`:

```bash
../.venv/bin/python -m pytest evals/LE8/ -v -rxX
```

`-rxX` prints xfail/xpass reasons in the summary. There are none left today,
but keep the flag: if a future bug is filed as `xfail(strict=True)`, that is
where its reason string shows up. If `pytest` is already on your PATH from the
repo venv, `pytest evals/LE8/ -v -rxX` works too.

Current state: **162 passed, 0 xfailed** — all five bugs found by this suite
are fixed, so every test is now a plain passing regression test.

Note the venv is at the **repo root** (`./.venv`, Python 3.9.6), not
`back-end/.venv`. Nothing was installed to run these; `pytest` 8.4.2 was
already present.

## How these tests import from a module that cannot be imported

`app.py` cannot be imported in this environment, for two pre-existing reasons
unrelated to these tests:

1. **Import chain** — it pulls in flask, openai, chromadb, pandas and pymongo
   and does env-var checks plus CSV/Chroma loading at module scope.
2. **Syntax** — the venv is Python 3.9.6 and `app.py` uses 3.10+ union
   annotations (`str | None` at `app.py:1830`), so it fails at parse time
   regardless of dependencies.

`le8_loader.py` sidesteps both: it parses `app.py` with `ast`, extracts only
the `FunctionDef` nodes under test, and `exec`s them in an isolated namespace
with `from __future__ import annotations` prepended so the 3.10-only
annotations are never evaluated.

This runs the **real source text** of those functions, not a copy — edit a
threshold in `app.py` and the next run sees it. If a function is renamed or
deleted, the loader raises `LookupError` rather than silently skipping tests.
**No fourth virtualenv was created**, and nothing was installed into the
shared one.

## Bug list

**All five bugs found by this suite are now fixed.** There are no `xfail`
markers left — every test is a plain passing assertion, and each one is a
regression test: reverting any part of a fix turns it red.

Ranked by patient-safety impact.

---

## FIXED — 2026-08-04

Two changes, both on 2026-08-04: bugs #1, #2 and #4 together (missing input
validation on the scoring functions, plus the caller-side catch), then bugs
#3 and #5 together (missing range guard on `_pa_score`).

### 1. Invalid biometric values scored as "Ideal" — was HIGH — FIXED
`_score_non_hdl` (`app.py:1822`), `_score_fasting_glucose` (`app.py:1814`),
`_score_hba1c` (`app.py:1802`)

A negative or zero value fell through the first `<` comparison and returned
**100 = Ideal**. That score was then written into an authoritative system note
(`"scores EXACTLY 100/100 (Ideal tier). Report this precisely; do not
recalculate it"`, `app.py:1884-1912`) that the model is instructed not to
question. Values arrive from regexes over free-text chat (`app.py:1784-1789`),
so a typo was enough. Highest severity: silently wrong *and* falsely
reassuring, with nothing visible to the user or the logs.

**Fix:** each function now raises `ValueError` on `value <= 0`. Not a clamp —
clamping to 100 *is* the bug, and clamping to 0 would fabricate a
critical-range reading from a typo. The scoring math is otherwise untouched.

### 4. Invalid HbA1c on the diabetic scale — was LOW — FIXED with #1
`_score_hba1c(..., has_diabetes=True)`

**Never an independent bug** — the same missing check, seen from the other
branch. The single `value <= 0` guard sits *above* the `has_diabetes` branch,
so one line fixed both scales. Listed separately only because its symptom
differed: the diabetic scale caps at 40, so an impossible value reported
40/Low rather than 100/Ideal — wrong, but understating rather than falsely
reassuring.

### 2. Caller did not catch the validation error — was HIGH — FIXED with #1
`_build_computed_value_note` (`app.py:1897-1957`)

Tracked separately because **shipping #1 alone would have made things
worse**: it would have converted a silently wrong score into an uncaught 500
on the chat route.

**Fix:** the three scoring calls inside `_build_computed_value_note` are each
wrapped in `try/except ValueError`, which logs at INFO and skips *that* note.
Only `ValueError` is caught — any other exception still surfaces.

The catch is per metric, not around the whole function or at the
`app.py:2880` call site, because those coarser placements would discard *all*
notes when one value is bad. A user who states a valid HbA1c and a typo'd
glucose keeps the correct HbA1c note. If every value is rejected, the
function degrades to the same `""` it already returns when it finds nothing
(`app.py:1863`), and the model just answers from profile data. Line 2880
needed no change.

Verified end-to-end: valid value → note produced; invalid → `""`, no crash;
mixed → only the bad note dropped.

### 3. Out-of-range PA score prescribed the hardest workout tier — was MEDIUM — FIXED
`_pa_score` (`app.py:671`)

The type guard was solid (strings, `None`, bools and malformed payloads all
correctly returned `None`), but there was no *range* guard. The PA score is
defined as capped at 100 (`app.py:3121`), yet 101, 500 and 1e9 passed through
and inferred **Advanced** — the most strenuous tier — for a cancer survivor
whose real activity level is unknown. Fatigue, neuropathy and deconditioning
are exactly the population risks the prompt warns about (`app.py:3179-3181`).

**Fix:** `_pa_score` now requires `0 <= score <= 100` and returns `None`
otherwise, with a `logger.warning` matching the existing style. A sentinel
rather than an exception — unlike #1 — because the function already documents
its contract as "the score, or None if unavailable" (`app.py:673`) and the
caller already degrades `None` to `Beginner`. NaN now fails the range check
too, which is the behaviour we want.

#### Second consequence, found while verifying the fix — the `difficulty_pa` leak

**Not in the original bug #3 write-up.** Tracing `difficulty_pa` end-to-end
after the fix showed the corrupt score had a second path to the user:
`_build_difficulty_note` interpolates it directly into text the model is
instructed to follow, gated only on `is not None` (`app.py:1534-1538`):

```
Their LE8 Physical Activity score is 1000000000.0/100.
```

So the wrong difficulty tier was only half the damage — the raw corrupt
number was also handed to the model to relay. The same range guard closes
this, because the corrupt value now arrives as `None` and the note takes its
"No LE8 Physical Activity score was available, so the default was used"
branch instead.

Confirmed by tracing real calls, not by reading the code:

| `score` in payload | `difficulty_pa` | inferred level | note branch |
|---|---|---|---|
| `1e9` | `None` | Beginner | "No LE8 Physical Activity score was available" |
| `85` | `85` | Advanced | "Their LE8 Physical Activity score is 85/100" |

Pinned by `test_corrupt_score_takes_the_no_score_branch` (which also asserts
the corrupt number appears nowhere in the note text) and
`test_valid_score_still_reaches_the_note` — the pair matters, since
suppressing *every* score would also make the first test pass.

### 5. Negative PA score — was LOW — FIXED with #3
`_pa_score`

Passed the type guard and reached `_resolve_exercise_difficulty` as a real
score. The resulting difficulty (`Beginner`) happened to be the safe default,
so the user-visible outcome was already fine — only the internal contract was
wrong. The same `0 <= score <= 100` guard fixes it; the difference now is
that `Beginner` is reached deliberately via the sentinel rather than by
accident of `-50 < 40`.

---

### Not bugs (asserted normally, passing)
Implausibly **high** values (10× the top threshold) fall through to
`return 0` on all three scoring functions. Unvalidated, but that is
effectively a clamp to the bottom of the scale and is directionally safe — it
reports the worst score rather than a falsely reassuring one.

## Caveat on the prompt-text regression guard

Blood pressure, BMI, sleep, diet and smoking have **no Python
implementation** — they are scored on the frontend and arrive pre-computed in
the `le8_data` payload (`app.py:1956-1972`). Their only definition in this
repo is prose in the system prompt.

Writing reference implementations for them here would be tautological: it
would test code no caller uses, and it would drift from the prompt. Instead
those tests assert the prompt still literally contains each threshold table.

**These are exact substring matches, so they are fragile to harmless edits.**
Reflowing a line, changing indentation, or realigning the `->` columns will
fail them without any threshold actually changing. That is the accepted cost
of guarding prose with no parser. **When one fails, diff the prompt first** —
confirm whether a number moved or only whitespace did, before treating it as
a real scoring regression.
