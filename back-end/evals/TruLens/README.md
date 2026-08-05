# TruLens

Interactive companion to [`evals/RAGAS`](../RAGAS): takes the rows RAGAS
already scored and flagged as weak, and re-scores them with an OpenAI judge
(`gpt-4o-mini`) for groundedness and context relevance, with full
chain-of-thought reasoning attached to each score. It doesn't call the
chatbot or re-run retrieval -- it replays RAGAS's own captured `Question` /
`Bot Answer` / `retrieved_context` through TruLens as "virtual" records.

Default output is `trulens_scored_results.csv` (one row per question, sorted
worst-groundedness-first). The TruLens dashboard is available too (`--dashboard`)
if you want to click through records interactively instead.

### How this maps to RAGAS's four metrics

Only two of RAGAS's four metrics have a TruLens equivalent used here:

| RAGAS metric | TruLens feedback used here | 
|---|---|
| `ragas_faithfulness` | `groundedness_measure_with_cot_reasons` -- same question (is the answer supported by the context?), reference-free |
| `ragas_context_precision` | `context_relevance_with_cot_reasons` -- rough match, both ask "is this chunk relevant to the question," reference-free |
| `ragas_answer_relevancy` | *(not built)* -- TruLens's `relevance(prompt, response)` would parallel it; needs no new data, just not requested |
| `ragas_context_recall` | *(no equivalent exists)* -- RAGAS computes this by comparing retrieved context against `ground_truth` to check whether the necessary info was retrieved. It's inherently reference-based, and `ground_truth` is deliberately dropped by `load_flagged_records.py` |

## Setup

This folder has its own venv (`evals/TruLens/.venv`), **not** the shared
`back-end/.venv`. Reason: `trulens-feedback` uses Python 3.10+ return-type
syntax (`float | tuple[float, dict]`) that raises `TypeError` at import time
on Python 3.9, which is what `back-end/.venv` is pinned to. If you ever need
to rebuild the venv:

```bash
cd back-end/evals/TruLens
python3.10 -m venv .venv   # any Python >=3.10 works
./.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `trulens`/`trulens-core`/etc. to `2.9.0`, the exact
versions this was tested against -- don't casually bump them. `2.10.0` (the
latest as of writing) has a broken `TP.submit` that silently drops all
feedback scoring: `add_record()`'s async dispatch never raises an error, it
just never calls the feedback function either, so every score silently stays
empty. See "Notes for whoever touches this next" below for the full story.

The OpenAI key is loaded the same way the RAGAS/DeepEval scripts get it --
`trulens_debug.py` calls `python-dotenv`'s `load_dotenv()` against
`back-end/.env` (`OPENAI_API_KEY`). Nothing is hardcoded; if the key's
missing, the script fails fast with where to put it.

## Usage

```bash
cd back-end
source evals/TruLens/.venv/bin/activate

# 1. Filter RAGAS_Baseline.csv down to the rows worth a look
python evals/TruLens/load_flagged_records.py
# -> Flagged 59 rows (of the input CSV) -> evals/TruLens/flagged_records.json

# 2. Score them in TruLens and write the CSV (no dashboard by default)
python evals/TruLens/trulens_debug.py
# -> Wrote 59 rows to evals/TruLens/trulens_scored_results.csv

# ...add --dashboard to also open the interactive dashboard afterward:
python evals/TruLens/trulens_debug.py --dashboard
# -> Dashboard started at http://localhost:XXXXX .
```

A row is flagged when `ragas_context_recall == 0` OR `ragas_faithfulness <
0.4`. `trulens_scored_results.csv` has one row per question, columns
`question, bot_answer, category, source_pdf, groundedness_score,
groundedness_reasoning, context_relevance_score,
context_relevance_reasoning`, sorted by `groundedness_score` ascending (worst
first). The reasoning columns are the same chain-of-thought text the
dashboard's feedback "pills" expand into -- see `_groundedness_reasoning`/
`_context_relevance_reasoning` in `trulens_debug.py` if the exact extraction
matters.

The SQLite file (`trulens_debug.sqlite`), the CSV, and `flagged_records.json`
are all regenerated on every run and gitignored -- don't hand-edit any of
them.

### Re-running on a new batch

Point `load_flagged_records.py` at whatever CSV you want to inspect next --
e.g. a fresh run of `evals/RAGAS/run_ragas_eval.py`, which writes to
`evals/RAGAS/results/ragas_<tag>_<timestamp>.csv` (same columns as
`RAGAS_Baseline.csv`):

```bash
python evals/TruLens/load_flagged_records.py \
    --input evals/RAGAS/results/ragas_post-fix-v1_20260731T120000Z.csv
python evals/TruLens/trulens_debug.py
```

Use `trulens_debug.py --limit 2` first if you just want to smoke-test that
scoring still works before burning API calls on a full batch.

Already scored and just want the dashboard back open (e.g. after editing
`dashboard_pages/`)? Skip re-scoring with:

```bash
python evals/TruLens/trulens_debug.py --dashboard-only
```

### Dashboard tabs

- **Records** -- the built-in per-record view. Each row is one flagged
  question; click its checkbox to open the full detail (input/output,
  feedback score "pills" -- click one to see its chain-of-thought reasoning
  -- and the retrieved chunks under Trace Details). Scores shown per feedback
  are the **plain mean across every retrieved chunk** (e.g. context_relevance
  scores each chunk 0-1 independently, then averages them) -- so one
  irrelevant chunk in an otherwise-good retrieval will visibly drag the
  average down, which is usually the actual signal worth chasing.
- **Flagged Overview** -- a custom page (`dashboard_pages/Flagged_Overview.py`)
  added because the built-in Records tab has no column for `category` /
  `source_pdf` -- that metadata only exists buried in `record_json.meta`.
  This page surfaces Category, Source PDF, Groundedness, and Context
  Relevance as real, sortable/filterable columns. ~Half the rows have a
  blank Source PDF -- expected, not a bug: those came from DeepEval-generated
  goldens, which (per `run_ragas_eval.py`) never carry a source PDF, only the
  manually-authored Question Bank rows do.
- **Leaderboard** -- broken, see below. Ignore it.

## Notes for whoever touches this next

Three non-obvious things about the currently-pinned `trulens-core==2.9.0`
(2.10.0 has the same first two issues; the third was reproduced directly on
2.9.0, not just inferred from 2.10.0):

- **`TRULENS_OTEL_TRACING=0`** -- this trulens release defaults to
  OpenTelemetry-based tracing, under which `VirtualRecord` /
  `TruVirtual.add_record` (the API used here, matching TruLens's own
  `trulens.apps.virtual` docstring examples) raise `"Not supported with OTel
  tracing enabled!"`. Setting this env var before the first `add_record`
  call restores the classic virtual-record flow.
- **Feedbacks are scored via `feedback.run_and_log(...)` in a plain
  sequential loop, not `add_record`'s automatic async dispatch.** The
  default path submits through `TP.submit` -> `TP._run_with_timeout` ->
  `TP._submit`, and in this release those two methods just resubmit to each
  other forever without ever calling the real feedback function -- verified
  by checking `trulens_feedbacks` in the sqlite db directly (stayed at 0
  rows no matter how long `add_record`'s background threads ran).
  `Feedback.run_and_log()` computes and persists in the calling thread
  instead, sidestepping that code path entirely. It's sequential (not
  parallelized) on purpose, to keep concurrent OpenAI calls predictable
  given the quota situation; 59 rows x 2 feedbacks takes a few minutes.
- **The dashboard's Leaderboard tab (the default page it opens to) crashes**
  with `NotImplementedError: Operator 'getitem' is not supported on this
  expression`, raised from `_get_leaderboard_aggregates_pre_otel` in
  `trulens/core/database/sqlalchemy.py` (~line 1234), which builds
  `Record.cost_json["n_tokens"].as_float()` as a SQL JSON-index expression --
  not supported by this SQLAlchemy/SQLite combination. This is a dashboard
  rendering bug, not a data bug: the underlying scores are fine (confirmed by
  calling `db.get_records_and_feedback()` directly -- all 59 records and both
  feedback columns come back clean). The **Records tab** is a structurally
  separate query path (`get_records_and_feedback`, not
  `get_leaderboard_aggregates`) and is unaffected -- use it instead, via the
  sidebar or `<dashboard-url>/Records`.

If a future `trulens-core` release fixes any of these, the corresponding
workaround is safe to remove -- they're isolated to the top of
`trulens_debug.py` (first two) or just a matter of avoiding one tab (third).

**`VirtualRecord.record_id` is not a reproducible content hash** -- rebuilding
the exact same records from the exact same `flagged_records.json` produces a
*different* `record_hash_...` id each time (confirmed empirically; something
in the hash includes a timestamp or other per-run value). Practically: you
cannot add a new feedback function later and score it against "the same"
already-added records by just calling `build_records()` again -- it'll create
59 new orphaned records instead of attaching to the existing ones. That's why
adding `context_relevance_with_cot_reasons` required wiping
`trulens_debug.sqlite` and re-scoring both feedbacks from scratch, rather
than scoring only the new one. If you add a feedback function in the future,
plan on a full re-score, or fetch+deserialize the existing `record_json` rows
back into `Record` objects instead of rebuilding from source (untested here).

Retrieved-context chunks are split on `"\n---\n"`, the exact separator
`run_ragas_eval.py` joins them with when writing `retrieved_context` to CSV.
`load_flagged_records.py` verifies this against the real file rather than
assuming it.
