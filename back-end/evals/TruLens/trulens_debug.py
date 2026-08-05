"""
Replays evals/TruLens/flagged_records.json through TruLens for interactive
debugging. No RAG app is re-run here -- each flagged row is wrapped as a
TruLens VirtualRecord from its already-captured question/answer/context, then
scored by two OpenAI-judged feedback functions (with chain-of-thought
reasoning) so the low-recall/low-faithfulness rows can be inspected.

Default behavior: score everything, write trulens_scored_results.csv, exit --
no Streamlit server. Pass --dashboard to also open the dashboard afterward.

Usage:
    cd back-end
    evals/TruLens/.venv/bin/python3 evals/TruLens/trulens_debug.py \
        [--input evals/TruLens/flagged_records.json] \
        [--limit 2]           # quick smoke test on first N rows
        [--dashboard]         # also launch the dashboard after scoring

Must run under evals/TruLens/.venv (Python 3.10+) -- trulens-feedback uses
`float | tuple[float, dict]`-style return annotations that raise TypeError
at import time on Python 3.9, which is what the shared back-end/.venv is
pinned to.

Requires OPENAI_API_KEY, loaded via python-dotenv from back-end/.env -- the
same file app.py's load_dotenv() reads for the RAGAS/DeepEval scripts.
"""

import argparse
import json
import os

import pandas as pd

# Must be set before the first add_record call: this trulens release
# defaults to OpenTelemetry-based tracing, under which VirtualRecord /
# TruVirtual.add_record (the API this script uses) raise "Not supported
# with OTel tracing enabled!". This restores the classic virtual-record flow.
os.environ.setdefault("TRULENS_OTEL_TRACING", "0")

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_HERE))  # back-end/
_DEFAULT_INPUT = os.path.join(_HERE, "flagged_records.json")
_DB_PATH = os.path.join(_HERE, "trulens_debug.sqlite")
_CSV_OUTPUT = os.path.join(_HERE, "trulens_scored_results.csv")

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# Adds the "Flagged Overview" tab (category/source_pdf as real columns --
# the built-in Records tab only exposes them buried in raw JSON). Read once
# at dashboard subprocess startup, so a running dashboard needs a restart to
# pick up changes to dashboard_pages/.
os.environ.setdefault("TRULENS_UI_CUSTOM_PAGES", os.path.join(_HERE, "dashboard_pages"))

from trulens.apps.virtual import TruVirtual, VirtualApp, VirtualRecord
from trulens.core import Feedback, FeedbackMode, Select, TruSession
from trulens.dashboard import run_dashboard
from trulens.providers.openai import OpenAI as TruOpenAI

# The call key the retrieved-context chunks are recorded under -- feedback
# selectors below refer back to this same selector.
RETRIEVE_CALL = Select.RecordCalls.retrieve


def _groundedness_reasoning(calls) -> str:
    """Flattens groundedness's per-chunk, per-claim breakdown into one
    readable block -- same underlying data the dashboard's "Groundedness"
    pill expands into (trulens.dashboard's expand_groundedness_df)."""
    if isinstance(calls, str):
        calls = json.loads(calls)
    parts = []
    for call in calls or []:
        for reason in (call.get("meta") or {}).get("reasons") or []:
            evidence = reason.get("supporting_evidence")
            if evidence:
                parts.append(evidence)
    return "\n\n".join(parts)


def _context_relevance_reasoning(calls) -> str:
    """One reason string per retrieved chunk (context_relevance_with_cot_reasons
    is called once per chunk) -- same data the dashboard's "Context Relevance"
    pill shows."""
    if isinstance(calls, str):
        calls = json.loads(calls)
    calls = calls or []
    parts = []
    for i, call in enumerate(calls, 1):
        reason = (call.get("meta") or {}).get("reason")
        if reason:
            parts.append(f"[Chunk {i}/{len(calls)}] {reason}")
    return "\n\n".join(parts)


def export_csv(session: TruSession, out_path: str) -> pd.DataFrame:
    df, _ = session.connector.db.get_records_and_feedback()
    rows = []
    for _, r in df.iterrows():
        record_json = r["record_json"]
        if isinstance(record_json, str):
            record_json = json.loads(record_json)
        meta = record_json.get("meta") or {}
        rows.append({
            "question": r["input"],
            "bot_answer": r["output"],
            "category": meta.get("category", ""),
            "source_pdf": meta.get("source_pdf", ""),
            "groundedness_score": r.get("groundedness_measure_with_cot_reasons"),
            "groundedness_reasoning": _groundedness_reasoning(
                r.get("groundedness_measure_with_cot_reasons_calls")
            ),
            "context_relevance_score": r.get("context_relevance_with_cot_reasons"),
            "context_relevance_reasoning": _context_relevance_reasoning(
                r.get("context_relevance_with_cot_reasons_calls")
            ),
        })
    out_df = pd.DataFrame(rows).sort_values("groundedness_score", ascending=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def build_records(rows: list[dict]) -> list[VirtualRecord]:
    records = []
    for row in rows:
        records.append(VirtualRecord(
            main_input=row["question"],
            main_output=row["bot_answer"],
            calls={
                RETRIEVE_CALL: {
                    "args": [row["question"]],
                    "rets": row["retrieved_context"],
                },
            },
            # Reference-only fields for reading the dashboard -- not scored.
            meta={"category": row["category"], "source_pdf": row["source_pdf"]},
        ))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=_DEFAULT_INPUT, help="Path to flagged_records.json.")
    parser.add_argument("--limit", type=int, default=None, help="Only load the first N records (smoke test).")
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Skip loading/scoring and just (re)open the dashboard against the existing trulens_debug.sqlite "
             "-- e.g. after editing dashboard_pages/, which only gets picked up on dashboard startup.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Also launch the Streamlit dashboard after scoring. Default is to score, write "
             "trulens_scored_results.csv, and exit -- no server.",
    )
    args = parser.parse_args()

    if args.dashboard_only:
        session = TruSession(database_url=f"sqlite:///{_DB_PATH}")
        run_dashboard(session)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY not set. Add it to {os.path.join(_BACKEND_DIR, '.env')}."
        )

    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} flagged records from {args.input}")

    session = TruSession(database_url=f"sqlite:///{_DB_PATH}")

    provider = TruOpenAI(model_engine="gpt-4o-mini")
    context = RETRIEVE_CALL.rets[:]

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons)
        .on(context)
        .on_output()
    )
    # _with_cot_reasons variant, not plain context_relevance -- plain
    # context_relevance returns a bare float with no reasoning attached,
    # which is no use for the CSV's context_relevance_reasoning column.
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons)
        .on_input()
        .on(context)
    )

    feedbacks = [f_groundedness, f_context_relevance]

    virtual_app = VirtualApp()
    tru_virtual = TruVirtual(
        app_name="Lifestyle-API RAG",
        app_version="flagged-ragas-rows",
        app=virtual_app,
        feedbacks=feedbacks,
    )

    print("Adding records...")
    records = build_records(rows)
    for i, record in enumerate(records, 1):
        # feedback_mode=NONE: add_record()'s normal async dispatch submits
        # through TP.submit -> TP._run_with_timeout -> TP._submit, which in
        # trulens-core 2.9.0/2.10.0 resubmit to each other forever and never
        # actually call the feedback function (confirmed: 0 rows ever land
        # in trulens_feedbacks). Score explicitly below instead.
        tru_virtual.add_record(record, feedback_mode=FeedbackMode.NONE)
        print(f"  [{i}/{len(records)}] added")

    print("Scoring feedbacks (sequential, gpt-4o-mini)...")
    total = len(records) * len(feedbacks)
    done = 0
    for record in records:
        for feedback in feedbacks:
            result = feedback.run_and_log(record=record, session=session, app=tru_virtual)
            done += 1
            score = result.result if result is not None else "ERROR"
            print(f"  [{done}/{total}] {feedback.name} = {score}")

    print(f"Writing {_CSV_OUTPUT} ...")
    out_df = export_csv(session, _CSV_OUTPUT)
    print(f"Wrote {len(out_df)} rows to {_CSV_OUTPUT}")

    if args.dashboard:
        run_dashboard(session)


if __name__ == "__main__":
    main()
