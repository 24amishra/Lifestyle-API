"""
RAGAS scoring script for the health-coach chatbot's RAG pipeline.

Runs only the questions flagged for RAGAS scoring -- manual questions whose
`Evaluation Method` column == "RAGAS" in the Question Bank sheet, plus the
DeepEval-generated "rag_grounded" goldens -- through the local Flask app
(same in-process test-client approach as evals/DeepEval/test_dataset_eval.py),
captures {question, answer, retrieved_context, ground_truth} for each, scores
every row with RAGAS (faithfulness, answer_relevancy, context_precision,
context_recall) plus DeepEval's AnswerRelevancyMetric as a cross-check, and
writes a timestamped CSV.

Rows in the Question Bank flagged "Manual" (safety/tone/jailbreak/crisis
probes meant for human review, not RAGAS's context-grounded metrics) are
excluded, as are blank/summary rows.

Usage:
    cd back-end
    python evals/RAGAS/run_ragas_eval.py \
        --manual-csv evals/RAGAS/manual_questions.csv \
        [--limit 5]   # quick smoke test on first N rows

Inputs:
  1. Manual set: "Lifestyle-API - Question Bank.csv" exported from the
     team's Google Sheet, with columns "Question (natural user phrasing)",
     "Expected Answer / Policy Rule", "Category", "Source PDF (exact
     filename)", "Evaluation Method". Only rows with Evaluation Method ==
     "RAGAS" are used. Path via --manual-csv (defaults to
     evals/RAGAS/manual_questions.csv).
  2. DeepEval set: evals/DeepEval/generated/chatbot_100_goldens.json,
     filtered to additional_metadata.category == "rag_grounded" (76 of the
     100 goldens -- the other 24 are edge/trap cases: safety probes,
     off-topic, prompt injection, malformed input, etc. and are intentionally
     excluded here).

Overlap handling: questions are deduped by normalized text (lowercased,
whitespace-collapsed, trailing punctuation stripped). When a manual question
and a DeepEval question are near-duplicates, the manual version (and its
ground_truth) wins, since that's the human-authored source of truth.

Known RAGAS answer_relevancy caveat (why there's a second relevancy column):
RAGAS's answer_relevancy forces a score of exactly 0.0 whenever its LLM judge
flags the answer as "noncommittal" (evasive/vague/ambiguous), regardless of
embedding similarity -- see ragas/metrics/_answer_relevance.py's
`_calculate_score`. That metric is designed to average 3 independent
noncommittal judgments (`strictness=3`) so one flaky call can't zero the
score alone, but `LangchainLLMWrapper.agenerate_text` implements n>1 by
mutating a *shared* `ChatOpenAI.n` attribute across concurrent async calls --
a race condition that intermittently collapses n back to 1 under RAGAS's
concurrent evaluation, silently disabling that safety net (visible as
"LLM returned 1 generations instead of requested 3" warnings). Passing
`bypass_n=True` below avoids the shared-state mutation entirely by issuing 3
separate sequential-per-item LLM calls instead, restoring the intended
majority-style robustness.

Because this app's Motivational-Interviewing tone deliberately hedges
("it depends on your situation," "check with your care team") in ways that
can plausibly still read as "noncommittal" to a strict LLM judge, every row
also gets DeepEval's AnswerRelevancyMetric (evals/DeepEval/test_dataset_eval.py
already uses this elsewhere in the repo) as an independent cross-check that
has no noncommittal zero-out behavior. Compare the two `*_answer_relevancy`
columns for rows where RAGAS's score looks suspiciously low.

Output: evals/RAGAS/results/ragas_<tag>_<UTC timestamp>.csv
Each row: question, source ("manual"|"deepeval"), category, source_pdf,
ground_truth, answer, retrieved_context (list, joined with "\n---\n" for
CSV), ragas_faithfulness, ragas_answer_relevancy, ragas_context_precision,
ragas_context_recall, deepeval_answer_relevancy, deepeval_answer_relevancy_reason.
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_HERE))  # back-end/
_DEEPEVAL_GOLDENS = os.path.join(
    _BACKEND_DIR, "evals", "DeepEval", "generated", "chatbot_100_goldens.json"
)
_RESULTS_DIR = os.path.join(_HERE, "results")

sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# Question loading + dedupe
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.?!,;:]+$", "", text)
    return text


def load_manual_questions(csv_path: str) -> list[dict]:
    """
    Loads the manual question bank (exported from the "Lifestyle-API -
    Question Bank" Google Sheet) and keeps only rows whose `Evaluation
    Method` column is "RAGAS" -- the sheet also contains "Manual"-flagged
    rows (safety/tone/jailbreak/crisis probes meant for human review, not
    RAGAS's context-grounded metrics) and some blank/summary rows, both of
    which are intentionally excluded here.

    Expected columns (case-insensitive, exact sheet headers preferred):
      "Question (natural user phrasing)"   -> question
      "Expected Answer / Policy Rule"      -> ground_truth
      "Category"                           -> category (kept for reference)
      "Source PDF (exact filename)"        -> source_pdf (kept for reference)
      "Evaluation Method"                  -> filter column, must == "RAGAS"

    Falls back to generic `question` / `ground_truth` headers (no
    `Evaluation Method` filter applied) if the sheet's exact headers aren't
    found, so a plainer CSV still works.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Manual questions CSV not found at {csv_path}. Export the "
            "Google Sheet as CSV first."
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = {c.lower().strip(): c for c in (reader.fieldnames or [])}

        q_col = (
            fieldnames.get("question (natural user phrasing)")
            or fieldnames.get("question")
            or fieldnames.get("input")
        )
        gt_col = (
            fieldnames.get("expected answer / policy rule")
            or fieldnames.get("ground_truth")
            or fieldnames.get("expected_output")
            or fieldnames.get("answer")
        )
        cat_col = fieldnames.get("category")
        src_col = fieldnames.get("source pdf (exact filename)")
        method_col = fieldnames.get("evaluation method")

        if not q_col:
            raise ValueError(
                f"Couldn't find a 'question' column in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )

        skipped_non_ragas = 0
        for row in reader:
            question = (row.get(q_col) or "").strip()
            if not question:
                continue
            if method_col:
                method = (row.get(method_col) or "").strip().upper()
                if method != "RAGAS":
                    skipped_non_ragas += 1
                    continue
            ground_truth = (row.get(gt_col) or "").strip() if gt_col else ""
            rows.append({
                "question": question,
                "ground_truth": ground_truth,
                "category": (row.get(cat_col) or "").strip() if cat_col else "",
                "source_pdf": (row.get(src_col) or "").strip() if src_col else "",
                "source": "manual",
            })

    if method_col:
        print(
            f"Manual sheet: kept {len(rows)} rows flagged 'RAGAS', "
            f"skipped {skipped_non_ragas} rows (Manual / blank / other)."
        )
    return rows


def load_deepeval_questions(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"DeepEval goldens not found at {json_path}. Run "
            "evals/DeepEval/generate_dataset.py first."
        )
    with open(json_path, encoding="utf-8") as f:
        goldens = json.load(f)
    rows = []
    for g in goldens:
        category = (g.get("additional_metadata") or {}).get("category", "")
        # Exclude the 24 hand-authored edge/trap cases (safety, off-topic,
        # prompt injection, malformed input, non-english, LE8 boundaries,
        # etc.) -- RAGAS's context-grounded metrics assume a normal,
        # answerable, RAG-eligible question with a real ground-truth answer.
        if category != "rag_grounded":
            continue
        question = (g.get("input") or "").strip()
        if not question:
            continue
        rows.append({
            "question": question,
            "ground_truth": (g.get("expected_output") or "").strip(),
            "category": "rag_grounded",
            "source_pdf": "",
            "source": "deepeval",
        })
    return rows


def build_question_set(manual_csv: str) -> list[dict]:
    manual = load_manual_questions(manual_csv)
    deepeval_qs = load_deepeval_questions(_DEEPEVAL_GOLDENS)

    seen = {}
    for row in manual:
        seen[_normalize(row["question"])] = row
    overlap = 0
    for row in deepeval_qs:
        key = _normalize(row["question"])
        if key in seen:
            overlap += 1
            continue  # manual version wins
        seen[key] = row

    combined = list(seen.values())
    print(
        f"Loaded {len(manual)} manual + {len(deepeval_qs)} DeepEval "
        f"rag_grounded questions, {overlap} overlap(s) deduped -> "
        f"{len(combined)} total questions."
    )
    return combined


# ---------------------------------------------------------------------------
# Chatbot calls (local Flask app, in-process test client)
# ---------------------------------------------------------------------------

def get_flask_test_client():
    from app import app as flask_app  # noqa: E402 (needs sys.path insert above)
    return flask_app.test_client()


def call_chatbot(client, question: str) -> tuple[str, list[str]]:
    resp = client.post(
        "/endpoint",
        json={"message": question, "history": [], "le8_data": {}, "show_chunks": True},
    )
    data = resp.get_json() or {}
    reply = data.get("reply", "")
    debug = data.get("rag_debug", {}) or {}
    chunks = debug.get("chunks", [])
    retrieved_context = [c["text"] for c in chunks if c.get("used_in_context")]
    return reply, retrieved_context


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------

def score_with_ragas(records: list[dict]) -> list[dict]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Explicitly wrap the judge LLM/embeddings rather than relying on ragas's
    # implicit defaults. ragas is unpinned in requirements.txt, so pip installs
    # whatever is newest; current ragas expects `evaluate()` to receive
    # `llm=`/`embeddings=` directly, and the auto-constructed default
    # embeddings object doesn't implement the `.embed_query()` interface these
    # deprecated metric singletons call internally -- that mismatch raises
    # `AttributeError: 'OpenAIEmbeddings' object has no attribute
    # 'embed_query'` on the answer_relevancy jobs if not wrapped explicitly.
    #
    # bypass_n=True works around a separate bug: answer_relevancy asks the
    # judge LLM for 3 independent generations (strictness=3) so a single
    # flaky "noncommittal" call can't zero the score alone, but
    # LangchainLLMWrapper's default n>1 path mutates a *shared*
    # `ChatOpenAI.n` attribute, which races under RAGAS's concurrent
    # evaluation and silently collapses back to n=1 (visible as "LLM
    # returned 1 generations instead of requested 3" warnings). bypass_n
    # forces 3 separate sequential LLM calls per item instead, avoiding the
    # shared mutable state.
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"), bypass_n=True)
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # RAGAS's context_precision/context_recall require non-empty retrieved
    # contexts and ground truths to produce meaningful scores; rows missing
    # either are still scored (RAGAS will emit NaN for the metrics that
    # can't be computed) so nothing silently disappears from the CSV.
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["retrieved_context"] or [""],
            "ground_truth": r["ground_truth"] or "",
        }
        for r in records
    ])

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    scored_df = result.to_pandas()

    for i, row in enumerate(records):
        row["ragas_faithfulness"] = scored_df.loc[i, "faithfulness"]
        row["ragas_answer_relevancy"] = scored_df.loc[i, "answer_relevancy"]
        row["ragas_context_precision"] = scored_df.loc[i, "context_precision"]
        row["ragas_context_recall"] = scored_df.loc[i, "context_recall"]
    return records


# ---------------------------------------------------------------------------
# DeepEval answer-relevancy cross-check
#
# RAGAS's answer_relevancy can force a 0.0 on answers that read as
# "noncommittal" to its LLM judge -- a real risk for this app's
# Motivational-Interviewing tone (hedging, open-ended follow-ups). DeepEval's
# AnswerRelevancyMetric judges relevancy from LLM-extracted statements
# without that all-or-nothing noncommittal penalty, so it's a useful
# independent second opinion on the same answers.
# ---------------------------------------------------------------------------

def score_with_deepeval_relevancy(records: list[dict]) -> list[dict]:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    metric = AnswerRelevancyMetric(threshold=0.6, include_reason=True)

    for i, r in enumerate(records, 1):
        test_case = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            retrieval_context=r["retrieved_context"] or None,
        )
        try:
            metric.measure(test_case, _show_indicator=False)
            r["deepeval_answer_relevancy"] = metric.score
            r["deepeval_answer_relevancy_reason"] = metric.reason
        except Exception as e:  # noqa: BLE001 - don't let one bad row kill the run
            r["deepeval_answer_relevancy"] = ""
            r["deepeval_answer_relevancy_reason"] = f"ERROR: {e}"
        print(f"  [{i}/{len(records)}] deepeval_answer_relevancy="
              f"{r['deepeval_answer_relevancy']}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual-csv",
        default=os.path.join(_HERE, "manual_questions.csv"),
        help="Path to the exported manual-questions CSV (question,ground_truth).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (smoke test).",
    )
    parser.add_argument(
        "--tag", default="baseline",
        help="Version tag included in the output filename (e.g. 'baseline', 'post-fix-v1').",
    )
    args = parser.parse_args()

    questions = build_question_set(args.manual_csv)
    if args.limit:
        questions = questions[: args.limit]

    client = get_flask_test_client()

    print(f"Calling chatbot for {len(questions)} questions...")
    records = []
    for i, q in enumerate(questions, 1):
        answer, retrieved_context = call_chatbot(client, q["question"])
        records.append({
            **q,
            "answer": answer,
            "retrieved_context": retrieved_context,
        })
        print(f"  [{i}/{len(questions)}] {q['question'][:60]!r}")

    print("Scoring with RAGAS (faithfulness, answer_relevancy, "
          "context_precision, context_recall)...")
    records = score_with_ragas(records)

    print("Scoring with DeepEval AnswerRelevancyMetric (cross-check)...")
    records = score_with_deepeval_relevancy(records)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(_RESULTS_DIR, f"ragas_{args.tag}_{timestamp}.csv")

    fieldnames = [
        "question", "source", "category", "source_pdf", "ground_truth", "answer",
        "retrieved_context", "ragas_faithfulness", "ragas_answer_relevancy",
        "ragas_context_precision", "ragas_context_recall",
        "deepeval_answer_relevancy", "deepeval_answer_relevancy_reason",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["retrieved_context"] = "\n---\n".join(row["retrieved_context"])
            writer.writerow(row)

    print(f"Saved {len(records)} scored rows to {out_path}")


if __name__ == "__main__":
    main()
