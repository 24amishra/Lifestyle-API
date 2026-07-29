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

Output: evals/RAGAS/results/ragas_<tag>.csv (no timestamp -- see "Crash
safety" below for why)
Each row: question, source ("manual"|"deepeval"), category, source_pdf,
ground_truth, answer, retrieved_context (list, joined with "\n---\n" for
CSV), ragas_faithfulness, ragas_answer_relevancy, ragas_context_precision,
ragas_context_recall, deepeval_answer_relevancy, deepeval_answer_relevancy_reason.

Crash safety / auto-resume / retry-then-stop
-----------------------------------------------
Same overall philosophy as evals/MedSafetyBench/run_medsafety_eval.py and
evals/WildJailbreak/run_wildjailbreak_eval.py (retry each external call with
backoff, stop the whole run rather than log-and-continue on a persistent
failure, auto-resume on a re-run with the same --tag) but split across THREE
checkpointed stages, because RAGAS's evaluate() call is architecturally a
single atomic batch operation over the whole dataset -- it cannot be
checkpointed row-by-row the way a simple per-row API loop can:

  1. Fetch chatbot answers -> evals/RAGAS/results/ragas_<tag>_answers.csv
     One /endpoint call per question, retried with backoff, checkpointed
     (flushed) after every row. A persistent failure stops the run with
     nothing written for that row; re-running the same command skips
     already-fetched questions and retries from there.
  2. Score with RAGAS -> evals/RAGAS/results/ragas_<tag>_ragas_scored.csv
     One atomic evaluate() call over ALL fetched answers. If it fails, none
     of this stage's results are cached -- but stage 1's answers are safe,
     so a re-run skips straight back to re-attempting just this stage
     (no wasted /endpoint calls). Once evaluate() succeeds, results are
     cached to this file immediately so a later failure in stage 3 never
     forces RAGAS to be re-run (real cost: LLM-judge calls per row).
  3. Score with DeepEval AnswerRelevancyMetric (cross-check) -> the FINAL
     output, evals/RAGAS/results/ragas_<tag>.csv. Per-row, retried with
     backoff, checkpointed after every row, same stop-then-resume behavior
     as stage 1.

Use a different --tag (or delete the relevant results/ragas_<tag>_*.csv
files) for a deliberately fresh run instead.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

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


def call_chatbot(
    client, question: str, max_retries: int = 2
) -> tuple[str, list[str], str]:
    """Returns (answer, retrieved_context, error). `error` is empty on success."""
    last_error = ""
    for attempt in range(max_retries + 1):
        resp = client.post(
            "/endpoint",
            json={"message": question, "history": [], "le8_data": {}, "show_chunks": True},
        )
        data = resp.get_json() or {}
        if resp.status_code == 200 and "reply" in data:
            debug = data.get("rag_debug", {}) or {}
            chunks = debug.get("chunks", [])
            retrieved_context = [c["text"] for c in chunks if c.get("used_in_context")]
            return data.get("reply", ""), retrieved_context, ""
        last_error = data.get("error", f"HTTP {resp.status_code}")
        if attempt < max_retries:
            time.sleep(2 ** attempt)
    return "", [], last_error


_ANSWER_FIELDNAMES = [
    "question", "source", "category", "source_pdf", "ground_truth",
    "answer", "retrieved_context_json", "error",
]


def fetch_answers(client, questions: list[dict], answers_path: str) -> list[dict]:
    """Stage 1 of the pipeline (see module docstring "Crash safety"): calls
    the chatbot for each question, checkpointing to answers_path after every
    row so a persistent failure never loses already-fetched answers. Raises
    SystemExit(1) (after printing a resume-friendly message) if it stops
    early; only returns once every question has a fetched answer.
    """
    records = []
    done_keys = set()
    if os.path.exists(answers_path):
        skipped_errors = 0
        with open(answers_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("error"):
                    # Not "done" -- drop so it's retried fresh below, and so
                    # the file gets compacted once a fresh attempt succeeds.
                    skipped_errors += 1
                    continue
                records.append(row)
                done_keys.add(_normalize(row["question"]))
        print(
            f"Found {len(records)} already-fetched answers at {answers_path}"
            + (f", {skipped_errors} previously-errored rows will be retried" if skipped_errors else "")
            + " -- auto-resuming."
        )
        answers_file = open(answers_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(answers_file, fieldnames=_ANSWER_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in _ANSWER_FIELDNAMES})
        answers_file.flush()
    else:
        answers_file = open(answers_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(answers_file, fieldnames=_ANSWER_FIELDNAMES)
        writer.writeheader()
        answers_file.flush()

    todo = [q for q in questions if _normalize(q["question"]) not in done_keys]
    print(f"{len(todo)} questions left to fetch answers for ({len(questions) - len(todo)} already done).")

    stopped_early = False
    try:
        for i, q in enumerate(todo, 1):
            answer, retrieved_context, error = call_chatbot(client, q["question"])
            if error:
                print(f"\nStopping after {len(records)}/{len(questions)} answers fetched: "
                      f"/endpoint failed after retries -- {error}\n"
                      f"Nothing was recorded for this question. Progress so far is saved at "
                      f"{answers_path}.\nJust re-run the same command (same --tag) to "
                      f"auto-resume and retry this question.")
                stopped_early = True
                break
            record = {
                **q,
                "answer": answer,
                "retrieved_context_json": json.dumps(retrieved_context),
                "error": "",
            }
            writer.writerow({k: record.get(k, "") for k in _ANSWER_FIELDNAMES})
            answers_file.flush()
            records.append(record)
            print(f"  [{len(questions) - len(todo) + i}/{len(questions)}] {q['question'][:60]!r}")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(records)}/{len(questions)} answers. "
              f"Progress is saved at {answers_path}.\nJust re-run the same command "
              f"to auto-resume.")
        stopped_early = True
    finally:
        answers_file.close()

    if stopped_early:
        raise SystemExit(1)

    by_key = {_normalize(r["question"]): r for r in records}
    ordered = []
    for q in questions:
        r = by_key[_normalize(q["question"])]
        ordered.append({
            **q,
            "answer": r["answer"],
            "retrieved_context": json.loads(r["retrieved_context_json"]) if r.get("retrieved_context_json") else [],
        })
    return ordered


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


_FINAL_FIELDNAMES = [
    "question", "source", "category", "source_pdf", "ground_truth", "answer",
    "retrieved_context", "ragas_faithfulness", "ragas_answer_relevancy",
    "ragas_context_precision", "ragas_context_recall",
    "deepeval_answer_relevancy", "deepeval_answer_relevancy_reason",
]
# Everything up through the RAGAS columns -- stage 2's cache file, so a
# stage-3 (DeepEval) failure never forces the atomic, LLM-judge-heavy RAGAS
# evaluate() call to be repeated. See module docstring "Crash safety".
_RAGAS_SCORED_FIELDNAMES = _FINAL_FIELDNAMES[:-2]


def _write_ragas_scored_cache(records: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RAGAS_SCORED_FIELDNAMES)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in _RAGAS_SCORED_FIELDNAMES}
            row["retrieved_context"] = "\n---\n".join(r.get("retrieved_context") or [])
            writer.writerow(row)


def _load_ragas_scored_cache(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["retrieved_context"] = r["retrieved_context"].split("\n---\n") if r["retrieved_context"] else []
        for k in ("ragas_faithfulness", "ragas_answer_relevancy",
                  "ragas_context_precision", "ragas_context_recall"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return rows


# ---------------------------------------------------------------------------
# DeepEval answer-relevancy cross-check (stage 3 -- the final output)
#
# RAGAS's answer_relevancy can force a 0.0 on answers that read as
# "noncommittal" to its LLM judge -- a real risk for this app's
# Motivational-Interviewing tone (hedging, open-ended follow-ups). DeepEval's
# AnswerRelevancyMetric judges relevancy from LLM-extracted statements
# without that all-or-nothing noncommittal penalty, so it's a useful
# independent second opinion on the same answers.
# ---------------------------------------------------------------------------

def score_with_deepeval_relevancy(
    records: list[dict], out_path: str, max_retries: int = 2
) -> list[dict]:
    """Stage 3 of the pipeline (see module docstring "Crash safety"): scores
    each row with DeepEval's AnswerRelevancyMetric, retried with backoff,
    checkpointed to out_path (the FINAL output file) after every row. Stops
    the run on a persistent per-row failure rather than silently recording an
    empty/error score. Raises SystemExit(1) if it stops early.
    """
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    metric = AnswerRelevancyMetric(threshold=0.6, include_reason=True)
    out_fieldnames = _FINAL_FIELDNAMES + ["error"]

    done = []
    done_keys = set()
    if os.path.exists(out_path):
        skipped_errors = 0
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("error"):
                    skipped_errors += 1
                    continue
                done.append(row)
                done_keys.add(_normalize(row["question"]))
        print(
            f"Found {len(done)} already-scored rows at {out_path}"
            + (f", {skipped_errors} previously-errored rows will be retried" if skipped_errors else "")
            + " -- auto-resuming."
        )
        out_file = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in done:
            writer.writerow({k: r.get(k, "") for k in out_fieldnames})
        out_file.flush()
    else:
        out_file = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=out_fieldnames)
        writer.writeheader()
        out_file.flush()

    todo = [r for r in records if _normalize(r["question"]) not in done_keys]
    print(f"{len(todo)} rows left for DeepEval cross-check ({len(records) - len(todo)} already done).")

    stopped_early = False
    try:
        for i, r in enumerate(todo, 1):
            test_case = LLMTestCase(
                input=r["question"],
                actual_output=r["answer"],
                retrieval_context=r["retrieved_context"] or None,
            )
            score, reason, last_error = "", "", ""
            for attempt in range(max_retries + 1):
                try:
                    metric.measure(test_case, _show_indicator=False)
                    score, reason, last_error = metric.score, metric.reason, ""
                    break
                except Exception as e:  # noqa: BLE001 - retried below
                    last_error = str(e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

            if last_error:
                print(f"\nStopping after {len(done)}/{len(records)} rows: DeepEval "
                      f"AnswerRelevancyMetric failed after retries -- {last_error}\n"
                      f"Nothing was recorded for this row. Progress so far is saved at "
                      f"{out_path}.\nJust re-run the same command (same --tag) to "
                      f"auto-resume -- fetching and RAGAS scoring are both cached "
                      f"already, so only this row will be retried.")
                stopped_early = True
                break

            row = {k: r.get(k, "") for k in _FINAL_FIELDNAMES}
            row["retrieved_context"] = "\n---\n".join(r.get("retrieved_context") or [])
            row["deepeval_answer_relevancy"] = score
            row["deepeval_answer_relevancy_reason"] = reason
            row["error"] = ""
            writer.writerow(row)
            out_file.flush()
            done.append(row)
            print(f"  [{len(records) - len(todo) + i}/{len(records)}] deepeval_answer_relevancy={score}")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(done)}/{len(records)} rows. Progress is saved "
              f"at {out_path}.\nJust re-run the same command to auto-resume.")
        stopped_early = True
    finally:
        out_file.close()

    if stopped_early:
        raise SystemExit(1)
    return done


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
        help=(
            "Names the output files (ragas_<tag>_answers.csv, "
            "ragas_<tag>_ragas_scored.csv, ragas_<tag>.csv). Re-running with "
            "the SAME tag automatically resumes each of the 3 stages from "
            "wherever it left off -- see 'Crash safety' above. Use a "
            "different --tag (or delete the existing files) to start a "
            "completely fresh run instead."
        ),
    )
    args = parser.parse_args()

    questions = build_question_set(args.manual_csv)
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    answers_path = os.path.join(_RESULTS_DIR, f"ragas_{args.tag}_answers.csv")
    ragas_cache_path = os.path.join(_RESULTS_DIR, f"ragas_{args.tag}_ragas_scored.csv")
    out_path = os.path.join(_RESULTS_DIR, f"ragas_{args.tag}.csv")

    client = get_flask_test_client()

    print(f"Fetching chatbot answers for {len(questions)} questions...")
    records = fetch_answers(client, questions, answers_path)

    if os.path.exists(ragas_cache_path):
        print(f"Found cached RAGAS scores at {ragas_cache_path} -- skipping "
              f"re-scoring (delete this file, or use a different --tag, to "
              f"force a fresh RAGAS pass).")
        records = _load_ragas_scored_cache(ragas_cache_path)
    else:
        print("Scoring with RAGAS (faithfulness, answer_relevancy, "
              "context_precision, context_recall)...")
        try:
            records = score_with_ragas(records)
        except Exception as e:
            print(
                f"\nRAGAS scoring failed: {e}\n"
                f"This is a single atomic call over the whole dataset (RAGAS's "
                f"evaluate() API has no partial/per-row checkpointing), so "
                f"nothing from this batch is cached -- but your fetched "
                f"chatbot answers ARE safe at {answers_path}. Just re-run the "
                f"same command (same --tag); answer-fetching will be skipped "
                f"(already done) and only RAGAS scoring will be retried."
            )
            raise
        _write_ragas_scored_cache(records, ragas_cache_path)
        print(f"Cached RAGAS scores to {ragas_cache_path}.")

    print("Scoring with DeepEval AnswerRelevancyMetric (cross-check)...")
    records = score_with_deepeval_relevancy(records, out_path)

    print(f"\nSaved {len(records)} scored rows to {out_path}")


if __name__ == "__main__":
    main()
