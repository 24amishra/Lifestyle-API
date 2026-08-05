"""
Filters evals/RAGAS/RAGAS_Baseline.csv down to the rows worth debugging in
TruLens: weak-recall or ungrounded answers.

Usage:
    cd back-end
    python evals/TruLens/load_flagged_records.py \
        [--input evals/RAGAS/RAGAS_Baseline.csv] \
        [--output evals/TruLens/flagged_records.json]

To debug a new batch, point --input at a fresh RAGAS results CSV (e.g. one
of evals/RAGAS/results/ragas_<tag>_<timestamp>.csv from run_ragas_eval.py) --
it must have the same columns as RAGAS_Baseline.csv.

Flags a row when ragas_context_recall == 0 OR ragas_faithfulness < 0.4
(blank/NaN scores never match either condition, since there's no signal to
flag on). retrieved_context is split into a chunk list on "\\n---\\n", the
separator run_ragas_eval.py joins chunks with when writing the CSV.

Output columns (renamed to snake_case for the JSON): question, bot_answer,
retrieved_context (list), ragas_faithfulness, ragas_context_recall, category,
source_pdf. source, ground_truth, and the deepeval_* columns are dropped --
unused by the TruLens debugging pass.
"""

import argparse
import csv
import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT = os.path.join(os.path.dirname(_HERE), "RAGAS", "RAGAS_Baseline.csv")
_DEFAULT_OUTPUT = os.path.join(_HERE, "flagged_records.json")

CONTEXT_SEPARATOR = "\n---\n"
RECALL_FLAG_VALUE = 0
FAITHFULNESS_FLAG_THRESHOLD = 0.4


def parse_score(value: str) -> Optional[float]:
    """Parses a numeric CSV cell, collapsing blank/'nan' to None (valid JSON has no NaN)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # f != f is True only for NaN


def is_flagged(recall: Optional[float], faithfulness: Optional[float]) -> bool:
    return recall == RECALL_FLAG_VALUE or (
        faithfulness is not None and faithfulness < FAITHFULNESS_FLAG_THRESHOLD
    )


def split_context(raw: str) -> list[str]:
    return raw.split(CONTEXT_SEPARATOR) if raw else []


def load_flagged_records(input_path: str) -> list[dict]:
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    flagged = []
    for row in rows:
        recall = parse_score(row.get("ragas_context_recall"))
        faithfulness = parse_score(row.get("ragas_faithfulness"))
        if not is_flagged(recall, faithfulness):
            continue
        flagged.append({
            "question": row.get("Question", ""),
            "bot_answer": row.get("Bot Answer", ""),
            "retrieved_context": split_context(row.get("retrieved_context", "")),
            "ragas_faithfulness": faithfulness,
            "ragas_context_recall": recall,
            "category": row.get("Category", ""),
            "source_pdf": row.get("source_pdf", ""),
        })
    return flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=_DEFAULT_INPUT, help="Path to a RAGAS results CSV.")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Path to write flagged_records.json.")
    args = parser.parse_args()

    flagged = load_flagged_records(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(flagged, f, indent=2)

    print(f"Flagged {len(flagged)} rows (of the input CSV) -> {args.output}")


if __name__ == "__main__":
    main()
