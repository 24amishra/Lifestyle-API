"""
Custom TruLens dashboard page (registered via TRULENS_UI_CUSTOM_PAGES in
trulens_debug.py). The built-in Records tab has no column for `category` /
`source_pdf` -- that metadata only exists buried in record_json.meta -- so
this page pulls the same data via the same dashboard_utils helper Records.py
uses and surfaces it as real, sortable columns instead.
"""

import json

import pandas as pd
import streamlit as st
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import set_page_config

set_page_config(page_title="Flagged Overview")
st.title("Flagged Overview")
st.caption(
    "category / source_pdf as sortable columns -- the built-in Records tab "
    "only exposes them buried in raw JSON under Trace Details."
)


@st.cache_data(ttl=30)
def load_rows() -> pd.DataFrame:
    df, _ = get_records_and_feedback()
    rows = []
    for _, r in df.iterrows():
        record_json = r["record_json"]
        if isinstance(record_json, str):
            record_json = json.loads(record_json)
        meta = record_json.get("meta") or {}
        rows.append({
            "Record ID": r["record_id"],
            "Category": meta.get("category", ""),
            "Source PDF": meta.get("source_pdf", ""),
            "Question": r["input"],
            "Bot Answer": r["output"],
            "Groundedness": r.get("groundedness_measure_with_cot_reasons"),
            "Context Relevance": r.get("context_relevance_with_cot_reasons"),
        })
    return pd.DataFrame(rows)


df = load_rows()
st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    column_config={
        "Question": st.column_config.TextColumn(width="medium"),
        "Bot Answer": st.column_config.TextColumn(width="large"),
        "Groundedness": st.column_config.NumberColumn(format="%.2f"),
        "Context Relevance": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption(
    f"{len(df)} records. Click a column header to sort (e.g. Context "
    "Relevance ascending = worst first). Copy a Record ID into the "
    "**Records** tab's search box for the full chunk-by-chunk reasoning."
)
