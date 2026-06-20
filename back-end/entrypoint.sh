#!/bin/sh

# ---------------------------------------------------------------------------
# Auto-ingest on first run.
#
# On Railway (or any container host) the chroma_db/ folder starts empty on
# a fresh deploy.  We run ingest.py once to embed all PDFs in documents/ and
# populate the vector store.  Subsequent restarts skip this step because the
# folder already contains data (persisted via a Railway Volume mounted at
# /app/chroma_db).
#
# Locally: delete chroma_db/ and restart to force a re-ingest after adding
# new documents.
# ---------------------------------------------------------------------------
CHROMA_DIR="$(dirname "$0")/chroma_db"

if [ "${FORCE_REINGEST:-0}" = "1" ] || [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    echo "chroma_db/ not found or empty (or FORCE_REINGEST=1) — running document ingestion..."
    python ingest.py
    echo "Ingestion complete."
else
    echo "chroma_db/ already populated — skipping ingestion."
fi

echo "Starting Flask app..."
exec gunicorn --bind 0.0.0.0:"${PORT:-5000}" --workers 1 --timeout 120 app:app
