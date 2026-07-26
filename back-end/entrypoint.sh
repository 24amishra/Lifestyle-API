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
# --limit-request-field_size / --limit-request-fields raised above gunicorn's
# defaults (8190 bytes / 100 fields). A large Cookie header (e.g. an
# accumulated session cookie, or a stale cookie left over from another local
# app that used this same port) can exceed the default limit; gunicorn then
# drops the connection without sending a real HTTP response, which the
# browser reports as a bare "TypeError: Failed to fetch" instead of a
# diagnosable error. Raising the limit avoids that failure mode for
# legitimately larger (but not abusive) header sets.
exec gunicorn --bind 0.0.0.0:"${PORT:-5000}" --workers 1 --timeout 120 \
    --limit-request-field_size 16380 --limit-request-fields 200 app:app
