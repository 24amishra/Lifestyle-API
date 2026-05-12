"""
apply_overrides.py
------------------
Patches animation URLs directly in ChromaDB without re-running a full ingest.

Usage:
  1. Open animation_overrides.json and fill in the Vimeo URL for each animation.
  2. Run:  python apply_overrides.py

Only chunks whose section_title matches a key in animation_overrides.json are
touched. Everything else (embeddings, text, reference URLs, research PDFs) is
left completely unchanged.

Safe to run multiple times — re-running with the same URLs is a no-op in effect,
it just overwrites the metadata with the same values.
"""

import json
import os
import sys
from dotenv import load_dotenv
import chromadb

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_FILE = os.path.join(_HERE, "animation_overrides.json")
CHROMA_PATH = os.path.join(_HERE, "chroma_db")


def load_overrides(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: {filepath} is not valid JSON — {e}")
        sys.exit(1)

    animations = data.get("animations", {})
    if not animations:
        print("No 'animations' key found in overrides file.")
        sys.exit(1)

    # Skip blank entries so a half-filled file doesn't wipe existing URLs
    active = {title: url for title, url in animations.items() if url.strip()}
    skipped = [title for title, url in animations.items() if not url.strip()]

    if skipped:
        print(f"Skipping {len(skipped)} blank entries (no URL provided yet):")
        for t in skipped:
            print(f"  - {t}")
        print()

    return active


def apply_overrides(overrides: dict) -> None:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("health_docs")
    except Exception as e:
        print(
            f"ERROR: Could not open ChromaDB collection at '{CHROMA_PATH}'.\n"
            f"Make sure you have run ingest.py at least once first.\nDetails: {e}"
        )
        sys.exit(1)

    total_updated = 0

    for title, new_url in overrides.items():
        # Fetch all chunks that belong to this animation section
        try:
            results = collection.get(
                where={"section_title": {"$eq": title}},
                include=["metadatas"],
            )
        except Exception as e:
            print(f"  ERROR querying for '{title}': {e}")
            continue

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        if not ids:
            print(f"  ⚠  No chunks found for: '{title}'")
            print(
                "     Check that the title matches exactly — "
                "copy it from the PDF section header."
            )
            continue

        # Patch animation_url in every chunk's metadata, leave everything else alone
        updated_metadatas = []
        for meta in metadatas:
            patched = dict(meta)
            patched["animation_url"] = new_url
            updated_metadatas.append(patched)

        try:
            collection.update(ids=ids, metadatas=updated_metadatas)
            print(f"  ✓  Updated {len(ids):>3} chunks  →  '{title}'")
            print(f"          URL: {new_url}")
            total_updated += len(ids)
        except Exception as e:
            print(f"  ERROR updating chunks for '{title}': {e}")

    print(f"\nDone. {total_updated} chunk(s) updated in ChromaDB.")
    print("No re-embedding needed — restart Flask to pick up the changes.")


if __name__ == "__main__":
    print(f"Loading overrides from: {OVERRIDES_FILE}\n")
    overrides = load_overrides(OVERRIDES_FILE)

    if not overrides:
        print("Nothing to apply — all entries are blank. Add URLs to animation_overrides.json first.")
        sys.exit(0)

    print(f"Applying {len(overrides)} override(s):\n")
    apply_overrides(overrides)
