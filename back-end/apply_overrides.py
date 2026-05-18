"""
apply_overrides.py
------------------
Patches animation URLs directly in ChromaDB without re-running a full ingest.

Usage:
  # See all section titles currently stored in ChromaDB (run this first):
  docker-compose exec backend python apply_overrides.py --list

  # Apply the URLs from animation_overrides.json:
  docker-compose exec backend python apply_overrides.py

Titles in animation_overrides.json must match the stored section_title exactly.
If a title is not found, the script prints the closest match from ChromaDB to
help you correct it.

Safe to re-run — overwriting with the same URL is a no-op in effect.
"""

import json
import os
import sys
import difflib
from dotenv import load_dotenv
import chromadb

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_FILE = os.path.join(_HERE, "animation_overrides.json")
CHROMA_PATH = os.path.join(_HERE, "chroma_db")


def get_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_collection("health_docs")
    except Exception as e:
        print(
            f"ERROR: Could not open ChromaDB collection at '{CHROMA_PATH}'.\n"
            f"Make sure you have run ingest.py at least once first.\nDetails: {e}"
        )
        sys.exit(1)


def all_section_titles(collection) -> list:
    """Return every unique non-empty section_title stored in ChromaDB, sorted."""
    try:
        # Fetch all metadata in batches — ChromaDB has no GROUP BY, so we page
        limit = 500
        offset = 0
        titles = set()
        while True:
            batch = collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )
            metas = batch.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                if m and m.get("section_title"):
                    titles.add(m["section_title"])
            if len(metas) < limit:
                break
            offset += limit
        return sorted(titles)
    except Exception as e:
        print(f"ERROR fetching section titles: {e}")
        sys.exit(1)


def load_overrides() -> dict:
    try:
        with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {OVERRIDES_FILE} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: {OVERRIDES_FILE} is not valid JSON — {e}")
        sys.exit(1)

    animations = data.get("animations", {})
    if not animations:
        print("No 'animations' key found in overrides file.")
        sys.exit(1)

    active = {t: u for t, u in animations.items() if u.strip()}
    skipped = [t for t, u in animations.items() if not u.strip()]

    if skipped:
        print(f"Skipping {len(skipped)} blank entries (no URL provided):")
        for t in skipped:
            print(f"  - {t}")
        print()

    return active


def apply_overrides(overrides: dict, collection, stored_titles: list) -> None:
    total_updated = 0
    not_found = []

    for title, new_url in overrides.items():
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
            not_found.append(title)
            continue

        updated_metadatas = []
        for meta in metadatas:
            patched = dict(meta)
            patched["animation_url"] = new_url
            updated_metadatas.append(patched)

        try:
            collection.update(ids=ids, metadatas=updated_metadatas)
            print(f"  ✓  {len(ids):>3} chunks  →  {title}")
            total_updated += len(ids)
        except Exception as e:
            print(f"  ERROR updating chunks for '{title}': {e}")

    print(f"\nDone. {total_updated} chunk(s) updated in ChromaDB.")

    # Report titles that didn't match, with closest suggestions
    if not_found:
        print(f"\n⚠  {len(not_found)} title(s) had no matching chunks in ChromaDB.")
        print("   Update animation_overrides.json to use the exact stored title.\n")
        for title in not_found:
            matches = difflib.get_close_matches(title, stored_titles, n=2, cutoff=0.4)
            print(f"  NOT FOUND: '{title}'")
            if matches:
                for m in matches:
                    print(f"    → Did you mean: '{m}'")
            else:
                print(f"    → No close match found. Run --list to see all stored titles.")
            print()


if __name__ == "__main__":
    if "--list" in sys.argv:
        print(f"Connecting to ChromaDB at: {CHROMA_PATH}\n")
        col = get_collection()
        titles = all_section_titles(col)
        print(f"Found {len(titles)} unique section title(s) in ChromaDB:\n")
        for t in titles:
            print(f"  {t}")
        sys.exit(0)

    print(f"Loading overrides from: {OVERRIDES_FILE}\n")
    overrides = load_overrides()

    if not overrides:
        print("Nothing to apply — all entries are blank.")
        sys.exit(0)

    col = get_collection()
    stored = all_section_titles(col)

    print(f"Applying {len(overrides)} override(s):\n")
    apply_overrides(overrides, col, stored)
    print("\nNo re-embedding needed. Restart Flask to pick up the changes.")
