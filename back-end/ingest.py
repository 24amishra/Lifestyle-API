import os
import sys
import re
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Paths — always absolute so the script works from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_HERE, "chroma_db")
DOCS_FOLDER = os.path.join(_HERE, "documents")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete existing collection so we start fresh each ingest run
try:
    chroma_client.delete_collection("health_docs")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    "health_docs",
    metadata={"hnsw:space": "cosine"},
)

# Chunk settings — 600 chars with 80-char overlap.
# Slightly larger than before so each chunk carries more coherent context.
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

# ---------------------------------------------------------------------------
# Transcript PDF: section-aware parsing
# ---------------------------------------------------------------------------

# Matches "Animation #7 – Title", "Animation #21: Title", "Animation: Title", etc.
_ANIM_HEADER = re.compile(
    r"^Animation\s+(?:Script:\s*)?#?\d*\s*[–:\-]?\s*(.+)",
    re.IGNORECASE,
)
# The explicit "Animation link: https://..." line present in some sections
_ANIM_LINK = re.compile(r"Animation link:\s*(https?://\S+)", re.IGNORECASE)
# Any URL in the text
_ALL_URLS = re.compile(r"https?://\S+")
# Vimeo URLs belong to animation_url metadata, not the reference list
_VIMEO = re.compile(r"vimeo\.com", re.IGNORECASE)
# Lines that look like continuation / body text rather than section headers
_SKIP_STARTS = re.compile(
    r"^[\•\-\(\[\d\"o]|^Narrator|^Scene|^https?|^\s*$|^\[",
    re.IGNORECASE,
)
# ---------------------------------------------------------------------------
# Text cleaners — two variants, applied to the correct document type.
# Research PDFs must NOT have parentheticals or bracketed content stripped:
# pypdf extracts column-layout text with line breaks in the middle of
# sentences, so "(e.g., presurgical, during treatment, and with bone
# metastasis)" ends up on its own line and looks like a stage direction.
# Stripping it destroys critical clinical content and corrupts embeddings.
# ---------------------------------------------------------------------------

# Patterns safe to remove from transcript / animation-script PDFs only.
_TRANSCRIPT_NOISE = re.compile(
    r"(?m)"                           # multiline
    r"^\s*Narrator:\s*\n?"            # "Narrator:" label lines
    r"|^\s*\[.*?\]\s*\n?"             # [Stage direction] lines
    r"|^\s*\(.*?\)\s*\n?"             # (parenthetical action) lines
    r"|\f"                            # form-feed chars left by pypdf
)

# For research PDFs: only strip form-feeds and normalise whitespace.
# Never remove parentheticals — they contain clinical context that pypdf
# places on their own line due to column-layout extraction.
_RESEARCH_NOISE = re.compile(r"\f")


def _clean_transcript(text: str) -> str:
    """Clean animation transcript text before chunking."""
    text = _TRANSCRIPT_NOISE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _clean_research(text: str) -> str:
    """Clean research paper text before chunking — conservative pass only."""
    text = _RESEARCH_NOISE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()



def _is_section_start(page_text: str) -> bool:
    """Heuristic: does this page open a new named section?"""
    stripped = page_text.strip()
    if not stripped:
        return False
    first_line = stripped.split("\n")[0].strip()
    return (
        bool(first_line)
        and len(first_line) < 80
        and not _SKIP_STARTS.match(first_line)
        and len(stripped) > 80  # page has real content, not just a stray heading
    )


def _extract_ref_urls(text: str) -> list:
    """Return deduplicated non-Vimeo https:// URLs from text."""
    seen = set()
    result = []
    for u in _ALL_URLS.findall(text):
        u = u.rstrip(".,);\"'>]")
        if _VIMEO.search(u) or len(u) < 25:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def read_pdf_pages(filepath: str) -> list:
    """Extract text from every page of a PDF. Returns list of page strings."""
    reader = PdfReader(filepath)
    return [page.extract_text() or "" for page in reader.pages]


def is_transcript_pdf(pages: list) -> bool:
    """
    Detect whether this PDF is an animation transcript file by scanning
    the first 10 pages for the 'Animation link:' marker or an Animation header.
    Accepts a pre-read pages list to avoid re-reading the file.
    """
    for page in pages[:10]:
        text = page.strip()
        if _ANIM_LINK.search(text) or _ANIM_HEADER.match(text):
            return True
    return False


def parse_transcript_sections(pages: list) -> list:
    """
    Section-aware parser for the combined_scripts transcript PDF.

    Accepts a pre-read list of page strings (from read_pdf_pages) so the
    file is only opened once.

    Returns a list of dicts, each representing one logical section:
      - title          : section heading
      - animation_url  : Vimeo URL string, or "" if none
      - reference_urls : "|||"-joined non-Vimeo URLs found in the section, or ""
      - text           : clean body text (noise + animation link line removed)
    """
    sections = []

    state = {
        "title": "General Content",
        "animation_url": "",
        "pages": [],
    }

    def flush():
        if not state["pages"]:
            return
        full_text = "\n".join(state["pages"])
        # Remove the "Animation link: …" line — it's stored in metadata
        clean_text = _ANIM_LINK.sub("", full_text)
        clean_text = _clean_transcript(clean_text)
        if not clean_text:
            return
        ref_urls = _extract_ref_urls(full_text)
        sections.append(
            {
                "title": state["title"],
                "animation_url": state["animation_url"],
                # ChromaDB metadata values must be plain strings
                "reference_urls": "|||".join(ref_urls),
                "text": clean_text,
            }
        )

    for page_text in pages:
        stripped = page_text.strip()
        if not stripped:
            continue

        anim_match = _ANIM_HEADER.match(stripped)
        if anim_match:
            flush()
            state["title"] = anim_match.group(1).strip()
            link_match = _ANIM_LINK.search(stripped)
            state["animation_url"] = (
                link_match.group(1).strip() if link_match else ""
            )
            state["pages"] = [stripped]

        elif _is_section_start(stripped):
            flush()
            state["title"] = stripped.split("\n")[0].strip()
            state["animation_url"] = ""
            state["pages"] = [stripped]

        else:
            state["pages"].append(stripped)

    flush()
    return sections


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

all_chunks = []
all_ids = []
all_metadatas = []
chunk_index = 0

if not os.path.isdir(DOCS_FOLDER):
    print(f"Documents folder '{DOCS_FOLDER}' does not exist. Creating it.")
    os.makedirs(DOCS_FOLDER, exist_ok=True)

files = os.listdir(DOCS_FOLDER)
if not files:
    print("No files found in documents/. Add PDF or TXT files and re-run.")
    sys.exit(0)

for filename in sorted(files):
    filepath = os.path.join(DOCS_FOLDER, filename)

    # ------------------------------------------------------------------
    # PDF files — detect type first, then ingest accordingly
    # ------------------------------------------------------------------
    if filename.endswith(".pdf"):
        try:
            pages = read_pdf_pages(filepath)
        except Exception as e:
            print(f"  → ERROR reading {filename}: {e}. Skipping.")
            continue

        if not any(p.strip() for p in pages):
            print(f"  → WARNING: No text extracted from {filename}, skipping.")
            continue

        # Transcript PDF — section-aware ingestion with animation metadata
        if is_transcript_pdf(pages):
            print(f"Reading transcript PDF (section-aware): {filename}")
            try:
                sections = parse_transcript_sections(pages)
            except Exception as e:
                print(f"  → ERROR parsing sections in {filename}: {e}. Skipping.")
                continue

            print(f"  → {len(sections)} sections found")
            for section in sections:
                if not section["text"].strip():
                    continue
                chunks = splitter.split_text(section["text"])
                anim_flag = "📹 " if section["animation_url"] else "   "
                print(
                    f"     {anim_flag}[{section['title'][:55]}]  "
                    f"{len(chunks)} chunks"
                )
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_ids.append(f"chunk_{chunk_index}")
                    all_metadatas.append(
                        {
                            "source": filename,
                            "section_title": section["title"],
                            "animation_url": section["animation_url"],
                            "reference_urls": section["reference_urls"],
                            "chunk_index": i,
                        }
                    )
                    chunk_index += 1

        # Standard research PDF — flat ingestion, conservative cleaning only
        else:
            print(f"Reading research PDF: {filename}")
            text = _clean_research("\n".join(pages))
            if not text:
                print(f"  → WARNING: No usable text in {filename}, skipping.")
                continue
            chunks = splitter.split_text(text)
            print(f"  → {len(chunks)} chunks created")
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"chunk_{chunk_index}")
                all_metadatas.append(
                    {
                        "source": filename,
                        "section_title": "",
                        "animation_url": "",
                        "reference_urls": "",
                        "chunk_index": i,
                    }
                )
                chunk_index += 1

    # ------------------------------------------------------------------
    # TXT files
    # ------------------------------------------------------------------
    elif filename.endswith(".txt"):
        print(f"Reading TXT: {filename}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = _clean_research(f.read())
        except Exception as e:
            print(f"  → ERROR reading {filename}: {e}. Skipping.")
            continue

        if not text:
            print(f"  → WARNING: No text in {filename}, skipping.")
            continue

        chunks = splitter.split_text(text)
        print(f"  → {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_index}")
            all_metadatas.append(
                {
                    "source": filename,
                    "section_title": "",
                    "animation_url": "",
                    "reference_urls": "",
                    "chunk_index": i,
                }
            )
            chunk_index += 1

    else:
        print(f"Skipping unsupported file: {filename}")

# ---------------------------------------------------------------------------

if not all_chunks:
    print("No text chunks were produced. Check your documents.")
    sys.exit(0)

print(f"\nTotal chunks to embed: {len(all_chunks)}")
print("Sending to OpenAI for embedding (this may take a moment)…")

BATCH_SIZE = 100
all_embeddings = []

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i : i + BATCH_SIZE]
    response = openai_client.embeddings.create(
        input=batch,
        model="text-embedding-3-small",
    )
    all_embeddings.extend([item.embedding for item in response.data])
    print(f"  Embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids,
)

# ---------------------------------------------------------------------------
# Re-apply animation_overrides.json automatically.
#
# ingest.py deletes and rebuilds the "health_docs" collection from scratch
# on every run (see delete_collection() above), which means any patches
# previously applied by apply_overrides.py — overrides that only ever lived
# in ChromaDB, not in the source PDFs — get silently wiped. Previously this
# required remembering to manually re-run apply_overrides.py afterward, with
# nothing enforcing it; editing a PDF and re-ingesting would quietly regress
# animation links back to whatever the raw transcript text contains. Running
# it automatically here closes that gap.
# ---------------------------------------------------------------------------
print("\nRe-applying animation_overrides.json (if present)…")
try:
    import apply_overrides as _apply_overrides_mod

    if os.path.exists(_apply_overrides_mod.OVERRIDES_FILE):
        _overrides = _apply_overrides_mod.load_overrides()
        if _overrides:
            _stored_titles = _apply_overrides_mod.all_section_titles(collection)
            _all_applied = _apply_overrides_mod.apply_overrides(
                _overrides, collection, _stored_titles
            )
            if not _all_applied:
                print(
                    "WARNING: one or more animation overrides did not match "
                    "any section in the freshly-ingested collection. Run "
                    "`python apply_overrides.py --list` to inspect current "
                    "section titles and fix animation_overrides.json."
                )
        else:
            print("  No non-blank overrides to apply.")
    else:
        print(f"  {_apply_overrides_mod.OVERRIDES_FILE} not found — skipping.")
except Exception as e:
    print(f"WARNING: failed to auto-apply animation_overrides.json: {e}")

print(f"\nDone! {len(all_chunks)} chunks saved to chroma_db/")
