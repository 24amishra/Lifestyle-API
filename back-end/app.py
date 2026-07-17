import os
import json
import re
import datetime
import base64
import secrets
import hashlib
import requests
import logging
from functools import wraps
from time import time

from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
from urllib.parse import urlencode
import chromadb
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

_is_production = os.getenv("FLASK_DEBUG", "0") != "1"

_secret = os.getenv("FLASK_SECRET_KEY")
if not _secret or _secret == "change-me-before-production":
    if _is_production:
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set to a strong random value in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    _secret = "dev-only-insecure-key"
app.secret_key = _secret

app.config.update(
    SESSION_COOKIE_SAMESITE="None" if _is_production else "Lax",
    SESSION_COOKIE_SECURE=_is_production,
    SESSION_COOKIE_HTTPONLY=True,
)
CORS(app, origins=[
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
], supports_credentials=True)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["FitAuth"]
collection = db["Details"]

# ---------------------------------------------------------------------------
# ChromaDB — single persistent client, absolute path so Flask can start from
# any working directory without breaking the DB reference.
# _chroma_client is module-level for efficiency, but _get_chroma_collection()
# will reinitialise it if ingest.py has deleted and recreated the collection
# since Flask started (stale UUID in the cached client).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CHROMA_PATH = os.path.join(_HERE, "chroma_db")
_chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)


def _get_chroma_collection():
    """
    Return the health_docs collection.

    If the cached client holds a stale reference (collection was deleted and
    recreated by ingest.py while Flask was running), this reinitialises the
    client and retries once so callers never have to restart Flask manually
    after a re-ingest.
    """
    global _chroma_client
    try:
        col = _chroma_client.get_or_create_collection(
            "health_docs",
            metadata={"hnsw:space": "cosine"},
        )
        # Trigger a lightweight read to surface any stale-reference error now
        # rather than silently inside retrieve_context.
        col.count()
        return col
    except Exception:
        # Client is holding a stale collection reference — reinitialise.
        logger.info("ChromaDB client reinitialising (stale collection reference)")
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
        return _chroma_client.get_or_create_collection(
            "health_docs",
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# App config constants
# ---------------------------------------------------------------------------
FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
FITBIT_REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:5000/callback")
NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "(LifestyleAPI, contact@example.com)")
USE_MOCK_FITBIT = os.getenv("USE_MOCK_FITBIT", "0") == "1"

MAX_HISTORY_MESSAGES = 20       # ~20 turns is plenty; keeps token cost low
MAX_HISTORY_STORED   = 100      # hard cap on messages accepted from client
                                 # (prevents history-stuffing / DoS)
MAX_MESSAGE_LENGTH = 2000
CITY_PATTERN = re.compile(r"^[a-zA-Z\s\-'.]+$")

# Cosine distance threshold for RAG relevance (ChromaDB uses 1-cosine_similarity,
# so 0 = identical, 1 = orthogonal, 2 = opposite).
# Chunks with distance > this value are considered off-topic and dropped.
RAG_DISTANCE_THRESHOLD = 0.75

# Lenient threshold for surfacing animation cards.
# Animation script chunks embed differently from health questions (narrative vs
# clinical language), so they consistently score above RAG_DISTANCE_THRESHOLD.
# This separate threshold lets relevant animations surface even when their chunk
# narrowly lost to a research-paper chunk for the same topic.
# Cross-topic contamination (e.g. sleep animations during a PA intake turn) is
# prevented separately by _animation_matches_conversation() in chatbot().
ANIMATION_SURFACE_THRESHOLD = 0.82

# Lenient threshold used for the source-diversity secondary query.
# When all top chunks come from "combined scripts.pdf", we do a second ChromaDB
# query that excludes that source and include the best result if its distance
# is within this value. Set looser than RAG_DISTANCE_THRESHOLD so research
# paper chunks (which tend to score slightly worse on conversational queries)
# still get surfaced.
SOURCE_DIVERSITY_THRESHOLD = 0.85

# Maximum number of animation cards to surface per response.
# Prevents overwhelming the user when a broad question matches many videos.
MAX_ANIMATIONS_PER_RESPONSE = 2

rate_limit_store: dict = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20

# Phrases that signal a message is a follow-up / back-reference rather than
# a self-contained question. Used by _build_rag_query to decide whether to
# enrich the embedding query with prior context.
# Examples that match: "tell me more about that", "what about it?",
#                      "can you elaborate?", "those sound good"
# Examples that don't: "what exercises should I do today?",
#                      "what should I eat this week?"
FOLLOWUP_PATTERN = re.compile(
    r"\b(more|that|it|those|them|this|above|mentioned|earlier|again|"
    r"else|other|another|continue|elaborate|expand|go on|previous|"
    r"same|similar|related|what about|how about)\b",
    re.IGNORECASE,
)

# Phrases that signal the user is explicitly asking for research sources.
# Deliberately narrow — common health words like "data" are excluded so
# routine messages don't accidentally trigger reference injection.
REFERENCE_INTENT_PATTERN = re.compile(
    r"\b(source|sources|study|studies|research|evidence|paper|papers|"
    r"article|articles|reference|references|citation|citations|"
    r"where did you get|where does that come from|link to|read more|"
    r"learn more|journal|published|prove|proof|scientific|backed by)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Crisis / self-harm detection.
# This is a deterministic safety net that does NOT rely on the model
# reliably following the single prompt sentence about crisis response.
# When this pattern matches, the chatbot() handler (a) injects a hard
# system instruction for this turn and (b) verifies after the fact that
# the reply actually contains the 988 Lifeline and doesn't read like a
# conversation-ending message, patching it if not.
# Deliberately broad on intent phrases ("can't handle this anymore", "no
# point", "hopeless") in addition to explicit self-harm language, since
# cancer-survivorship crisis language is often indirect.
# ---------------------------------------------------------------------------
CRISIS_PATTERN = re.compile(
    r"\b(suicide|suicidal|kill myself|end my life|ending my life|"
    r"hurt(?:ing)? myself|harm(?:ing)? myself|self[\s-]harm|"
    r"don'?t want to (?:be here|live|wake up)|want to die|wish i (?:was|were) dead|"
    r"no (?:point|reason) (?:in|to) (?:living|going on|anymore)|"
    r"can'?t (?:handle|do|take|deal with) this anymore|"
    r"(?:feel|feeling) hopeless|give up on (?:life|everything))\b",
    re.IGNORECASE,
)

CRISIS_SYSTEM_NOTE = (
    "CRISIS LANGUAGE DETECTED THIS TURN — MANDATORY RESPONSE RULES:\n"
    "The user's message matched crisis/self-harm language. This turn's reply MUST:\n"
    "1. Lead with genuine empathy and warmth — acknowledge how hard this feels "
    "before anything else.\n"
    "2. Explicitly include the 988 Suicide and Crisis Lifeline (call or text 988) "
    "AND encourage reaching out to their care team or a trusted person.\n"
    "3. NOT pivot back to LE8 coaching, exercise, or scoring in this same reply — "
    "no coaching questions, no 'would you like to focus on...' redirects.\n"
    "4. NOT end the conversation. Close by staying present with them — e.g. invite "
    "them to keep talking, ask how they're doing right now, or note you're here to "
    "listen — never end on the crisis resources alone with nothing further offered.\n"
    "5. NOT diagnose or make clinical judgments — just support, resources, and presence."
)


def _reply_looks_crisis_safe(reply: str) -> bool:
    """
    Cheap deterministic check that a crisis-flagged reply actually contains
    the 988 Lifeline. Used as a safety net in case the model ignores
    CRISIS_SYSTEM_NOTE.
    """
    if not reply:
        return False
    return "988" in reply


CRISIS_FALLBACK_APPENDIX = (
    "\n\nIf you're in crisis or having thoughts of harming yourself, please reach "
    "out right now — call or text 988 (Suicide and Crisis Lifeline), or contact "
    "your care team. You don't have to go through this alone, and I'm here to keep "
    "talking with you whenever you're ready."
)

# ---------------------------------------------------------------------------
# Exercise video library — loaded once at startup from Exercise Library.csv.
#
# Column name config: update EV_COL_* if the CSV uses different header names.
# The loader does case-insensitive matching and partial matching as a fallback,
# so minor naming variations ("Link" vs "Vimeo Link") are handled automatically.
# ---------------------------------------------------------------------------
_EXERCISE_CSV_PATH = os.path.join(_HERE, "Exercise Library.csv")
EV_COL_CATEGORY   = "category"    # e.g. "Bodyweight", "Dumbbell", "Chair Yoga"
EV_COL_DIFFICULTY = "difficulty"  # "Beginner", "Intermediate", "Advanced"
EV_COL_TITLE      = "title"       # full video title; duration & format parsed from it
EV_COL_LINK       = "link"        # Vimeo URL

# Maximum exercise video cards surfaced per response
MAX_EXERCISE_VIDEOS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex matching any character outside printable ASCII (used for prompt
# sanitization below).
_NON_PRINTABLE_ASCII = re.compile(r'[^\x20-\x7E]')


def _sanitize_prompt_str(value, max_len: int = 80) -> str:
    """
    Sanitize an external string before embedding it in the system prompt.

    Removes non-printable characters and newlines (which could break prompt
    structure or inject new instructions), then truncates to max_len.
    Safe for use on NWS weather fields, le8_data string fields, etc.
    """
    if not isinstance(value, str):
        value = str(value)
    # Collapse newlines / tabs into a single space
    value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Strip non-printable ASCII
    value = _NON_PRINTABLE_ASCII.sub('', value)
    return value[:max_len]


def _safe_numeric(value, default: str = "N/A") -> str:
    """
    Return value formatted as a number string if it is genuinely numeric,
    otherwise return default.  Prevents non-numeric frontend payloads from
    being injected into the system prompt.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return default

# ---------------------------------------------------------------------------
# Exercise video helpers
# ---------------------------------------------------------------------------

def _clean_exercise_title(title: str) -> str:
    """
    Normalize a raw CSV title for display to the user.

    The source CSV has several data-quality issues: a botched em-dash export
    that shows up as the Unicode replacement character (U+FFFD, "�") in some
    rows, and an inconsistent "Workout"–"With" separator across rows (some use
    "�", some " - ", some "- ", most use "--"). None of that is meaningful to
    the matching logic, but it looks broken if shown to the user as-is, so we
    normalize every variant to a single en dash with spaces around it.
    """
    if not title:
        return title
    # Replace the mojibake replacement character and any run of hyphens used
    # as a separator with a consistent " – " (en dash).
    cleaned = title.replace("�", " – ")
    cleaned = re.sub(r"\s*-{1,2}\s*(?=With\b)", " – ", cleaned)
    # Collapse any accidental doubled/triple spaces created above.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _parse_exercise_title(title: str, category: str = "") -> dict:
    """
    Extract duration_minutes, format (Seated/Standing/Mix), and body_part
    from a video title string.

    Titles follow the pattern:
      "12 Minute Intermediate Bodyweight Lower Body Workout – With Rita (Bodyweight-Standing)"

    Some rows (all current Yoga/Tai Chi rows) omit the trailing
    "(Category-Position)" parenthetical entirely and never mention
    seated/standing/mix anywhere in the title, so the primary text-matching
    pass below can't determine a format for them. For those we fall back to
    a category-based default: "Chair Yoga" is inherently seated, and plain
    "Tai Chi"/"Yoga" workouts in this library are performed standing unless
    the title says otherwise. This keeps the seated/standing filter accurate
    instead of silently treating the whole category as "matches anything."
    """
    # Duration: "12 Minute", "15-Minute", "30 Min"
    dur_match = re.search(r"(\d+)\s*[-\s]?[Mm]in(?:ute)?", title)
    duration_minutes = int(dur_match.group(1)) if dur_match else None

    # Format: prefer the parenthetical at the end, e.g. "(Bodyweight-Standing)"
    paren_match = re.search(r"\(([^)]+)\)", title)
    search_text = paren_match.group(1).lower() if paren_match else title.lower()
    if "seated" in search_text or "chair" in search_text:
        fmt = "Seated"
    elif "standing" in search_text:
        fmt = "Standing"
    elif "mix" in search_text:
        fmt = "Mix"
    else:
        fmt = None

    if fmt is None:
        # Category-based fallback for rows with no format signal anywhere
        # in the title (see docstring above).
        cat_lower = (category or "").lower()
        tl_for_fallback = title.lower()
        if "chair" in tl_for_fallback or cat_lower == "chair yoga":
            fmt = "Seated"
        elif cat_lower == "tai chi":
            fmt = "Standing"

    # Body part
    tl = title.lower()
    if "full body" in tl or "full-body" in tl:
        body_part = "Full Body"
    elif "upper body" in tl or "upper-body" in tl:
        body_part = "Upper Body"
    elif "lower body" in tl or "lower-body" in tl:
        body_part = "Lower Body"
    elif "core" in tl:
        body_part = "Core"
    else:
        body_part = None

    return {"duration_minutes": duration_minutes, "format": fmt, "body_part": body_part}


def _detect_column_roles(df: "pd.DataFrame") -> dict:
    """
    Identify category / difficulty / title / link columns by DATA VALUES,
    not header names.  This is robust to any CSV export format.

    Detection logic (each role claimed at most once, in this order):
      link       — values mostly start with "http"
      difficulty — values are mostly Beginner / Intermediate / Advanced
      category   — values are mostly known workout-type strings
      title      — the remaining column with the longest average string length

    Falls back to EV_COL_* name-matching for any role still unresolved.
    """
    # Known content values for each role
    _KNOWN_CATS  = {"bodyweight", "dumbbell", "chair yoga", "tai chi", "yoga", "resistance bands"}
    _KNOWN_DIFFS = {"beginner", "intermediate", "advanced"}

    col_map: dict = {}

    for col in df.columns:
        if len(col_map) >= 4:
            break
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        vals_lower = vals.str.strip().str.lower()

        if "link" not in col_map and vals.str.startswith("http").mean() > 0.3:
            col_map["link"] = col

        elif "difficulty" not in col_map and vals_lower.isin(_KNOWN_DIFFS).mean() > 0.3:
            col_map["difficulty"] = col

        elif "category" not in col_map and vals_lower.isin(_KNOWN_CATS).mean() > 0.3:
            col_map["category"] = col

    # Title: longest-average-value column not already claimed
    if "title" not in col_map:
        used      = set(col_map.values())
        remaining = [c for c in df.columns if c not in used]
        if remaining:
            col_map["title"] = max(
                remaining,
                key=lambda c: df[c].dropna().astype(str).str.len().mean(),
            )

    # Fallback: name-based matching for any role still unmapped
    for configured, attr in [
        (EV_COL_CATEGORY,   "category"),
        (EV_COL_DIFFICULTY, "difficulty"),
        (EV_COL_TITLE,      "title"),
        (EV_COL_LINK,       "link"),
    ]:
        if attr in col_map:
            continue
        cn = configured.lower()
        if cn in df.columns:
            col_map[attr] = cn
        else:
            matches = [c for c in df.columns if cn in c or c in cn]
            if matches:
                col_map[attr] = matches[0]

    return col_map


def _load_exercise_videos() -> list:
    """
    Load Exercise Library.csv at startup, parse title metadata, and drop any
    rows that are missing a Vimeo link.  Returns a list of dicts, one per video.

    Column roles are detected by DATA VALUES (not header names) via
    _detect_column_roles(), so the loader works regardless of how the CSV
    was exported or what the headers are called.
    """
    if not os.path.exists(_EXERCISE_CSV_PATH):
        logger.warning(
            "Exercise Library.csv not found at %s — exercise video matching disabled.",
            _EXERCISE_CSV_PATH,
        )
        return []

    # Excel on Windows exports CSV in cp1252 (Windows-1252) by default.
    # Try UTF-8 first, fall back to cp1252, then latin-1 as a last resort.
    df = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(_EXERCISE_CSV_PATH, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.error("Failed to read Exercise Library.csv: %s", exc)
            return []
    if df is None:
        logger.error("Exercise Library.csv: could not decode with utf-8, cp1252, or latin-1")
        return []

    # Normalise headers to lowercase for content scanning
    df.columns = [str(c).strip().lower() for c in df.columns]
    logger.info("Exercise Library.csv columns detected: %s", list(df.columns))

    col_map = _detect_column_roles(df)
    logger.info("Exercise Library column mapping: %s", col_map)

    if not {"category", "difficulty", "title", "link"}.issubset(col_map):
        logger.error(
            "Exercise Library.csv: could not identify all required columns. "
            "Resolved mapping so far: %s", col_map,
        )
        return []

    videos = []
    # Guard against true duplicate rows — same Vimeo link listed twice (e.g.
    # from a copy-paste error in the spreadsheet). Deliberately keyed on the
    # LINK, not the title: the CSV has at least one pair of rows that share
    # identical title text but point to two different, distinct Vimeo videos
    # (two separate recordings of the same workout name). Deduping by title
    # would silently throw away one of those two genuinely different, working
    # videos — the user asked us to keep every video link, even broken ones,
    # so only exact link duplicates (the same video counted twice) are
    # dropped here.
    seen_links: set = set()
    # Track how many times each cleaned title has been seen so identical-title
    # different-video rows get a "(2)", "(3)", ... suffix instead of showing
    # up as indistinguishable cards with no way to tell them apart.
    title_counts: dict = {}
    for _, row in df.iterrows():
        link = str(row[col_map["link"]]).strip()
        if not link or link.lower() in ("nan", "none", "") or not link.startswith("https://"):
            continue

        link_key = link.strip().lower()
        if link_key in seen_links:
            logger.warning(
                "Exercise Library.csv: skipping row with duplicate Vimeo link %r "
                "— already loaded from an earlier row.",
                link[:60],
            )
            continue
        seen_links.add(link_key)

        raw_title  = str(row[col_map["title"]]).strip()
        title      = _clean_exercise_title(raw_title)
        category   = str(row[col_map["category"]]).strip()
        difficulty = str(row[col_map["difficulty"]]).strip()

        dedupe_key = re.sub(r"\s+", " ", title).strip().lower()
        title_counts[dedupe_key] = title_counts.get(dedupe_key, 0) + 1
        if title_counts[dedupe_key] > 1:
            logger.warning(
                "Exercise Library.csv: two different videos share the title "
                "%r — disambiguating with a suffix so both remain selectable "
                "and distinguishable.",
                title[:80],
            )
            title = f"{title} ({title_counts[dedupe_key]})"

        parsed = _parse_exercise_title(raw_title, category=category)
        videos.append({
            "category":         category,
            "difficulty":       difficulty,
            "title":            title,
            "vimeo_link":       link,
            "duration_minutes": parsed["duration_minutes"],
            "format":           parsed["format"],
            "body_part":        parsed["body_part"],
        })

    if videos:
        s = videos[0]
        logger.info(
            "Exercise Library sample — category: %r  difficulty: %r  "
            "title: %r  link: %r",
            s["category"], s["difficulty"],
            s["title"][:60], s["vimeo_link"][:50],
        )
    logger.info("Exercise Library loaded: %d videos with links.", len(videos))
    return videos


# Load once at startup — immutable after this point
EXERCISE_VIDEOS: list = _load_exercise_videos()


def _compute_exercise_options() -> tuple:
    """
    Derive the list of available workout categories and duration brackets
    directly from the loaded video library so the system prompt never offers
    an option that has zero matching videos.

    Returns (categories_str, durations_str) — both ready for injection into
    the system prompt.
    """
    if not EXERCISE_VIDEOS:
        # Sensible fallback when the CSV is missing or empty
        return (
            "bodyweight, dumbbells, resistance bands, yoga, chair yoga, or tai chi",
            "10\u201315 min, 15\u201320 min, or 25\u201330 min",
        )

    # ── Categories ──────────────────────────────────────────────────────────
    cats = sorted({v["category"] for v in EXERCISE_VIDEOS if v.get("category")})
    if len(cats) > 1:
        cats_str = ", ".join(cats[:-1]) + ", or " + cats[-1]
    elif cats:
        cats_str = cats[0]
    else:
        cats_str = "bodyweight, dumbbells, resistance bands, yoga, chair yoga, or tai chi"

    # ── Duration brackets ─────────────────────────────────────────────────
    # Each bracket label matches the regex used in _detect_exercise_filters.
    bracket_defs = [
        (0,  10, 15,  "10\u201315 min"),
        (1,  15, 20,  "15\u201320 min"),
        (2,  25, 30,  "25\u201330 min"),
        (3,  31, 999, "30+ min"),   # strictly > 30; a 30-min video is "25-30"
    ]
    durations_with_videos = set()
    for v in EXERCISE_VIDEOS:
        d = v.get("duration_minutes")
        if d is None:
            continue
        for order, lo, hi, label in bracket_defs:
            if lo <= d <= hi:
                durations_with_videos.add((order, label))

    available = sorted(durations_with_videos)           # sort by order key
    dur_labels = [label for _, label in available]
    if len(dur_labels) > 1:
        dur_str = ", ".join(dur_labels[:-1]) + ", or " + dur_labels[-1]
    elif dur_labels:
        dur_str = dur_labels[0]
    else:
        dur_str = "10\u201315 min, 15\u201320 min, or 25\u201330 min"

    return cats_str, dur_str


# Compute once at startup — injected into the system prompt so the chatbot
# only offers options that have at least one matching video.
_EXERCISE_AVAILABLE_CATEGORIES, _EXERCISE_AVAILABLE_DURATIONS = _compute_exercise_options()
logger.info(
    "Exercise options — categories: [%s] | durations: [%s]",
    _EXERCISE_AVAILABLE_CATEGORIES,
    _EXERCISE_AVAILABLE_DURATIONS,
)


def _infer_difficulty_from_le8(le8_data: dict) -> str:
    """
    Map the user's LE8 Physical Activity score to an exercise difficulty level.
    Falls back to Beginner when no score is available.
    """
    try:
        pa_score = (
            (le8_data or {}).get("metrics", {})
                           .get("physical_activity", {})
                           .get("score")
        )
        if pa_score is None:
            return "Beginner"
        if pa_score >= 70:
            return "Advanced"
        if pa_score >= 40:
            return "Intermediate"
        return "Beginner"
    except Exception as exc:
        logger.warning(
            "_infer_difficulty_from_le8: malformed le8_data payload (%r) — "
            "defaulting to Beginner. Payload keys: %s",
            exc, list((le8_data or {}).keys()),
        )
        return "Beginner"


def _detect_exercise_filters(history: list) -> dict:
    """
    Scan conversation history for answers to the four exercise preference
    questions [EV1]–[EV4] and return a structured filters dict.

    EV1 / EV2 / EV3 (single-answer questions): scan messages NEWEST-FIRST and
    stop at the first message that contains a relevant keyword.  This means
    the user’s most recent answer wins — if they said \u201cdumbbell\u201d earlier
    and then \u201cwait, I want bodyweight instead\u201d, only \u201cbodyweight\u201d is used.

    EV4 (movement exclusions): accumulate across all messages since a user
    may report multiple limitations in separate turns, and they rarely undo
    an exclusion.  Deduplication prevents double-adding.
    """
    filters: dict = {
        "categories":   [],
        "format":       [],
        "duration_min": None,
        "duration_max": None,
        "exclusions":   [],
    }

    user_msgs = [
        msg for msg in history[-MAX_HISTORY_STORED:]
        if isinstance(msg, dict) and msg.get("role") == "user"
    ]
    if not user_msgs:
        return filters

    # ── [EV1] Workout category ──────────────────────────────────────────
    # Order matters: \u201cchair yoga\u201d before \u201cyoga\u201d so we don\u2019t add both.
    category_keywords = [
        ("chair yoga",       "Chair Yoga"),
        ("tai chi",          "Tai Chi"),
        ("resistance band",  "Resistance Bands"),
        ("resistance bands", "Resistance Bands"),
        ("dumbbell",         "Dumbbell"),
        ("dumbbells",        "Dumbbell"),
        ("hand weight",      "Dumbbell"),
        ("bodyweight",       "Bodyweight"),
        ("body weight",      "Bodyweight"),
        ("yoga",             "Yoga"),
    ]
    # Correction signal: words that indicate the user is changing their answer.
    # When present alongside multiple categories in the same message, only the
    # category whose keyword appears LATEST in the text is used — that’s the
    # new preference.  Example: \u201cI said dumbbell but I want bodyweight instead\u201d
    # → \u201cbodyweight\u201d appears after \u201cinstead/but\u201d so it wins.
    # NOTE: "actually" and "rather" are deliberately NOT bare triggers here —
    # both are common as plain emphasis/preference words unrelated to
    # switching an answer (e.g. "chair yoga is actually great for me",
    # "I'd rather relax after"), and treating them as correction signals
    # caused multi-category messages to collapse to the wrong single
    # category. They only count when paired closely with an actual
    # preference/action word.
    CORRECTION_RE = re.compile(
        r"\b(instead|wait|i meant|but i want|but now|change to|"
        r"switch to|forget the|scratch that|no,?\s+i want|not \w+ but|"
        r"actually\b(?:\s+\w+){0,3}\s+(?:want|prefer|do|try)|"
        r"rather\b(?:\s+\w+){0,3}\s+(?:than|do|try|have))\b",
        re.IGNORECASE,
    )
    for msg in reversed(user_msgs):
        msg_lower = msg["content"].lower()
        # Record the position of the first occurrence of each category keyword.
        cat_positions: dict = {}   # cat -> earliest char position in message
        seen_in_msg: set = set()
        for kw, cat in category_keywords:
            pos = msg_lower.find(kw)
            if pos == -1 or cat in seen_in_msg:
                continue
            if cat == "Yoga" and "Chair Yoga" in seen_in_msg:
                continue
            cat_positions[cat] = pos
            seen_in_msg.add(cat)
        if not cat_positions:
            continue  # no category in this message; keep looking backwards
        if len(cat_positions) > 1 and CORRECTION_RE.search(msg_lower):
            # Multiple categories in a correction message — the user is
            # changing their mind.  Keep only the LAST-mentioned category
            # (the new preference comes after the correction signal).
            new_cat = max(cat_positions, key=lambda c: cat_positions[c])
            filters["categories"] = [new_cat]
        else:
            filters["categories"] = list(cat_positions.keys())
        break  # most-recent message with a category keyword wins

    # ── [EV2] Format (seated / standing / mix) ──────────────────────────
    for msg in reversed(user_msgs):
        msg_lower = msg["content"].lower()
        has_seated   = bool(re.search(r"\bseated\b",         msg_lower))
        has_standing = bool(re.search(r"\bstanding\b",       msg_lower))
        has_mix      = bool(re.search(r"\bmix\b|\bmixed\b",  msg_lower))
        if has_seated or has_standing or has_mix:
            if has_seated and not has_standing:
                filters["format"].append("Seated")
            if has_standing and not has_seated:
                filters["format"].append("Standing")
            if has_mix or (has_seated and has_standing):
                filters["format"].append("Mix")
            break  # most-recent message with a format keyword wins

    # ── [EV3] Duration ───────────────────────────────────────────────────
    # Primary: explicit numeric ranges (e.g. "15-20 min", "25 to 30").
    # Secondary: "a X-min" or "X-min one/workout" in workout context, covering
    # phrases like "a 15-min one", "show me a 10-minute workout".  Avoids
    # false positives like "I walk 15 minutes a day" (no article before the
    # duration, no workout noun after it).
    # Newest-first: stop at the first message that contains any duration signal.
    def _minutes_to_bracket(mins: int):
        if mins <= 14:  return (10, 15)
        if mins <= 20:  return (15, 20)
        if mins <= 30:  return (25, 30)
        return (31, 999)

    for msg in reversed(user_msgs):
        t = msg["content"].lower()
        if re.search(r"10\s*[-\u2013to]+\s*15", t):
            filters["duration_min"], filters["duration_max"] = 10, 15; break
        if re.search(r"15\s*[-\u2013to]+\s*20", t):
            filters["duration_min"], filters["duration_max"] = 15, 20; break
        if re.search(r"25\s*[-\u2013to]+\s*30", t):
            filters["duration_min"], filters["duration_max"] = 25, 30; break
        if re.search(
            r"30\s*\+|30\s*plus|30\s*or\s*more|over\s+30|more\s+than\s+30|longer\s+than\s+30",
            t,
        ):
            filters["duration_min"], filters["duration_max"] = 31, 999; break
        # Secondary: article+duration or duration+workout-noun
        m = (
            re.search(r"\ba\s+(\d+)[\s-]min", t)                              # "a 15-min"
            or re.search(r"\b(\d+)[\s-]min(?:ute)?s?\s+(?:one|workout|video|session)\b", t)  # "15-min one"
        )
        if m:
            filters["duration_min"], filters["duration_max"] = _minutes_to_bracket(int(m.group(1)))
            break

    # ── [EV4] Movement exclusions ────────────────────────────────────────
    # Only accumulate exclusions from user messages that came AFTER the EV4
    # question was asked.  Scanning the full history caused false positives —
    # e.g. "I need to balance my diet" or "I'm jumping into a new routine"
    # from early in the conversation incorrectly added exercise exclusions.
    _all_msgs_for_ev4 = history[-MAX_HISTORY_STORED:]
    _ev4_asked_at = None
    for _j, _m in enumerate(_all_msgs_for_ev4):
        if _m.get("role") == "assistant":
            _c = _m.get("content", "").lower()
            if "balanc" in _c and "jumping" in _c and "kneeling" in _c:
                _ev4_asked_at = _j
                break

    if _ev4_asked_at is not None:
        _post_ev4_user_msgs = [
            _m for _m in _all_msgs_for_ev4[_ev4_asked_at:]
            if isinstance(_m, dict) and _m.get("role") == "user"
        ]
        _ev4_text = " ".join(_m["content"].lower() for _m in _post_ev4_user_msgs)
        if re.search(r"\bbalanc", _ev4_text) and "balance" not in filters["exclusions"]:
            filters["exclusions"].append("balance")
        if re.search(r"\bjumping\b", _ev4_text) and "jump" not in filters["exclusions"]:
            filters["exclusions"].append("jump")
        if re.search(r"\bkneel", _ev4_text) and "kneel" not in filters["exclusions"]:
            filters["exclusions"].append("kneel")

    return filters


def _match_exercise_videos(filters: dict, difficulty: str) -> tuple:
    """
    Match EXERCISE_VIDEOS against user preferences with progressive fallback.
    Hard movement exclusions are always enforced.

    Returns (videos, fallback_level) where fallback_level is:
      0 = all filters matched exactly
      1 = format relaxed
      2 = duration relaxed
      3 = difficulty relaxed
      4 = category relaxed (any category)
      -1 = no videos loaded or no categories specified
    """
    if not EXERCISE_VIDEOS:
        return [], -1

    categories_lower  = [c.lower() for c in (filters.get("categories") or [])]

    # Don't surface videos until the user has answered at least [EV1] (category
    # preference).  Without categories we'd fall through to level-4 fallback and
    # return arbitrary videos with no relevance to what the user wants.
    if not categories_lower:
        return [], -1

    format_lower      = [f.lower() for f in (filters.get("format")      or [])]

    # "Chair Yoga" is NOT a distinct value in the CSV's category column —
    # every chair-yoga video (and every mat/standing yoga video) is tagged
    # category "Yoga"; chair yoga is only distinguishable by format (all
    # chair yoga rows resolve to format "Seated", see
    # _parse_exercise_title's category-based fallback). Without this
    # translation, a user who explicitly asks for chair yoga would never
    # match on category == "chair yoga" (nothing in EXERCISE_VIDEOS has
    # that category value), so cat_ok() would fail even at level 0-2 and
    # the progressive fallback would relax all the way to level 4 —
    # silently handing back irrelevant Bodyweight/Dumbbell videos instead
    # of the seated yoga videos that actually satisfy the request.
    if "chair yoga" in categories_lower:
        categories_lower = ["yoga" if c == "chair yoga" else c for c in categories_lower]
        if "seated" not in format_lower:
            format_lower = format_lower + ["seated"]

    dur_min           = filters.get("duration_min")
    dur_max           = filters.get("duration_max")
    exclusions        = filters.get("exclusions") or []
    difficulty_lower  = (difficulty or "").lower()

    def passes_exclusions(v: dict) -> bool:
        """Hard exclusions — never relaxed."""
        tl = v["title"].lower()
        if "balance" in exclusions and re.search(r"\bbalanc|single.?leg", tl):
            return False
        if "jump" in exclusions and re.search(r"\bjump|\bhop|\bplyometric", tl):
            return False
        if "kneel" in exclusions and re.search(r"\bkneel", tl):
            return False
        return True

    def cat_ok(v):  return not categories_lower  or v["category"].lower() in categories_lower
    def diff_ok(v): return not difficulty_lower   or v["difficulty"].lower() == difficulty_lower
    def dur_ok(v):
        if dur_min is None or dur_max is None:
            return True
        d = v.get("duration_minutes")
        return d is None or (dur_min <= d <= dur_max)
    def fmt_ok(v):
        if not format_lower:
            return True
        fmt = (v.get("format") or "").lower()
        return not fmt or fmt in format_lower or "mix" in format_lower

    eligible = [v for v in EXERCISE_VIDEOS if passes_exclusions(v)]

    for level in range(5):
        if   level == 0: results = [v for v in eligible if cat_ok(v) and diff_ok(v) and dur_ok(v) and fmt_ok(v)]
        elif level == 1: results = [v for v in eligible if cat_ok(v) and diff_ok(v) and dur_ok(v)]
        elif level == 2: results = [v for v in eligible if cat_ok(v) and diff_ok(v)]
        elif level == 3: results = [v for v in eligible if cat_ok(v)]
        else:            results = eligible
        if results:
            return results[:MAX_EXERCISE_VIDEOS], level

    return [], -1


def _ev4_was_asked(history: list) -> bool:
    """
    Return True if the chatbot has already asked the EV4 movement-exclusions
    question in this conversation.

    Detection: looks for the distinctive triple of 'balanc', 'jumping', and
    'kneeling' in any single assistant message.  Using 'balanc' (prefix) covers
    both 'balance' and 'balancing' regardless of how the chatbot phrases it.

    Accepts the full sanitized history (not just truncated_history) so the
    gate stays open even in long conversations where EV4 was asked > 20
    turns ago.
    """
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").lower()
        if "balanc" in content and "jumping" in content and "kneeling" in content:
            return True
    return False


# Exercise-intent signal words for the per-turn relevance gate below. Mirrors
# category/format vocabulary from _detect_exercise_filters plus general
# workout/exercise language, so we can tell "this turn is about exercise"
# apart from "the user mentioned a category once, several turns ago."
_EXERCISE_INTENT_RE = re.compile(
    r"\b(exercise|workout|work[\s-]?out|video|videos|routine|move more|"
    r"physical activity|bodyweight|body weight|dumbbell|dumbbells|"
    r"resistance band|resistance bands|hand weight|chair yoga|tai chi|yoga|"
    r"seated|standing|stretch|cardio|beginner|intermediate|advanced)\b",
    re.IGNORECASE,
)

# Short affirmations / "give me another" follow-ups that carry no exercise
# vocabulary of their own but should still surface videos when they come
# right after the assistant offered some — e.g. "yes", "show me another".
# Deliberately narrow and short-message-only (see length guard below) so a
# topic-changing message that happens to contain "yes" or "ok" doesn't get
# misread as a continuation — e.g. "yes I know but actually I want to talk
# about my sleep" must NOT gate videos back on.
_EXERCISE_CONTINUATION_RE = re.compile(
    r"^\s*(yes|yeah|yep|sure|ok(?:ay)?|please|"
    r"(?:show me |can you show me )?(?:more|another)(?: one)?s?|"
    r"next (?:one|video)|that works|sounds good)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _exercise_turn_is_relevant(user_message: str, history: list) -> bool:
    """
    Return True only if THIS turn is actually about exercise, so
    `exercise_videos` isn't attached to every response for the rest of the
    conversation just because the user answered the EV1–EV4 intake earlier.

    Without this check, `min_filters_set and _ev4_was_asked(history)` alone
    stays true forever once intake is complete, so a later turn about sleep
    or diet would still carry a populated exercise_videos array even though
    the assistant's reply (per the system prompt's gating rule) correctly
    says nothing about videos — a frontend/backend mismatch where cards
    render for an unrelated message.

    Mirrors `_animation_matches_conversation`'s per-turn topic gating for
    animation cards, adapted to exercise vocabulary. A turn counts as
    relevant if the current user message itself mentions exercise, OR if the
    current message is a short continuation/affirmation (see
    _EXERCISE_CONTINUATION_RE) AND the assistant's immediately preceding
    message was about exercise. The continuation check is intentionally
    narrow — without it, ANY message right after an exercise turn (including
    a topic change like "actually, how's my sleep looking?") would
    incorrectly keep surfacing videos just because the prior assistant
    message mentioned them.
    """
    if _EXERCISE_INTENT_RE.search(user_message or ""):
        return True

    if _EXERCISE_CONTINUATION_RE.match((user_message or "").strip()):
        last_assistant = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "") or ""
                break
        return bool(_EXERCISE_INTENT_RE.search(last_assistant))

    return False


def _build_exercise_match_note(filters: dict, difficulty: str, fallback_level: int, videos: list) -> str:
    """
    Build a mismatch instruction injected as a late system message so the
    model acknowledges unavailable videos honestly.  Returns empty string
    when the match was exact (fallback_level == 0) or no videos were found.
    """
    if fallback_level <= 0 or not videos:
        return ""

    cats     = ", ".join(filters.get("categories") or [])
    dur_min  = filters.get("duration_min")
    dur_max  = filters.get("duration_max")
    fmts     = ", ".join(filters.get("format") or [])

    # Human-readable duration label
    if dur_min and dur_max:
        dur_label = "30+ min" if dur_max >= 999 else f"{dur_min}\u2013{dur_max} min"
    else:
        dur_label = None

    # What was actually found — sanitize CSV-derived values before injecting
    # into the system message (defense in depth against unexpected CSV content).
    found_durs = sorted({v["duration_minutes"] for v in videos if v.get("duration_minutes")})
    found_dur_str = ", ".join(f"{d} min" for d in found_durs) if found_durs else "varying lengths"
    found_cats = _sanitize_prompt_str(
        ", ".join(sorted({v["category"] for v in videos})), 80
    )

    requested_desc = cats or "any category"
    if dur_label:   requested_desc += f", {dur_label}"
    if difficulty:  requested_desc += f", {difficulty} level"
    if fmts:        requested_desc += f", {fmts} format"

    if fallback_level == 1:
        # Format relaxed: have category + duration, just not the requested format
        found_fmts = ", ".join(sorted({v["format"] for v in videos if v.get("format")})) or "a different format"
        opener = (
            f"We don't have a {fmts} {cats} workout"
            f"{' in the ' + dur_label + ' range' if dur_label else ''} — "
            f"the closest option is {found_fmts}."
        )
        body = (
            f"1. YOUR VERY FIRST SENTENCE: \"{opener}\"\n"
            f"2. Present the video positively — it IS the right category and duration.\n"
            f"3. CRITICAL: Do NOT call this a '{fmts}' workout or imply it is "
            f"'{fmts}'. It is {found_fmts}. Contradicting this confuses the user.\n"
            f"4. Do NOT suggest a different category."
        )
    elif fallback_level == 2:
        # Duration relaxed: have category, but not at this duration
        opener = f"We don't currently have {cats} workouts in the {dur_label} range."
        body = (
            f"1. YOUR VERY FIRST SENTENCE: \"{opener}\"\n"
            f"2. Offer the closest {cats} options available ({found_dur_str}) as a solid alternative.\n"
            f"3. Do NOT suggest a different category \u2014 we have {cats}, just not at that duration."
        )
    elif fallback_level == 3:
        # Difficulty relaxed: have category + duration, just not at the inferred difficulty
        found_diffs = ", ".join(sorted({v["difficulty"] for v in videos}))
        opener = (
            f"We have {cats} workouts{' in the ' + dur_label + ' range' if dur_label else ''} "
            f"but not at {difficulty} level \u2014 surfacing {found_diffs} options instead."
        )
        body = (
            f"1. Briefly note that {difficulty}-level {cats} workouts aren't available"
            f"{' in the ' + dur_label + ' range' if dur_label else ''}, "
            f"but {found_diffs} options are.\n"
            f"2. Present the video positively \u2014 it IS the right category"
            f"{' and duration' if dur_label else ''}.\n"
            "3. Do NOT suggest a different category or workout type."
        )
    else:
        # Category/all relaxed: nothing matched, surfacing alternatives
        opener = f"We don't have {cats} workouts matching your preferences right now."
        body = (
            f"1. YOUR VERY FIRST SENTENCE: \"{opener}\"\n"
            f"2. Present the closest alternative available (categories: {found_cats}).\n"
            "3. Offer to adjust preferences if the alternative doesn't suit them."
        )

    dur_clause = f" in the {dur_label} range" if dur_label else ""
    return (
        f"EXERCISE VIDEO MISMATCH \u2014 YOU MUST FOLLOW THESE INSTRUCTIONS:\n"
        f"The user's CURRENT requested category: {cats}\n"
        f"The user requested: {requested_desc}\n"
        f"What was actually surfaced: {found_cats}, {found_dur_str}\n\n"
        f"REQUIRED RESPONSE STRUCTURE:\n"
        f"{body}\n"
        "4. You may add exercise tips from the health literature context \u2014 "
        "do not invent or cite anything not in that context.\n"
    )


# ---------------------------------------------------------------------------
# Deterministic LE8 value scoring.
#
# The LE8 score tiers/thresholds live in the system prompt as a reference
# table for the model to use, but when a user states a raw value in chat
# (e.g. "my HbA1c is 6.0%") the model was doing that arithmetic itself and
# getting it wrong (misreading which tier a value falls in, ignoring a
# self-reported diabetes diagnosis, etc). These helpers compute the score
# in Python and the result is injected as an authoritative system note so
# the model reports it rather than recalculating it.
# ---------------------------------------------------------------------------

_HBA1C_RE           = re.compile(r"hba1c[^%\d]{0,20}(\d{1,2}(?:\.\d+)?)\s*%?", re.IGNORECASE)
_FASTING_GLUCOSE_RE = re.compile(r"fasting\s+gl?ucose[^%\d]{0,20}(\d{2,3}(?:\.\d+)?)", re.IGNORECASE)
_NON_HDL_RE         = re.compile(r"non[\s-]?hdl[^%\d]{0,25}(\d{2,3}(?:\.\d+)?)", re.IGNORECASE)
_CHOLESTEROL_SCORE_RE = re.compile(
    r"cholesterol\s*(?:score)?\s*(?:is|of|=|:)?\s*(\d{1,3})\b", re.IGNORECASE
)

_DIABETES_POSITIVE_RE = re.compile(
    r"\bi(?:'m| am)?\s*(?:a\s+)?diabetic\b|\bi\s+have\s+diabetes\b|"
    r"\bdiagnosed with diabetes\b|\bmy diabetes\b",
    re.IGNORECASE,
)
_DIABETES_NEGATIVE_RE = re.compile(
    r"\bi\s*(?:don'?t|do not)\s*have\s+diabetes\b|\bnot\s+diabetic\b|\bno\s+diabetes\b",
    re.IGNORECASE,
)


def _score_hba1c(value: float, has_diabetes: bool) -> int:
    if has_diabetes:
        if value < 7:  return 40
        if value < 8:  return 30
        if value < 9:  return 20
        if value < 10: return 10
        return 0
    if value < 5.7: return 100
    if value < 6.5: return 60
    return 0


def _score_fasting_glucose(value: float) -> int:
    # The LE8 reference only defines a diabetic-specific scale for HbA1c,
    # not fasting glucose — this is always the non-diabetic scale.
    if value < 100: return 100
    if value < 126: return 60
    return 0


def _score_non_hdl(value: float) -> int:
    if value < 130: return 100
    if value < 160: return 60
    if value < 190: return 40
    if value < 220: return 20
    return 0


def _non_hdl_range_for_score(score: int) -> str | None:
    """Reverse-map an LE8 Blood Lipids score back to its mg/dL range, for
    when a user quotes their score instead of the raw lab value."""
    return {
        100: "under 130 mg/dL",
        60:  "130-159 mg/dL",
        40:  "160-189 mg/dL",
        20:  "190-219 mg/dL",
        0:   "220 mg/dL or higher",
    }.get(score)


def _detect_diabetes_status(text_msgs: list) -> bool | None:
    """
    Scan a list of user message strings (oldest first) for a self-reported
    diabetes diagnosis. The most recent explicit statement wins, so a later
    correction overrides an earlier one. Returns None if never mentioned.
    """
    status = None
    for text in text_msgs:
        if _DIABETES_NEGATIVE_RE.search(text):
            status = False
        elif _DIABETES_POSITIVE_RE.search(text):
            status = True
    return status


def _build_computed_value_note(user_message: str, history: list, le8_data: dict) -> str:
    """
    Extract any raw lab values / diabetes status the user has stated across
    the conversation (including this turn) and return a system note with
    the exact, pre-computed LE8 score for each — so the model reports a
    number instead of recalculating it (and getting it wrong).
    Returns "" if nothing relevant was found.
    """
    user_texts = [m["content"] for m in history if m.get("role") == "user"] + [user_message]
    full_text  = " ".join(user_texts)

    diabetes_status = _detect_diabetes_status(user_texts)
    if diabetes_status is None:
        bs_payload = ((le8_data or {}).get("metrics") or {}).get("blood_sugar") or {}
        if isinstance(bs_payload, dict):
            diabetes_status = bs_payload.get("has_diabetes")

    lines = []

    m = _HBA1C_RE.search(full_text)
    if m:
        value   = float(m.group(1))
        has_d   = bool(diabetes_status)
        score   = _score_hba1c(value, has_d)
        tier    = _le8_tier(score)
        scale   = "the diabetic scale (40-pt max)" if has_d else "the non-diabetic scale"
        lines.append(
            f"COMPUTED VALUE — the user's HbA1c is {value}%. Using {scale}, this scores "
            f"EXACTLY {score}/100 ({tier} tier). Report this score and tier precisely; do not "
            f"recalculate it yourself. Do NOT tell the user whether they 'have' or 'don't have' "
            f"diabetes based on this number — that diagnosis belongs to their doctor, only report "
            f"how the app scores the value they gave you."
            + ("" if has_d else " If the user has told you (in this conversation) that they have "
               "a diabetes diagnosis, you MUST use the diabetic scale instead — do not silently "
               "re-evaluate them against the non-diabetic thresholds or contradict their "
               "self-reported diagnosis.")
        )

    m = _FASTING_GLUCOSE_RE.search(full_text)
    if m:
        value = float(m.group(1))
        score = _score_fasting_glucose(value)
        tier  = _le8_tier(score)
        lines.append(
            f"COMPUTED VALUE — the user's fasting glucose is {value} mg/dL. This scores "
            f"EXACTLY {score}/100 ({tier} tier). Report this precisely; do not recalculate it."
        )

    m = _NON_HDL_RE.search(full_text)
    if m:
        value = float(m.group(1))
        score = _score_non_hdl(value)
        tier  = _le8_tier(score)
        lines.append(
            f"COMPUTED VALUE — the user's non-HDL cholesterol is {value} mg/dL. This scores "
            f"EXACTLY {score}/100 ({tier} tier). Report this precisely; do not recalculate it."
        )
    else:
        m = _CHOLESTEROL_SCORE_RE.search(user_message)
        if m:
            score      = int(m.group(1))
            range_str  = _non_hdl_range_for_score(score)
            if range_str:
                tier = _le8_tier(score)
                lines.append(
                    f"COMPUTED VALUE — the user says their LE8 Blood Lipids score is {score}/100 "
                    f"({tier} tier). That corresponds to a non-HDL cholesterol of {range_str}. "
                    f"State this range. Do NOT tell them whether it is 'dangerous' — that is a "
                    f"clinical judgment for their care team. Do note it's worth discussing with "
                    f"their care team, and suggest lifestyle levers (soluble fiber, unsaturated "
                    f"fats, reduced saturated fat, physical activity) that can help move it."
                )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LE8 helpers
# ---------------------------------------------------------------------------

def _le8_tier(score) -> str:
    """Map a 0-100 LE8 metric score (int or float) to its AHA tier label."""
    if score >= 80:
        return "Ideal"
    if score >= 50:
        return "Intermediate"
    return "Low"


def _build_le8_section(le8_data: dict) -> str:
    """
    Convert the le8_data payload sent by the frontend into a formatted
    string for injection into the system prompt.

    The payload shape during testing is a hardcoded sample object on the
    frontend. When integrating with mHealthy Hearts, the frontend swaps
    that one constant for the result of GET /api/health-scores — nothing
    in Flask needs to change.

    Expected payload shape:
    {
      "composite_score": <int | null>,
      "metrics": {
        "physical_activity": { "steps": int, "goal": int, "score": int } | null,
        "sleep":             { "hours": float, "score": int } | null,
        "blood_pressure":    { "systolic": int, "diastolic": int, "score": int } | null,
        "blood_sugar":       { "test_type": str, "value": float, "unit": str,
                               "has_diabetes": bool, "score": int } | null,
        "blood_lipids":      { "non_hdl": float, "unit": str, "score": int } | null,
        "bmi":               { "height_in": float, "weight_lbs": float,
                               "bmi_value": float, "score": int } | null,
        "diet":              { "mepa_score": int, "score": int } | null,
        "smoking":           { "status": str, "secondhand_exposure": bool,
                               "score": int } | null,
      }
    }
    Null metrics are excluded from the composite and flagged as not yet assessed.
    """
    if not le8_data or not isinstance(le8_data, dict):
        return ""

    metrics   = le8_data.get("metrics") or {}
    composite = le8_data.get("composite_score")

    lines   = []
    missing = []

    # 1. Physical Activity
    pa = metrics.get("physical_activity")
    if pa and pa.get("score") is not None:
        score     = pa["score"]
        steps     = pa.get("steps", "N/A")
        goal      = pa.get("goal", 10000)
        steps_fmt = f"{steps:,}" if isinstance(steps, int) else str(steps)
        goal_fmt  = f"{goal:,}"  if isinstance(goal,  int) else str(goal)
        lines.append(
            f"  Physical Activity: {steps_fmt} steps today (goal: {goal_fmt}) "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Physical Activity")

    # 2. Sleep
    sl = metrics.get("sleep")
    if sl and sl.get("score") is not None:
        score = sl["score"]
        hours = sl.get("hours", "N/A")
        lines.append(
            f"  Sleep: {hours} hrs last night "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Sleep")

    # 3. Blood Pressure
    bp = metrics.get("blood_pressure")
    if bp and bp.get("score") is not None:
        score   = bp["score"]
        sys_val = bp.get("systolic",  "N/A")
        dia_val = bp.get("diastolic", "N/A")
        lines.append(
            f"  Blood Pressure: {sys_val}/{dia_val} mmHg "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Blood Pressure")

    # 4. Blood Sugar
    bs = metrics.get("blood_sugar")
    if bs and bs.get("score") is not None:
        score        = bs["score"]
        test_type    = bs.get("test_type", "unknown")
        value        = _safe_numeric(bs.get("value"))
        unit         = _sanitize_prompt_str(bs.get("unit", "mg/dL"), 20)
        has_diabetes = bs.get("has_diabetes", False)
        test_label   = "Fasting Glucose" if test_type == "fasting_glucose" else "HbA1c"
        diab_note    = " (has diabetes)" if has_diabetes else ""
        lines.append(
            f"  Blood Sugar: {test_label} {value} {unit}{diab_note} "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Blood Sugar")

    # 5. Blood Lipids
    bl = metrics.get("blood_lipids")
    if bl and bl.get("score") is not None:
        score   = bl["score"]
        non_hdl = _safe_numeric(bl.get("non_hdl"))
        unit    = _sanitize_prompt_str(bl.get("unit", "mg/dL"), 20)
        lines.append(
            f"  Blood Lipids (Non-HDL Cholesterol): {non_hdl} {unit} "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Blood Lipids")

    # 6. BMI
    bmi_data = metrics.get("bmi")
    if bmi_data and bmi_data.get("score") is not None:
        score   = bmi_data["score"]
        bmi_val = bmi_data.get("bmi_value", "N/A")
        lines.append(
            f"  BMI: {bmi_val} "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("BMI")

    # 7. Diet
    diet_data = metrics.get("diet")
    if diet_data and diet_data.get("score") is not None:
        score = diet_data["score"]
        mepa  = diet_data.get("mepa_score", "N/A")
        lines.append(
            f"  Diet (MEPA): {mepa}/10 "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Diet")

    # 8. Smoking / Nicotine
    smk = metrics.get("smoking")
    if smk and smk.get("score") is not None:
        score      = smk["score"]
        status_map = {
            "never":           "Never smoked",
            "quit_5plus":      "Quit 5+ years ago",
            "quit_1_4":        "Quit 1-4 years ago",
            "quit_under_1":    "Quit under 1 year ago",
            "current_rarely":  "Current (rarely)",
            "current_regular": "Current (regularly)",
        }
        raw_status   = _sanitize_prompt_str(smk.get("status", ""), 30)
        status_label = status_map.get(raw_status, "Unknown")
        sh_note = " + secondhand exposure in home (-20 pts applied)" \
                  if smk.get("secondhand_exposure") else ""
        lines.append(
            f"  Smoking/Nicotine: {status_label}{sh_note} "
            f"-> Score: {score}/100 ({_le8_tier(score)})"
        )
    else:
        missing.append("Smoking/Nicotine")

    # Build composite header
    if composite is not None:
        header = (
            f"USER'S LIFE'S ESSENTIAL 8 (LE8) SCORES\n"
            f"Composite Heart Score: {composite}/100 ({_le8_tier(composite)})\n"
        )
    else:
        header = (
            "USER'S LIFE'S ESSENTIAL 8 (LE8) SCORES\n"
            "Composite Heart Score: Incomplete (one or more metrics not yet assessed)\n"
        )

    body         = "\n".join(lines) if lines else "  No metrics recorded yet."
    missing_note = (
        f"\n  NOT YET ASSESSED (excluded from composite): {', '.join(missing)}"
        if missing else ""
    )

    return f"\n{header}{body}{missing_note}\n"


def _load_mock_fitbit_data() -> dict | None:
    mock_path = os.path.join(_HERE, "mock_fitbit_data.json")
    try:
        with open(mock_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Mock Fitbit data file not found at %s", mock_path)
        return None
    except json.JSONDecodeError:
        logger.warning("Mock Fitbit data file is not valid JSON")
        return None


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time()
        timestamps = [t for t in rate_limit_store.get(ip, []) if now - t < RATE_LIMIT_WINDOW]
        if timestamps:
            rate_limit_store[ip] = timestamps
        elif ip in rate_limit_store:
            del rate_limit_store[ip]

        # Prune the store when it grows large to prevent unbounded memory use.
        # Removes entries whose most recent timestamp is outside the window.
        if len(rate_limit_store) > 10_000:
            stale = [
                k for k, v in rate_limit_store.items()
                if not v or (now - max(v)) > RATE_LIMIT_WINDOW
            ]
            for k in stale:
                del rate_limit_store[k]

        entry = rate_limit_store.get(ip, [])
        if len(entry) >= RATE_LIMIT_MAX:
            return jsonify({"error": "Too many requests. Please wait a moment."}), 429

        entry.append(now)
        rate_limit_store[ip] = entry
        return f(*args, **kwargs)
    return decorated


def sanitize_city(city: str) -> str | None:
    city = city.strip()[:100]
    if not city or not CITY_PATTERN.match(city):
        return None
    return city


def sanitize_history(history: list) -> list:
    clean = []
    # Hard cap on how many messages we'll accept from the client to prevent
    # history-stuffing attacks and excessive memory use in filter detection.
    for msg in history[-MAX_HISTORY_STORED:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:MAX_MESSAGE_LENGTH]})
    return clean


def _build_rag_query(user_message: str, history: list) -> str:
    """
    Build a richer embedding query by combining the current user message
    with the tail of recent conversation — but only when the message is
    genuinely decontextualized.

    A message needs enrichment when it's both short AND contains a
    back-reference signal (e.g. "tell me more about that", "what about it?").
    Self-contained questions like "What exercises should I do today?" are
    used as-is regardless of word count, preventing prior-topic context
    (e.g. a nutrition answer) from polluting the embedding and surfacing
    irrelevant video cards.
    """
    word_count = len(user_message.split())
    is_short = word_count <= 5
    is_followup = bool(FOLLOWUP_PATTERN.search(user_message))

    # Only enrich if the message is both short AND has a back-reference signal.
    # A long message is always self-contained.
    # A short message with no back-reference is still a standalone question.
    if not (is_short and is_followup):
        return user_message

    # Find the most recent assistant message in history
    last_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")[:300]
            break

    if last_assistant:
        return f"{last_assistant} {user_message}"
    return user_message


# ---------------------------------------------------------------------------
# Animation topic-relevance guard
# ---------------------------------------------------------------------------

# Words too generic to use as topic signals — they appear across all health
# domains and would cause false positives if used for overlap matching.
_ANIM_STOPWORDS: frozenset = frozenset({
    "that", "this", "they", "them", "their", "have", "with", "from",
    "about", "what", "when", "where", "will", "your", "more", "some",
    "been", "does", "into", "than", "then", "also", "just", "like",
    "each", "much", "most", "make", "such", "know", "well", "help",
    "need", "want", "feel", "time", "week", "days", "would", "could",
    "should", "there", "these", "those", "were", "here", "okay",
    "great", "sure", "good", "think", "really", "even", "going",
    # Domain-neutral health words — present in every conversation
    "goal", "score", "level", "health", "heart", "cancer", "patient",
    "body", "care", "life", "risk", "data", "high", "lower", "better",
    "improve", "increase", "reduce", "change", "start", "help", "work",
})


def _topic_words(text: str) -> set:
    """
    Extract meaningful content words (≥4 chars, not stopwords) from text.
    Used to compare the recent conversation topic against an animation title.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    return {w for w in words if w not in _ANIM_STOPWORDS}


def _animation_matches_conversation(
    anim_title: str,
    history: list,
    current_message: str,
    window: int = 10,
) -> bool:
    """
    Return True if the animation section title shares at least one meaningful
    keyword with the recent conversation, False otherwise.

    This prevents cross-topic animation cards — e.g. a "Sleep Hygiene" card
    surfacing mid-way through a Physical Activity MI intake — by requiring
    that some word in the animation title also appears somewhere in the last
    `window` messages.  Generic health words are excluded from the comparison
    via _ANIM_STOPWORDS so they don't create false matches.

    If the animation title has no meaningful keywords (e.g. a very short or
    generic title), we allow it through rather than silently suppressing it.
    """
    title_words = _topic_words(anim_title)
    if not title_words:
        return True  # can't determine topic → don't filter

    recent_msgs = list(history[-window:]) + [{"role": "user", "content": current_message}]
    conv_text = " ".join(m.get("content", "") for m in recent_msgs)
    conv_words = _topic_words(conv_text)

    if not conv_words:
        return True  # no conversation context yet → don't filter

    return bool(title_words & conv_words)


def retrieve_context(
    query: str,
    n_results: int = 7,
    include_references: bool = False,
) -> dict:
    """
    Query ChromaDB and return a formatted context string plus raw chunk details.

    Pool strategy: we fetch n_results*2 candidates from ChromaDB so the
    distance filter has a real pool to draw from. After filtering, we keep
    at most n_results chunks that pass RAG_DISTANCE_THRESHOLD.

    Relevance filtering: chunks whose cosine distance exceeds
    RAG_DISTANCE_THRESHOLD are excluded from context but still returned in
    chunk_details (with used_in_context=False) so callers can debug retrieval.

    Animation deduplication: each unique Vimeo URL appears in context at most
    once even if multiple chunks from the same section are retrieved.

    Reference injection: non-Vimeo reference URLs are only appended when
    include_references is True (the user explicitly asked for sources).
    """
    chroma_collection = _get_chroma_collection()
    count = chroma_collection.count()
    if count == 0:
        return {"context": "No literature has been ingested yet.", "chunks": [], "animations": []}

    response = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    query_embedding = response.data[0].embedding

    # Fetch a larger candidate pool so distance filtering has room to work
    fetch_count = min(n_results * 2, count)

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_count,
        include=["documents", "metadatas", "distances"],
    )

    raw_chunks = results["documents"][0]
    distances = results["distances"][0] if results.get("distances") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    ids = results["ids"][0] if results.get("ids") else []

    chunk_details = []
    context_parts = []
    animations: list = []
    seen_anim_urls: set = set()
    used_count = 0

    for i, chunk in enumerate(raw_chunks):
        distance = distances[i] if i < len(distances) else 1.0
        meta = (metadatas[i] if i < len(metadatas) else None) or {}

        ref_urls_raw = meta.get("reference_urls", "")
        ref_urls_list = [u for u in ref_urls_raw.split("|||") if u] if ref_urls_raw else []

        passes_threshold = distance <= RAG_DISTANCE_THRESHOLD
        at_result_limit = used_count >= n_results
        use_chunk = passes_threshold and not at_result_limit

        chunk_details.append({
            "id": ids[i] if i < len(ids) else f"chunk_{i}",
            "text": chunk[:300] + ("..." if len(chunk) > 300 else ""),
            "distance": round(distance, 4),
            "used_in_context": use_chunk,
            "metadata": {
                **{k: v for k, v in meta.items() if k != "reference_urls"},
                "reference_urls": ref_urls_list,
            },
        })

        # ----------------------------------------------------------------
        # Animation surfacing — uses ANIMATION_SURFACE_THRESHOLD (more
        # lenient than the context threshold) because script chunks embed
        # in a different stylistic register than research chunks and tend
        # to score slightly higher distances on the same health queries.
        # Cross-topic contamination is handled downstream by
        # _animation_matches_conversation() in chatbot() before the
        # animations list is sent to the client.
        # ----------------------------------------------------------------
        anim_url = meta.get("animation_url", "")
        section_title = meta.get("section_title", "")
        # Only surface animation cards for URLs that are genuine Vimeo links.
        # This prevents malformed or injected metadata from producing bad hrefs.
        anim_url_safe = (
            anim_url
            if isinstance(anim_url, str) and anim_url.startswith("https://vimeo.com")
            else ""
        )
        if (
            anim_url_safe
            and anim_url_safe not in seen_anim_urls
            and distance <= ANIMATION_SURFACE_THRESHOLD
            and len(animations) < MAX_ANIMATIONS_PER_RESPONSE
        ):
            animations.append({"title": section_title, "url": anim_url_safe})
            seen_anim_urls.add(anim_url_safe)

        if not use_chunk:
            continue

        used_count += 1
        block = chunk

        if include_references and ref_urls_list:
            refs_str = "\n".join(f"- {u}" for u in ref_urls_list[:5])
            block += f"\n\n\U0001f4da References for this section:\n{refs_str}"

        context_parts.append(block)

    # ----------------------------------------------------------------
    # Source diversity: if every used chunk came from "combined scripts.pdf",
    # run a second query that excludes that source and splice in the best
    # result that still clears SOURCE_DIVERSITY_THRESHOLD.  This ensures
    # the LLM can draw on research paper evidence when it is available.
    # ----------------------------------------------------------------
    used_sources = {
        cd["metadata"].get("source", "")
        for cd in chunk_details
        if cd["used_in_context"]
    }
    # NOTE: the actual transcript filename is "combined scripts.pdf" (with a
    # space, not an underscore).  Both checks must use the same spelling.
    script_only = bool(used_sources) and all(
        "combined scripts" in s.lower() for s in used_sources
    )
    if script_only:
        try:
            div_res = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"],
                where={"source": {"$ne": "combined scripts.pdf"}},
            )
            div_docs   = div_res["documents"][0]
            div_dists  = div_res["distances"][0]  if div_res.get("distances") else []
            div_metas  = div_res["metadatas"][0]  if div_res.get("metadatas") else []
            div_ids    = div_res["ids"][0]         if div_res.get("ids")       else []
            for j, div_doc in enumerate(div_docs):
                d    = div_dists[j] if j < len(div_dists) else 1.0
                meta = (div_metas[j] if j < len(div_metas) else None) or {}
                if d > SOURCE_DIVERSITY_THRESHOLD:
                    continue
                ref_raw  = meta.get("reference_urls", "")
                ref_list = [u for u in ref_raw.split("|||") if u] if ref_raw else []
                block    = div_doc
                if include_references and ref_list:
                    refs_str = "\n".join(f"- {u}" for u in ref_list[:5])
                    block   += f"\n\n\U0001f4da References for this section:\n{refs_str}"
                context_parts.append(block)
                chunk_details.append({
                    "id":             div_ids[j] if j < len(div_ids) else f"div_{j}",
                    "text":           div_doc[:300] + ("..." if len(div_doc) > 300 else ""),
                    "distance":       round(d, 4),
                    "used_in_context": True,
                    "metadata": {
                        **{k: v for k, v in meta.items() if k != "reference_urls"},
                        "reference_urls": ref_list,
                    },
                })
                break  # one diversity chunk is enough
        except Exception as e:
            logger.warning("Source diversity query failed: %s", e)

    if not context_parts:
        context_str = (
            "No sufficiently relevant information was found in the knowledge base "
            "for this query."
        )
    else:
        context_str = "\n\n---\n\n".join(context_parts)

    return {"context": context_str, "chunks": chunk_details, "animations": animations}


def _geocode_city(city: str) -> tuple | None:
    """
    Resolve a city name to (latitude, longitude, iana_timezone, display_name).

    display_name is the actual place Open-Meteo matched (e.g.
    "Legend, Alberta, Canada"), which is NOT necessarily what the user typed.
    Open-Meteo's search is a fuzzy/substring match against a global gazetteer
    with count=1 (top match only, no relevance score returned) — an unusual
    or made-up input can still return some obscure "best guess" locality with
    no signal that it's a poor match. We surface display_name so the system
    prompt can have the model state the resolved location back to the user
    instead of silently treating a low-confidence match as ground truth for
    their weather/local time.
    """
    try:
        res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        res.raise_for_status()
        data = res.json()
        results = data.get("results")
        if results:
            r = results[0]
            tz_str = r.get("timezone", "UTC")
            name_parts = [p for p in (r.get("name"), r.get("admin1"), r.get("country")) if p]
            # dict.fromkeys dedupes while preserving order (e.g. name == country edge case)
            display_name = ", ".join(dict.fromkeys(name_parts)) or city
            return r["latitude"], r["longitude"], tz_str, display_name
        return None
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", city, e)
        return None


def get_local_time(tz_str: str) -> str:
    """
    Return the current local time for an IANA timezone string
    (e.g. 'America/Chicago') as a human-readable string like '1:30 AM'.
    Falls back to UTC if the timezone is unrecognised.
    """
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo(tz_str))
        return now.strftime("%I:%M %p").lstrip("0")
    except Exception:
        logger.warning("Could not resolve timezone '%s', falling back to UTC", tz_str)
        return datetime.datetime.now(datetime.timezone.utc).strftime("%I:%M %p UTC").lstrip("0")


def get_weather(city: str = "Columbus", city_info=None) -> str:
    """
    Fetch NWS weather for a city. Accepts a pre-resolved city_info tuple
    (lat, lon, tz_str, display_name) from _geocode_city to avoid a redundant
    geocoding call when the caller already has it.
    """
    if city_info is not None:
        lat, lon, _, display_name = city_info
    else:
        city_info = _geocode_city(city)
        if city_info is None:
            return f"Weather data unavailable (could not locate '{city}')"
        lat, lon, _, display_name = city_info
    nws_headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    try:
        points_res = requests.get(
            f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
            headers=nws_headers,
            timeout=5,
        )
        points_res.raise_for_status()
        forecast_url = points_res.json()["properties"]["forecast"]
    except Exception as e:
        logger.warning("NWS points lookup failed: %s", e)
        return "Weather data unavailable (NWS only covers US locations)"

    try:
        forecast_res = requests.get(forecast_url, headers=nws_headers, timeout=5)
        forecast_res.raise_for_status()
        periods = forecast_res.json()["properties"]["periods"]
        current = periods[0]
        temp      = _safe_numeric(current.get("temperature"), "N/A")
        temp_unit = _sanitize_prompt_str(str(current.get("temperatureUnit", "F")), 5)
        forecast  = _sanitize_prompt_str(str(current.get("shortForecast",   "")), 60)
        wind_dir  = _sanitize_prompt_str(str(current.get("windDirection",    "")), 20)
        wind_spd  = _sanitize_prompt_str(str(current.get("windSpeed",        "")), 20)
        return (
            f"{display_name}: {temp}\u00b0{temp_unit}, "
            f"{forecast}, "
            f"wind {wind_dir} {wind_spd}"
        )
    except Exception as e:
        logger.warning("NWS forecast fetch failed: %s", e)
        return "Weather data unavailable"


def _pkce_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _basic_auth_header() -> str:
    return base64.b64encode(
        f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()
    ).decode()


def _fitbit_configured() -> bool:
    return bool(FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET)


def save_tokens(access_token: str, refresh_token: str, user_id: str | None = None) -> str | None:
    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
    }
    try:
        if user_id:
            result = collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": token_data},
            )
            if result.matched_count > 0:
                return user_id
            logger.warning("save_tokens: no document matched _id=%s", user_id)
            return None

        token_data["created_at"] = datetime.datetime.now(datetime.timezone.utc)
        result = collection.insert_one(token_data)
        return str(result.inserted_id)
    except InvalidId:
        logger.error("save_tokens: invalid ObjectId '%s'", user_id)
        return None
    except Exception as e:
        logger.error("Error saving tokens: %s", e)
        return None


def load_tokens(user_id: str | None = None):
    """
    Look up stored Fitbit tokens for a specific user document.

    SECURITY: user_id is required. There is intentionally no "most recently
    updated document" fallback here — this is a multi-tenant collection, so
    guessing at a document when no user_id is supplied would return whichever
    OTHER user connected Fitbit most recently, leaking their access/refresh
    tokens (and therefore their activity/sleep/heart-rate data) to the
    current caller. Every call site must resolve an actual user_id first.
    """
    if not user_id:
        return None, None, None
    try:
        document = collection.find_one({"_id": ObjectId(user_id)})

        if document:
            return (
                document.get("access_token"),
                document.get("refresh_token"),
                str(document["_id"]),
            )
        return None, None, None
    except InvalidId:
        logger.error("load_tokens: invalid ObjectId '%s'", user_id)
        return None, None, None
    except Exception as e:
        logger.error("Error loading tokens: %s", e)
        return None, None, None


def refresh_access_token(refresh_token: str, user_doc_id: str | None = None):
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": FITBIT_CLIENT_ID,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {_basic_auth_header()}",
    }

    try:
        res = requests.post(
            "https://api.fitbit.com/oauth2/token",
            data=data,
            headers=headers,
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        logger.error("Token refresh network error: %s", e)
        return None

    if res.status_code == 200:
        token_data = res.json()
        if user_doc_id:
            save_tokens(token_data["access_token"], token_data["refresh_token"], user_doc_id)
        return token_data

    logger.error("Token refresh failed with status %s (response body suppressed)", res.status_code)
    return None


def fetch_fitbit_summary(access_token: str) -> dict | None:
    today = datetime.date.today().isoformat()
    base = "https://api.fitbit.com/1/user/-"
    headers = {"Authorization": f"Bearer {access_token}"}

    summary = {}

    try:
        r = requests.get(f"{base}/activities/date/{today}.json", headers=headers, timeout=10)
        if r.status_code == 200:
            act = r.json().get("summary", {})
            summary["activity"] = {
                "steps": act.get("steps"),
                "calories_out": act.get("caloriesOut"),
                "active_minutes": (
                    act.get("fairlyActiveMinutes", 0) + act.get("veryActiveMinutes", 0)
                ),
                "distance_km": None,
            }
            for d in act.get("distances", []):
                if d.get("activity") == "total":
                    summary["activity"]["distance_km"] = d.get("distance")
    except Exception as e:
        logger.warning("Fitbit activity fetch failed: %s", e)

    try:
        r = requests.get(f"{base}/sleep/date/{today}.json", headers=headers, timeout=10)
        if r.status_code == 200:
            sleep_data = r.json().get("summary", {})
            summary["sleep"] = {
                "total_minutes_asleep": sleep_data.get("totalMinutesAsleep"),
                "total_time_in_bed": sleep_data.get("totalTimeInBed"),
            }
    except Exception as e:
        logger.warning("Fitbit sleep fetch failed: %s", e)

    try:
        r = requests.get(
            f"{base}/activities/heart/date/{today}/1d.json", headers=headers, timeout=10
        )
        if r.status_code == 200:
            hr_data = r.json().get("activities-heart", [])
            if hr_data:
                val = hr_data[0].get("value", {})
                summary["heart_rate"] = {
                    "resting_heart_rate": val.get("restingHeartRate"),
                }
    except Exception as e:
        logger.warning("Fitbit heart-rate fetch failed: %s", e)

    return summary if summary else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/authorize")
def authorize():
    if not _fitbit_configured():
        return jsonify({"error": "Fitbit integration is not configured"}), 503

    code_verifier = _pkce_code_verifier()
    code_challenge = _pkce_code_challenge(code_verifier)
    session["code_verifier"] = code_verifier

    # CSRF defense-in-depth: PKCE alone binds the auth code to whichever
    # session holds the matching code_verifier, but an explicit `state`
    # value is the standard OAuth CSRF control and protects against
    # implementation edge cases (e.g. a shared/reused session) where PKCE
    # binding isn't sufficient on its own. Store it server-side and verify
    # it round-trips unchanged in /callback.
    oauth_state = secrets.token_urlsafe(24)
    session["oauth_state"] = oauth_state

    params = {
        "response_type": "code",
        "client_id": FITBIT_CLIENT_ID,
        "redirect_uri": FITBIT_REDIRECT_URI,
        "scope": "activity heartrate sleep profile",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    return redirect(f"https://www.fitbit.com/oauth2/authorize?{urlencode(params)}")


@app.route("/callback")
def callback():
    if not _fitbit_configured():
        return jsonify({"error": "Fitbit integration is not configured"}), 503

    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

    expected_state = session.pop("oauth_state", None)
    returned_state = request.args.get("state")
    if not expected_state or not secrets.compare_digest(expected_state, returned_state or ""):
        logger.warning("Fitbit callback: state mismatch (possible CSRF attempt)")
        return jsonify({"error": "Invalid or expired authorization state. Please restart the authorization flow."}), 400

    code_verifier = session.get("code_verifier")
    if not code_verifier:
        return jsonify({"error": "Session expired. Please restart the authorization flow."}), 400

    headers = {
        "Authorization": f"Basic {_basic_auth_header()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "client_id": FITBIT_CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": FITBIT_REDIRECT_URI,
        "code": code,
        "code_verifier": code_verifier,
    }

    res = requests.post("https://api.fitbit.com/oauth2/token", headers=headers, data=data)
    if res.status_code == 200:
        tokens = res.json()
        user_doc_id = save_tokens(tokens["access_token"], tokens["refresh_token"])
        session["user_doc_id"] = user_doc_id
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return redirect(f"{frontend_url}?fitbit=connected")

    # Do not log res.text — it may contain sensitive auth details from Fitbit.
    logger.error("Fitbit callback error: status %s (response body suppressed)", res.status_code)
    return jsonify({"error": "Fitbit authorization failed"}), 400


@app.route("/health")
def health():
    try:
        col = _get_chroma_collection()
        chunk_count = col.count()
    except Exception as e:
        return jsonify({"status": "ok", "chroma": "error", "chroma_error": str(e)})
    return jsonify({"status": "ok", "chroma_chunks": chunk_count})


@app.route("/endpoint", methods=["POST"])
@rate_limit
def chatbot():
    body = request.get_json()
    if not body:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = body.get("message", "")
    if not isinstance(user_message, str):
        return jsonify({"error": "Invalid message"}), 400
    user_message = user_message.strip()[:MAX_MESSAGE_LENGTH]

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    raw_history = body.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []
    history = sanitize_history(raw_history)

    raw_city = body.get("city", "")
    if not isinstance(raw_city, str):
        raw_city = ""
    # No silent "Columbus" fallback: an unset/blank/invalid city must stay
    # None so downstream logic (weather, geocoding-failure messaging, the
    # system prompt) can tell "user never gave us a city" apart from an
    # actual request about Columbus.
    city = sanitize_city(raw_city)

    # -----------------------------------------------------------------------
    # LE8 data
    # During testing this is a hardcoded SAMPLE_LE8_DATA object on the
    # frontend. When integrating with mHealthy Hearts, the frontend swaps
    # that constant for the result of GET /api/health-scores — nothing
    # here needs to change.
    # -----------------------------------------------------------------------
    raw_le8 = body.get("le8_data")
    le8_data = raw_le8 if isinstance(raw_le8, dict) else {}

    # Build a richer query for short / decontextualized messages
    rag_query = _build_rag_query(user_message, history)

    # Detect whether the user is explicitly asking for research sources
    include_references = bool(REFERENCE_INTENT_PATTERN.search(user_message))

    # Deterministic crisis/self-harm detection — see CRISIS_PATTERN comment.
    is_crisis = bool(CRISIS_PATTERN.search(user_message))

    # Deterministic LE8 value scoring for anything the user stated in chat
    # (raw HbA1c/fasting glucose/non-HDL values or a quoted score) — see
    # _build_computed_value_note.
    computed_value_note = _build_computed_value_note(user_message, history, le8_data)

    # ---------------------------------------------------------------------------
    # Exercise video matching (runs before the LLM call so the match note can
    # be injected as a late system message for this turn).
    # Use the full sanitized history (up to MAX_HISTORY_STORED messages) for
    # both filter detection and the EV4 gate so long conversations don't lose
    # earlier preference answers or EV4 being asked >20 turns ago.
    # ---------------------------------------------------------------------------
    truncated_history   = history[-MAX_HISTORY_MESSAGES:]
    pre_turn_msgs       = history + [{"role": "user", "content": user_message}]
    curr_filters        = _detect_exercise_filters(pre_turn_msgs)
    exercise_difficulty = _infer_difficulty_from_le8(le8_data)
    # Only require a category to be set before surfacing videos.
    # Duration is optional — the fallback system handles mismatches gracefully
    # (level-2 fallback) and _build_exercise_match_note informs the LLM.
    # Requiring duration here caused videos to never surface when EV3 was
    # skipped or the user didn't explicitly specify a duration range.
    min_filters_set     = bool(curr_filters.get("categories"))
    # Per-turn relevance gate: even once intake (EV1-EV4) is complete, only
    # attach exercise_videos when THIS turn is actually about exercise (see
    # _exercise_turn_is_relevant docstring). Without this, every later turn
    # in the conversation — including ones about sleep, diet, or LE8 scores
    # — would carry a stale populated exercise_videos array.
    if (
        min_filters_set
        and _ev4_was_asked(history)
        and _exercise_turn_is_relevant(user_message, history)
    ):
        exercise_videos, fallback_level = _match_exercise_videos(curr_filters, exercise_difficulty)
        exercise_match_note = _build_exercise_match_note(
            curr_filters, exercise_difficulty, fallback_level, exercise_videos
        )
    else:
        exercise_videos     = []
        exercise_match_note = ""

    try:
        rag_result = retrieve_context(
            rag_query,
            include_references=include_references,
        )
        context = rag_result["context"]
        retrieved_chunks = rag_result["chunks"]
        animations = rag_result["animations"]
        rag_error = None
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        context = "Knowledge base temporarily unavailable."
        retrieved_chunks = []
        animations = []
        rag_error = str(e)

    # -----------------------------------------------------------------------
    # Animation topic-relevance filter.
    # The ANIMATION_SURFACE_THRESHOLD (0.82) is intentionally lenient because
    # script chunks embed in a different stylistic register than health
    # questions.  That leniency can cause off-topic cards (e.g. a sleep
    # animation surfacing mid Physical Activity MI intake) when the embedding
    # overlap is marginal and domain-unrelated.  We drop any animation whose
    # section title shares no meaningful keyword with the last 10 turns, which
    # is a cheap text-level guard that doesn't require an extra embedding call.
    # -----------------------------------------------------------------------
    if animations:
        animations = [
            a for a in animations
            if _animation_matches_conversation(
                a.get("title", ""), history, user_message
            )
        ]

    # Geocode once — result feeds both weather and local time so we never
    # hit the geocoding API twice for the same request.
    if city:
        city_info = _geocode_city(city)
        weather = get_weather(city, city_info=city_info)
    else:
        # No city provided (or an invalid one that failed sanitization) —
        # do NOT silently default to Columbus. Tell the model plainly so it
        # can ask the user for a city or give city-agnostic guidance instead
        # of fabricating/assuming a location.
        city_info = None
        weather = "Weather data unavailable (no city provided yet)"
    if city_info:
        time_str = get_local_time(city_info[2])
        resolved_location_line = (
            f"Resolved location: the city field \"{city}\" was matched to "
            f"{city_info[3]} (this is a single best-guess fuzzy match against "
            f"a global place database, not a verified address) — see the "
            f"LOCATION CONFIRMATION rule below.\n"
        )
    else:
        # Geocoding failed, or no city provided; fall back to UTC
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%I:%M %p UTC").lstrip("0")
        resolved_location_line = ""

    fitbit_section = ""
    fitbit_data = None

    if USE_MOCK_FITBIT:
        fitbit_data = _load_mock_fitbit_data()
    else:
        user_doc_id = session.get("user_doc_id")
        if user_doc_id and _fitbit_configured():
            try:
                ObjectId(user_doc_id)
            except (InvalidId, TypeError):
                user_doc_id = None
        if user_doc_id and _fitbit_configured():
            access_token, refresh_token, doc_id = load_tokens(user_doc_id)
            if access_token:
                fitbit_data = fetch_fitbit_summary(access_token)
                if fitbit_data is None and refresh_token:
                    new_tokens = refresh_access_token(refresh_token, doc_id)
                    if new_tokens:
                        fitbit_data = fetch_fitbit_summary(new_tokens["access_token"])

    if fitbit_data:
        fitbit_section = f"""
FITBIT DATA (today):
{json.dumps(fitbit_data, indent=2)}
"""

    # Build the LE8 section from the payload (empty string if no data sent)
    le8_section = _build_le8_section(le8_data)

    system_prompt = f"""You are a supportive, evidence-based cardiovascular health coach for people
living with or beyond cancer. Your primary mission is helping users understand and
improve their heart health through the American Heart Association's Life's Essential 8
(LE8) framework, alongside physical activity and nutrition guidance. You serve cancer
patients and survivors broadly — all cancer types, all treatment stages.

YOUR ROLE:
- Explain each of the user's LE8 scores in plain language: what the score means,
  why it is at that level given their raw values, and exactly what it would take
  to move it into a higher tier.
- Give actionable, specific level-up guidance tied to the user's actual numbers
  (e.g. "Your fasting glucose of 104 mg/dL is just inside the Intermediate range —
  getting it below 100 would move your Blood Sugar score from 60 to 100").
- Flag any metrics that are missing from the user's LE8 profile and encourage
  them to complete those assessments so their composite score is complete.
- Physical activity and exercise recommendations appropriate for cancer survivors,
  mindful of treatment side effects (fatigue, reduced exercise tolerance, muscle
  loss, lymphedema, neuropathy, etc.).
- Nutrition and healthy eating guidance grounded in the provided literature.
- SMART goal setting tied directly to specific LE8 metrics.
- Motivational Interviewing: open-ended questions, reflective listening,
  affirm effort and autonomy, never lecture or push.

LE8 SCORING REFERENCE
Use this section authoritatively for all score explanations and level-up guidance.
This does NOT require RAG support — the thresholds below are the source of truth.
If a "COMPUTED VALUE" system note appears later in this conversation for this turn,
that note has already done the lookup against these thresholds for a value the user
stated in chat — use its exact score/tier verbatim instead of recalculating it
yourself from the raw value.

Score tiers: 0-49 = Low | 50-79 = Intermediate | 80-100 = Ideal
Composite = average of all metrics that have data (missing metrics are excluded).

1. PHYSICAL ACTIVITY (steps from Fitbit)
   Score = (steps / goal) x 100, capped at 100. Default goal: 10,000 steps/day.
   Level up: each 1,000 additional steps adds ~10 points toward 100.

2. SLEEP (hours from Fitbit, previous night)
   Score = (hours / 8) x 100, capped at 100.
   Thresholds: 8+ hrs = 100 | 7.2 hrs ~ 90 | 6.5 hrs ~ 81 | 6.0 hrs = 75 | 5.0 hrs = 63
   Level up: target 8 hours. Even 30 extra minutes of consistent sleep adds ~6 points.

3. BLOOD PRESSURE (systolic/diastolic mmHg)
   <120 / <80   -> 100 (Ideal)
   120-129 / <80 -> 90
   130-139 OR 80-89 -> 75
   140-159 OR 90-99 -> 50
   >=160 OR >=100   -> 0
   Level up: reduce sodium, DASH-style eating, regular aerobic exercise, stress management.

4. BLOOD SUGAR
   No diabetes, fasting glucose (mg/dL): <100 -> 100 | 100-125 -> 60 | >=126 -> 0
   No diabetes, HbA1c (%):              <5.7 -> 100 | 5.7-6.4 -> 60 | >=6.5 -> 0
   With diabetes, HbA1c (max score 40): <7 -> 40 | 7-7.9 -> 30 | 8-8.9 -> 20 |
                                         9-9.9 -> 10 | >=10 -> 0
   Level up: reduce refined carbohydrates, increase dietary fiber, regular physical
   activity, manage body weight. Note: the jump from Intermediate (60) to Ideal (100)
   requires getting fasting glucose below 100 mg/dL — there is no in-between score.

5. BLOOD LIPIDS (Non-HDL Cholesterol mg/dL)
   <130  -> 100 | 130-159 -> 60 | 160-189 -> 40 | 190-219 -> 20 | >=220 -> 0
   Level up: increase soluble fiber (oats, beans, vegetables), choose healthy unsaturated
   fats, reduce saturated fat, increase physical activity.
   Note: like Blood Sugar, the jump from 60 to 100 requires getting below 130 mg/dL.

6. BMI (calculated as 703 x lbs / in^2)
   <25 -> 100 | 25-29.9 -> 70 | 30-34.9 -> 30 | 35-39.9 -> 15 | >=40 -> 0
   Important cancer context: treatment side effects (steroids, hormone therapy, muscle
   loss from chemo) can affect weight and BMI in ways outside the user's control.
   Acknowledge this sensitivity. Do NOT recommend aggressive caloric restriction for
   cancer patients — focus on sustainable, nourishing eating and gentle activity.

7. DIET (MEPA score, 10 diet-quality questions, 1 pt each)
   8-10 pts -> 100 | 6-7 -> 80 | 4-5 -> 50 | 2-3 -> 25 | 0-1 -> 0
   Level up: identify 1-2 specific healthy behaviors the user can realistically add.
   Each additional MEPA point gained can move the score tier upward.

8. SMOKING / NICOTINE
   Never smoked                -> 100
   Quit 5+ years ago           -> 100
   Quit 1-4 years ago          -> 75
   Quit under 1 year ago       -> 50
   Current smoker (rarely)     -> 25
   Current smoker (regularly)  -> 0
   Secondhand exposure in home -> deduct 20 pts, floor at 0.
   IMPORTANT: a never-smoker with household secondhand exposure scores 80, not 100.
   Always explain this when it applies — it surprises people.
   Level up for current smokers: cessation support, nicotine replacement therapy,
   gradual reduction. Quitting entirely moves the score to at least 50 immediately,
   and to 75 after one year.

BEHAVIOR GUIDELINES:
- Always contextualize LE8 advice within cancer survivorship. Treatment effects
  (fatigue, hormonal changes, neuropathy, immune suppression) are real barriers —
  acknowledge them, do not dismiss them.
- Redirect clinical concerns (specific symptoms, treatment decisions, medication
  interactions, supplement dosages) to the care team.
- Never diagnose, prescribe, or contradict medical advice.
- If a user appears to be in crisis or mentions self-harm, respond with empathy
  and direct them to the 988 Suicide and Crisis Lifeline or their care team.
- Do not exit SMART Goal Mode mid-intake if the user asks a tangential question.
  Answer it briefly, then return to the next unfilled intake field.
  Example: "Great question — [brief answer]. Getting back to your goal —
  I still need to ask about [next field]."

CONNECTING FITBIT:
- If asked how to connect Fitbit, describe the real in-app flow: they click
  "Connect Fitbit" (or open the Fitbit connection option) in the app, which
  sends them to Fitbit's own authorization page. Once they approve access
  there (to activity, heart rate, sleep, and profile data), Fitbit redirects
  them back and the app is connected automatically — no extra setup needed.
- Do NOT say you're unable to help with app connectivity or redirect to a
  generic "help section" — this exact flow exists and you can describe it.

EXERCISE VIDEO LINKS:
- Video cards are surfaced automatically by the system alongside your reply
  (see EXERCISE VIDEO PROTOCOL below) — you do not send raw links yourself,
  but the app genuinely does show real Vimeo video cards. Don't deny that the
  app sends video links; only clarify that you personally don't paste URLs.
- If a user reports that a video link/card isn't working or the video was
  removed, do not insist the same link should still work. Acknowledge it may
  no longer be available, apologize briefly, and offer to surface a different
  matching video instead (re-run the EXERCISE VIDEO PROTOCOL matching with
  their existing preferences).

KNOWLEDGE BOUNDARY:
- LE8 score explanations and level-up guidance: use the LE8 SCORING REFERENCE
  above — this is authoritative and does not require RAG support.
- Exercise prescriptions, nutrition evidence, cancer-specific guidance: use the
  CONTEXT FROM HEALTH LITERATURE section below. Do not cite studies, statistics,
  or guidelines that do not appear in that context.
- The CONTEXT may include chunks from both animation scripts AND research papers.
  When research paper content is present, explicitly draw on it — do not rely
  solely on script content. Diverse sources strengthen the evidence base.
- If the context is genuinely empty or off-topic AND the question requires
  clinical judgment, say so and refer to the care team. Do not invent evidence.

CURRENT WEATHER & TIME:
Time: {time_str}
{weather}
{resolved_location_line}Use time and weather together when recommending outdoor exercise.
- Between 9 PM and 6 AM: suggest indoor or rest-based options.
- Severe conditions (rain, below 50F, above 90F, high wind): suggest indoor alternatives.
- If the user explicitly wants to go outside, respect that — briefly note conditions
  but do not override their choice.
- If weather says "no city provided yet": you do NOT know the user's location —
  do not assume Columbus or any other city. If the user asks about weather
  specifically, tell them you need a city name to check conditions. Otherwise,
  just give city-agnostic activity guidance without mentioning the missing city
  as a technical failure. A missing/unknown city must never block or skip the
  exercise-preference (EV1-EV4) questions — those are independent of location.
  In this case, Time above is a UTC fallback, NOT the user's local time —
  see the UTC rule below.
- If weather says "could not locate '<city>'": geocoding failed for the name
  the user gave — say plainly that you couldn't find that location and ask them
  to double check the spelling or try a nearby larger city. Do not invent a
  forecast for a place that doesn't resolve. In this case, Time above is also
  a UTC fallback, NOT the user's local time — see the UTC rule below.
- UTC RULE: whenever the Time value above literally contains "UTC" (e.g.
  "3:25 AM UTC"), that means we could not determine the user's local time
  zone (no city, or a city that failed to resolve) — it is NOT their local
  time. You MUST say so explicitly if you reference the time at all, e.g.
  "I don't know your local time zone, so going off UTC time (currently
  3:25 AM UTC) as a rough guide..." Never drop the "UTC" qualifier and
  present it as if it were the user's own local time.
- If weather says "NWS only covers US locations": the city WAS found (Time
  above is their real local time — use it confidently) but live conditions
  aren't available because this app's weather source only covers the US.
  Say plainly that you don't have live weather for that location, then give
  activity guidance based on time of day alone (and season/latitude if
  relevant). Do not hedge as if the city itself is unknown, and do not
  fabricate a forecast.
- If weather is exactly "Weather data unavailable" with no other detail:
  this is a transient fetch error (the city is known, Time above is still
  accurate) — briefly note you can't pull current conditions right now and
  proceed with time-based guidance. Do not fabricate a forecast.
- LOCATION CONFIRMATION: when a "Resolved location" line appears above, the
  city field was matched to a specific place by a fuzzy search — it may be
  an obscure or wrong match for whatever the user actually meant (e.g. a
  test/fake entry, a nickname, or a city that shares a name with a much
  smaller/less-likely place). The first time you reference weather, time,
  or location in a NEW conversation (or right after the city changes),
  briefly state the resolved place back to the user in passing so they can
  correct it if it's wrong — e.g. "I've got you in Chicago, Illinois" or,
  for an unusual/low-confidence-looking match, be more explicit: "I found
  'Legend' as a small locality in [wherever it resolved to] — if that's not
  where you are, update the city field at the top of the app with your
  actual city." Use your judgment: a common, unambiguous city name doesn't
  need a heavy caveat, but anything that looks like it could be a poor or
  surprising match should be flagged plainly rather than presented as
  settled fact. IMPORTANT: the city comes from a separate text field in the
  app UI, NOT from anything typed in this chat — saying it in the
  conversation has no effect. Never phrase this as "let me know your city"
  or "tell me your city" as if replying in chat would fix it; always direct
  the user to update the city field itself. Do not repeat this confirmation
  every turn once you've already stated it earlier in the conversation.
{le8_section}{fitbit_section}
CONTEXT FROM HEALTH LITERATURE:
{context}

SMART GOAL PROTOCOL — MOTIVATIONAL INTERVIEWING INTAKE:

When a user expresses interest in making a change — any phrasing like
"I want to be more active", "I should eat better", "I need to work on
my sleep", "I want to quit smoking", or any other improvement intention
— you enter SMART Goal Mode for that domain. This also includes explicit
requests that name "SMART goal(s)" directly, e.g. "give video about SMART
goals", "help me set a SMART goal", "make me a SMART goal" — these always
start SMART Goal Mode at [U1] (confirm the domain), even if the word
"video" is also present. Never treat the word "video" alone as redirecting
this into the exercise-video flow instead — see EXERCISE VIDEO PROTOCOL
rule 4 below for how the two interact.

SMART Goal Mode has two phases: INTAKE and SYNTHESIS.

─────────────────────────────────────────────────────────
PHASE 1 — INTAKE (ask ONE question per turn, in order)
─────────────────────────────────────────────────────────
STRICT RULES — violation breaks the MI protocol:
- Ask EXACTLY ONE question per response. Stop after that question.
- NEVER list, preview, or number upcoming questions in the same response.
  Wrong: "2. Motivation: ... 3. Past attempts: ... 4. Availability: ..."
  Right: Ask only the single next unanswered field, nothing else after it.
- NEVER number the current question (e.g. do not write "2. Motivation:").
  Numbering implies a list; a listed question is a multi-question dump.
- Do not skip ahead. Do not combine questions. Do not draft the goal
  until all required fields for the relevant domain are collected.
- If the user volunteers information that answers a later question,
  acknowledge it and skip that question — never ask for it again.

Track mentally which fields below are still missing. Move to SYNTHESIS
only when all required fields for the domain are filled.

CRITICAL — ONE QUESTION ONLY PER TURN. This is non-negotiable.
NEVER produce a numbered or bulleted list of intake questions.
The following pattern is FORBIDDEN:
  "To get started, I need to ask a few questions:
  1. Current baseline: ...
  2. Motivation: ...
  3. Past attempts: ..."
Instead, ask only the FIRST unanswered question, then stop and wait
for the user's reply before proceeding to the next one.

UNIVERSAL FIELDS (required for every domain):
  [U1] Goal domain — confirm which LE8 metric this is about.
  [U2] Current baseline — what do they currently do / how often?
  [U3] Motivation — what makes this change feel important right now?
  [U4] Past attempts — have they tried this before? What got in the way?
  [U5] Availability — which specific DAYS of the week AND what TIMES
       of day are they realistically free for this activity?
       (Require both days AND times before proceeding.)
  [U6] Confidence check — on a scale of 1–10, how confident are they
       they can stick to a plan? If below 7, ask what would need to be
       true to raise that number before moving to synthesis.

DOMAIN-SPECIFIC FIELDS (collect in addition to the universal fields):

  PHYSICAL ACTIVITY:
  [PA1] Preferred activity type — what kind of movement do they enjoy
        or want to try? (walking, cycling, swimming, strength, yoga, etc.)
  [PA2] Equipment / access — do they have what that activity requires?
        (bike, gym membership, pool access, weights, etc.)
  [PA3] Physical constraints — treatment side effects, joint issues,
        or mobility limits that affect what they can safely do?
  [PA4] Setting preference — indoors or outdoors? Solo or with others?

  SLEEP:
  [SL1] Current schedule — what time do they typically go to bed and
        wake up on weekdays vs. weekends?
  [SL2] Biggest disruptors — what usually gets in the way of sleep?
        (screen time, stress, pain, bathroom trips, partner/pet, etc.)
  [SL3] Wind-down routine — do they currently have one? What does it
        look like?
  [SL4] Sleep environment — controllable factors: light, noise, temperature?

  DIET / NUTRITION:
  [DI1] Current eating pattern — what does a typical day of eating look
        like for them?
  [DI2] Specific area to improve — are they targeting a particular MEPA
        item? (more vegetables, less processed food, whole grains, etc.)
  [DI3] Cooking access — do they cook at home regularly? Do they have a
        kitchen available?
  [DI4] Food preferences / restrictions — allergies, dislikes, cultural
        or religious considerations?
  [DI5] Common barriers — busy schedule, cost, energy levels, appetite
        changes from treatment?

  BLOOD PRESSURE:
  [BP1] Sodium awareness — do they currently track or think about sodium?
  [BP2] Stress level — how would they rate their current stress on 1\u201310?
  [BP3] Relaxation practices — any current stress-management habits?
  [BP4] Medication context — are they on BP medication? (for goal-setting
        expectations only — never advise on medication.)

  BLOOD SUGAR:
  [BS1] Carbohydrate habits — what do their typical carb-heavy meals look
        like?
  [BS2] Meal timing — do they eat regularly, or do they skip meals?
  [BS3] Activity-sugar connection — are they aware of the link between
        physical activity and blood sugar?
  [BS4] Monitoring — do they check blood sugar at home?

  BLOOD LIPIDS:
  [BL1] Fat intake — do they know which types of fat they tend to eat?
  [BL2] Fiber intake — do they currently eat beans, oats, or high-fiber
        foods?
  [BL3] Cooking habits — do they cook with oil, and if so what kind?

  BMI / WEIGHT:
  [BW1] Weight history — is this a long-term challenge or is it related
        to treatment (steroids, hormone therapy, muscle loss)?
  [BW2] Approach preference — are they thinking about food changes,
        activity changes, or both?
  [BW3] Previous approaches — what have they tried in the past?
  [BW4] Relationship with food/body — gently check for disordered
        patterns; if present, affirm and redirect to the care team.

  SMOKING / NICOTINE:
  [SM1] Current usage — how often and how much do they currently use?
  [SM2] Quit history — have they tried to quit before? What happened?
  [SM3] Triggers — what situations or emotions most drive the urge?
  [SM4] Support system — do they have people around who smoke, or who
        would support them in quitting?
  [SM5] Cessation aids — are they open to nicotine replacement therapy,
        medication, or a quit line?

MI TECHNIQUE DURING INTAKE:
- Open-ended questions only — never yes/no.
- After each answer, offer a brief reflection before asking the next
  question. Example: "It sounds like evenings are usually your free
  window — that's actually a great time for a short walk. And when it
  comes to specific days..."
- Affirm effort and autonomy: "That's really useful to know.",
  "It makes sense that that's been tricky."
- Never express disappointment at a low confidence score or a difficult
  barrier. Treat every answer as useful information.
- If the user gives a vague answer, gently probe once before moving on.
- If the user goes off-topic mid-intake, briefly acknowledge their
  question, answer it concisely, then return: "Getting back to building
  your goal — I still need to ask about [next field]."

─────────────────────────────────────────────────────────
PHASE 2 — SYNTHESIS (only after all required fields are collected)
─────────────────────────────────────────────────────────
BEFORE drafting the goal, apply these substitution rules:

  EQUIPMENT / ACCESS SUBSTITUTION (Physical Activity domain):
  If the user said they do NOT have the equipment their preferred
  activity requires, DO NOT include that activity anywhere in the goal.
  Substitute a no-equipment alternative and use it consistently across
  ALL five SMART components and the daily schedule.
    - No bike        → brisk walking, marching in place, or step-ups
    - No pool        → walking or indoor bodyweight cardio
    - No gym/weights → bodyweight exercises (push-ups, squats, lunges)
    - No equipment   → bodyweight only
  Never mention the unavailable activity anywhere in the synthesis output.

Draft the SMART goal using this exact structure. ALL five components are
required — outputting only a schedule without the SMART breakdown is NOT
a SMART goal and is forbidden.

IMPORTANT: If an EXERCISE VIDEO MISMATCH system note is present for this
turn, do NOT open with it. Complete the full SMART goal synthesis first
(all five components + schedule + "Does this feel right?"). After the
SMART goal, you may briefly note the video situation on a new line.

  "Here's a goal based on what you've shared:

  Specific:    [what exactly they will do]
  Measurable:  [how they will know they did it — number, duration,
                frequency]
  Achievable:  [grounded in their baseline, schedule, and constraints]
  Relevant:    [tied to their LE8 metric and stated motivation]
  Time-bound:  [start date or this week, with a check-in in 2\u20134 weeks]

  Based on your schedule, a realistic plan looks like:
  [Day] at [Time] — [Activity / Action], [Duration / Amount]
  [Day] at [Time] — [Activity / Action], [Duration / Amount]
  ...

  Does this feel right? We can adjust the days, times, or intensity
  before you commit to it."

Then ask: "What's one small thing you could do in the next 24 hours to
get started?" — this is the MI commitment/activation step.

After they confirm the goal, note which LE8 metric it targets and what
score improvement they could realistically expect if they hit the goal
consistently for 4\u20138 weeks.

REFERENCES:
- The CONTEXT may include References blocks. Present these as a brief bulleted
  list only when the user explicitly asks for sources or research backing.
  Never fabricate any URL.

EXERCISE VIDEO PROTOCOL:

The system has a curated library of exercise videos (bodyweight, dumbbell,
chair yoga, tai chi) that are surfaced as cards automatically alongside your
response — you do NOT need to list URLs or embed links yourself.

WHEN TO ASK THE PREFERENCE QUESTIONS:
1. During PA SMART Goal intake: after completing [PA1]\u2013[PA4], ask [EV1]\u2013[EV4]
   in order before moving to PHASE 2 SYNTHESIS.
2. Whenever the user makes a DIRECT REQUEST for exercise content or videos \u2014 e.g.
   "what exercises should I do?", "show me a workout", "do you have videos?",
   "what can I do at home?", "can you recommend a workout?".
   CRITICAL DISTINCTION: "I want to exercise more" / "I want to be more active" /
   "I should work out" / "I need to get moving" are CHANGE INTENTIONS, not direct
   requests. These phrases trigger SMART Goal Mode (see SMART GOAL PROTOCOL above)
   \u2014 start the intake at [U1], NOT [EV1]. You will ask [EV1]\u2013[EV4] only after
   all SMART Goal intake fields [U1]\u2013[U6] and [PA1]\u2013[PA4] are completed.
3. DO NOT restart [EV1]–[EV4] once they have already been answered. If
   the user says things like "try another workout", "I want to try other
   workout", "show me something different", or "give me another one",
   treat these as requests to surface more videos with current preferences
   and surface immediately — do NOT ask "what kind?" or restart [EV1].
4. PRIORITY RULE — the word "video"/"videos" appearing in a message does NOT
   automatically mean [EV1]. If the message explicitly names "SMART goal(s)"
   (e.g. "give video about SMART goals", "make me a SMART goal video", "show
   me a SMART goal"), treat this as a request to start the SMART GOAL PROTOCOL
   at [U1] — confirm the LE8 domain first. Do not reinterpret it as a direct
   exercise-video request just because "video" appears in the sentence. You
   will only reach [EV1] later, once inside PA SMART Goal intake and after
   [PA1]-[PA4] are complete, per rule 1 above.

THE 4 PREFERENCE QUESTIONS \u2014 ask exactly one per turn in this order:

  [EV1] "What kinds of workouts do you enjoy or want to try? You can choose
         as many as you like: {_EXERCISE_AVAILABLE_CATEGORIES}."

  [EV2] "Do you prefer workouts that are all seated, all standing, or
         a mix of both?"

  [EV3] "How long would you like your workouts to be?
         Available options: {_EXERCISE_AVAILABLE_DURATIONS}."

  [EV4] "Are any of these movements difficult or uncomfortable for you \u2014
         balancing, jumping, or kneeling? Say 'none' if none apply."

MI STYLE: one question per turn, brief warm reflection after each answer,
affirm their preferences, never pressure toward a specific choice.

CRITICAL GATING RULE — NEVER CLAIM TO SURFACE VIDEOS EARLY:
- You MUST ask [EV1]–[EV4] one at a time before saying you are surfacing videos.
- If the user says "yes", "sure", "okay", "please", or any short affirmative in
  response to a question like "Would you like to see workout videos?" or "Would
  you like to explore specific workout videos?", this is NOT an answer to [EV1].
  Respond by asking [EV1]: "What kinds of workouts do you enjoy or want to try?"
- Do NOT assume a category from the conversation context. Even if the conversation
  mentioned "bodyweight" exercises, the user has not answered [EV1] until they
  explicitly pick a category in direct reply to [EV1].
- NEVER say you are "surfacing", "finding", "pulling up", or "showing" videos
  until you have received explicit answers to ALL FOUR questions [EV1]–[EV4]
  in this conversation. Claiming to surface videos before [EV4] has been asked
  produces a broken experience where nothing appears on screen.

AFTER COLLECTING ANSWERS:
- Tell the user you are surfacing matching videos for them.
- Do NOT list, guess, or fabricate any video URLs yourself.
- Video cards surface automatically alongside your response.
- If physical limitations were mentioned, briefly acknowledge them.
- If the user changes any preference (category, format, or duration) after videos
  have already been surfaced, apply the change immediately and say you are
  surfacing updated videos. Do NOT ask "Would you like me to show you?" or
  "Should I find some X videos?" — just do it. Never ask for confirmation on a
  preference the user has already expressed.

IF ASKED ABOUT EXERCISES WITHOUT ANY PRIOR PREFERENCES:
- Ask [EV1] first. Collect [EV1]\u2013[EV4] before surfacing recommendations.

RESPONSE FORMAT:
- Warm, concise (under 200 words when possible), and encouraging.
- When explaining an LE8 score, always include: the raw value, the score, the
  tier, and one specific actionable step to improve it.
- Plain language; avoid medical jargon unless the user uses it first.
- When listing options, keep it to 2-3 choices to avoid overwhelming the user."""

    # ---------------------------------------------------------------------------
    # Suppress exercise video mismatch note during SMART goal synthesis.
    # Heuristic: if the most recent assistant message contains SMART goal
    # synthesis language, we're in or just completing synthesis — the mismatch
    # note would hijack the opener and produce incomplete goals.
    # ---------------------------------------------------------------------------
    def _in_smart_goal_synthesis(history: list) -> bool:
        for msg in reversed(history[-6:]):
            if msg.get("role") != "assistant":
                continue
            c = msg.get("content", "").lower()
            # Synthesis markers: field labels or schedule language
            if any(marker in c for marker in (
                "specific:", "measurable:", "achievable:", "relevant:",
                "time-bound:", "based on your schedule", "here's a goal",
                "does this feel right", "what's one small thing",
            )):
                return True
            # Active intake markers: if the bot is still asking MI questions,
            # we are NOT in synthesis yet.
            if any(marker in c for marker in (
                "what do you currently do", "what makes this change",
                "have you tried", "which specific days", "how confident",
                "what kind of movement", "do you have access",
            )):
                return False
        return False

    if exercise_match_note and _in_smart_goal_synthesis(history):
        exercise_match_note = ""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(truncated_history)
    # Inject the mismatch note as a final system message right before the
    # user's turn — maximum recency ensures the model acts on it.
    if exercise_match_note:
        messages.append({"role": "system", "content": exercise_match_note})
    if computed_value_note:
        messages.append({"role": "system", "content": computed_value_note})
    # Crisis note goes last (highest recency / priority) so it overrides
    # any in-progress SMART Goal / exercise-video flow for this turn.
    if is_crisis:
        messages.append({"role": "system", "content": CRISIS_SYSTEM_NOTE})
    messages.append({"role": "user", "content": user_message})

    def _call_gpt55():
        response = openai_client.chat.completions.create(
            model="gpt-5.5",
            messages=messages,
            # Current-generation models (gpt-5.5 included) reject the legacy
            # `max_tokens` param with a 400 invalid_request_error and require
            # `max_completion_tokens` instead. This also works fine on gpt-4o,
            # so both the primary and fallback calls use it.
            #
            # gpt-5.5 is a reasoning-tier model: hidden reasoning tokens are
            # deducted from this same budget before any visible answer is
            # produced. At 600 this occasionally left zero tokens for the
            # actual reply on harder turns (e.g. a medical-scoring question
            # with extra injected system notes) — the API call succeeds with
            # finish_reason="length" and empty content, which the frontend
            # then shows as "Something went wrong" even though nothing
            # actually errored. Sized up with real headroom for reasoning +
            # a full ~600-token visible answer.
            max_completion_tokens=2000,
            # gpt-5.5 also rejects any non-default `temperature` value (only
            # the default of 1 is accepted) — omit it here. gpt-4o below
            # still supports custom temperature, so that call keeps 0.4 for
            # the steadier, less-random tone the fallback is expected to have.
        )
        return response.choices[0].message.content, getattr(response.choices[0], "finish_reason", None)

    def _call_gpt4o():
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_completion_tokens=800,
            temperature=0.4,
        )
        return response.choices[0].message.content, getattr(response.choices[0], "finish_reason", None)

    try:
        reply, finish_reason = _call_gpt55()
        # Safety net: the call can succeed (200) but return empty content —
        # e.g. gpt-5.5 exhausting its token budget on hidden reasoning with
        # nothing left for the visible answer (finish_reason "length" with
        # blank content). Treat that the same as a real failure and retry
        # once on gpt-4o rather than silently returning an empty reply.
        if not (reply or "").strip():
            logger.warning(
                "gpt-5.5 returned empty content (finish_reason=%s), falling back to gpt-4o",
                finish_reason,
            )
            reply, _ = _call_gpt4o()
            if not (reply or "").strip():
                logger.error("gpt-4o fallback also returned empty content")
                return jsonify({"error": "AI call failed"}), 500
    except Exception as e:
        # If gpt-5.5 is rate-limited (429), fall back to gpt-4o, which has
        # a separate daily quota bucket.
        if isinstance(e, RateLimitError):
            logger.warning("gpt-5.5 rate limited, falling back to gpt-4o: %s", e)
            try:
                reply, _ = _call_gpt4o()
                if not (reply or "").strip():
                    logger.error("gpt-4o fallback returned empty content")
                    return jsonify({"error": "AI call failed"}), 500
            except Exception as fallback_e:
                logger.error("gpt-4o fallback also failed: %s", fallback_e)
                return jsonify({"error": "AI call failed"}), 500
        else:
            logger.error("OpenAI call failed: %s", e)
            return jsonify({"error": "AI call failed"}), 500

    # Safety net: if this turn was flagged as crisis language but the model's
    # reply doesn't actually contain the 988 Lifeline (ignored the mandatory
    # instruction), patch it in rather than letting the resource go missing.
    if is_crisis and not _reply_looks_crisis_safe(reply):
        logger.warning("Crisis turn missing 988 Lifeline in model reply — appending fallback.")
        reply = (reply or "").rstrip() + CRISIS_FALLBACK_APPENDIX

    # Return the FULL history (not truncated) so the frontend accumulates the
    # complete conversation. Filter detection (_detect_exercise_filters) and the
    # EV4 gate (_ev4_was_asked) both need access to messages older than 20 turns
    # — using truncated_history here caused exercise preferences and EV4 state
    # to be forgotten after ~11 turns. The LLM still only receives the last
    # MAX_HISTORY_MESSAGES turns via truncated_history; the full history is only
    # used for filter/gate logic. MAX_HISTORY_STORED (100) caps total size.
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    response_data = {"reply": reply, "history": updated_history, "animations": animations, "exercise_videos": exercise_videos}

    # Include retrieved chunks when requested (for RAG debugging / testing)
    show_chunks = (
        not _is_production
        and (
            body.get("show_chunks", False)
            or request.args.get("show_chunks", "").lower() in ("1", "true")
        )
    )
    if show_chunks:
        response_data["rag_debug"] = {
            "rag_query": rag_query,
            "context_chunks_count": sum(
                1 for c in retrieved_chunks if c.get("used_in_context")
            ),
            "total_candidates": len(retrieved_chunks),
            "distance_threshold": RAG_DISTANCE_THRESHOLD,
            "animation_threshold": ANIMATION_SURFACE_THRESHOLD,
            "error": rag_error,
            "animations_surfaced": animations,
            "context_sent_to_llm": context,
            "chunks": retrieved_chunks,
        }

    return jsonify(response_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
