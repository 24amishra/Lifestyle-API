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
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
from urllib.parse import urlencode
import chromadb

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
MAX_MESSAGE_LENGTH = 2000
CITY_PATTERN = re.compile(r"^[a-zA-Z\s\-'.]+$")

# Cosine distance threshold for RAG relevance (ChromaDB uses 1-cosine_similarity,
# so 0 = identical, 1 = orthogonal, 2 = opposite).
# Chunks with distance > this value are considered off-topic and dropped.
RAG_DISTANCE_THRESHOLD = 0.75

# More lenient threshold used exclusively for surfacing animation cards.
# A chunk doesn't need to be relevant enough to inform the LLM's answer to
# still warrant showing the user a related video.
ANIMATION_SURFACE_THRESHOLD = 0.82

# Maximum number of animation cards to surface per response.
# Prevents overwhelming the user when a broad question matches many videos.
MAX_ANIMATIONS_PER_RESPONSE = 2

rate_limit_store: dict = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20

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
# Helpers
# ---------------------------------------------------------------------------

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
    for msg in history:
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
    with the tail of recent conversation.

    Short or decontextualized messages like "tell me more about that" produce
    poor embeddings on their own. Prepending the last assistant turn gives the
    retrieval model enough topic signal to find the right chunks.
    """
    if len(user_message.split()) >= 8:
        # Message is long enough to be self-contained
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
        # Animation surfacing — intentionally decoupled from use_chunk.
        # Uses ANIMATION_SURFACE_THRESHOLD (more lenient than the context
        # threshold) so a video can surface even when its chunk was too
        # marginal to include in the LLM context.
        # ----------------------------------------------------------------
        anim_url = meta.get("animation_url", "")
        section_title = meta.get("section_title", "")
        if (
            anim_url
            and anim_url not in seen_anim_urls
            and distance <= ANIMATION_SURFACE_THRESHOLD
            and len(animations) < MAX_ANIMATIONS_PER_RESPONSE
        ):
            animations.append({"title": section_title, "url": anim_url})
            seen_anim_urls.add(anim_url)

        if not use_chunk:
            continue

        used_count += 1
        block = chunk

        if include_references and ref_urls_list:
            refs_str = "\n".join(f"- {u}" for u in ref_urls_list[:5])
            block += f"\n\n\U0001f4da References for this section:\n{refs_str}"

        context_parts.append(block)

    if not context_parts:
        context_str = (
            "No sufficiently relevant information was found in the knowledge base "
            "for this query."
        )
    else:
        context_str = "\n\n---\n\n".join(context_parts)

    return {"context": context_str, "chunks": chunk_details, "animations": animations}


def _geocode_city(city: str) -> tuple[float, float, str] | None:
    """
    Resolve a city name to (latitude, longitude, iana_timezone).
    The IANA timezone string (e.g. 'America/Chicago') comes free from
    Open-Meteo's geocoding response and is used to compute local time.
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
            return r["latitude"], r["longitude"], tz_str
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
    (lat, lon, tz_str) from _geocode_city to avoid a redundant geocoding
    call when the caller already has it.
    """
    if city_info is not None:
        lat, lon, _ = city_info
    else:
        city_info = _geocode_city(city)
        if city_info is None:
            return f"Weather data unavailable (could not locate '{city}')"
        lat, lon, _ = city_info
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
        return (
            f"{city}: {current['temperature']}°{current['temperatureUnit']}, "
            f"{current['shortForecast']}, "
            f"wind {current['windDirection']} {current['windSpeed']}"
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
    try:
        if user_id:
            document = collection.find_one({"_id": ObjectId(user_id)})
        else:
            document = collection.find_one(sort=[("updated_at", -1)])

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

    res = requests.post("https://api.fitbit.com/oauth2/token", data=data, headers=headers)
    if res.status_code == 200:
        token_data = res.json()
        if user_doc_id:
            save_tokens(token_data["access_token"], token_data["refresh_token"], user_doc_id)
        return token_data

    logger.error("Token refresh failed: %s %s", res.status_code, res.text)
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

    params = {
        "response_type": "code",
        "client_id": FITBIT_CLIENT_ID,
        "redirect_uri": FITBIT_REDIRECT_URI,
        "scope": "activity heartrate sleep profile",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return redirect(f"https://www.fitbit.com/oauth2/authorize?{urlencode(params)}")


@app.route("/callback")
def callback():
    if not _fitbit_configured():
        return jsonify({"error": "Fitbit integration is not configured"}), 503

    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

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
        return redirect(f"{frontend_url}?fitbit=connected&uid={user_doc_id}")

    logger.error("Fitbit callback error: %s %s", res.status_code, res.text)
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

    raw_city = body.get("city", "Columbus")
    if not isinstance(raw_city, str):
        raw_city = "Columbus"
    city = sanitize_city(raw_city) or "Columbus"

    # Build a richer query for short / decontextualized messages
    rag_query = _build_rag_query(user_message, history)

    # Detect whether the user is explicitly asking for research sources
    include_references = bool(REFERENCE_INTENT_PATTERN.search(user_message))

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

    # Geocode once — result feeds both weather and local time so we never
    # hit the geocoding API twice for the same request.
    city_info = _geocode_city(city)
    weather = get_weather(city, city_info=city_info)
    if city_info:
        time_str = get_local_time(city_info[2])
    else:
        # Geocoding failed; fall back to UTC
        time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%I:%M %p UTC").lstrip("0")

    fitbit_section = ""
    fitbit_data = None

    if USE_MOCK_FITBIT:
        fitbit_data = _load_mock_fitbit_data()
    else:
        user_doc_id = session.get("user_doc_id") or body.get("user_doc_id")
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
USER'S FITBIT DATA (today):
{json.dumps(fitbit_data, indent=2)}
Use this data to personalize recommendations. Reference their actual steps,
sleep, and heart-rate when relevant."""

    system_prompt = f"""You are a supportive, evidence-based health coach chatbot designed primarily
for men who are currently undergoing prostate cancer treatment or who have survived
prostate cancer. You also serve cancer survivors more broadly. You cover two core
pillars: physical activity and nutrition.

YOUR ROLE:
- Exercise and physical activity recommendations appropriate for cancer survivors,
  mindful of treatment side effects common in prostate cancer.
- Nutrition and healthy eating guidance: food choices, meal planning, reading
  labels, building a balanced plate, grocery shopping, and debunking common
  food myths — all grounded in the provided literature.
- Setting SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound).
- Motivational Interviewing: ask open-ended questions, use reflective listening,
  affirm effort and autonomy, never lecture or push.
- Supporting emotional well-being alongside physical health.

BEHAVIOR GUIDELINES:
- Always be sensitive to the physical and emotional realities of living with or
  recovering from cancer.
- "Medical concerns" that should be redirected to the care team means: specific
  symptoms, treatment decisions, medication interactions, supplement dosages,
  or anything requiring clinical judgement. It does NOT mean general nutrition
  advice, healthy eating patterns, food choices, or exercise guidance — those
  are squarely within your role and should be answered from the context.
- Never diagnose, prescribe, or contradict medical advice.
- Do not provide specific dosages for supplements or medications.
- If a user appears to be in crisis or mentions self-harm, respond with empathy
  and direct them to appropriate emergency resources (988 Suicide & Crisis Lifeline,
  or their care team).

CRITICAL KNOWLEDGE BOUNDARY — READ CAREFULLY:
You must answer exclusively from the CONTEXT FROM HEALTH LITERATURE section below.
This context comes from two source types:
  1. Peer-reviewed research papers and clinical guidelines (your factual backbone).
  2. Educational animation transcripts written for patients (plain-language summaries
     of the same evidence).

You must NOT:
- Draw on your training knowledge, even for facts you are confident about.
- Cite any study, statistic, or guideline that does not appear in the context.
- Speculate or extrapolate beyond what the provided documents explicitly state.
- Fabricate or guess any URL.

When to use the context vs. when to defer:
- If relevant context is present — even partially — USE it to answer. Do not
  say you lack information just because the context does not answer every detail.
- Only respond with "I don’t have information on that in my knowledge base.
  Please check with your healthcare provider." when the context is genuinely
  empty or contains only unrelated content AND the question requires clinical
  judgement (symptoms, treatment, medication).
- For questions about nutrition, food, exercise, sleep, or stress where the
  context has relevant content, always answer from that content.

CURRENT WEATHER & TIME:
Time: {time_str}
{weather}
Use the time and weather together when recommending outdoor exercise.
- If it is between 9 PM and 6 AM, do not suggest outdoor activity — acknowledge
  the time and suggest indoor or rest-based options instead.
- If conditions are severe (rain, below 50°F, above 90°F, or high wind), default
  to suggesting indoor alternatives and explain why.
- However, if the user explicitly says they want to exercise outdoors right now,
  respect that preference and give outdoor suggestions regardless of time or
  temperature — you can briefly note the conditions but do not override their choice.
{fitbit_section}

CONTEXT FROM HEALTH LITERATURE:
{context}

SMART GOAL PROTOCOL:
- If a goal is vague, ask one clarifying question at a time.
- Once you have enough information, reflect the goal back in full SMART format.
- Suggest a realistic timeline and check-in cadence.

REFERENCES:
- The CONTEXT may include 📚 References blocks. Present these as a brief
  bulleted list only when the user explicitly asked for sources or research
  backing. Never include reference URLs otherwise. Never fabricate any URL.

RESPONSE FORMAT:
- Keep responses warm, concise (under 200 words when possible), and encouraging.
- Use plain language; avoid medical jargon unless the user uses it first.
- When listing options, keep it to 2-3 choices to avoid overwhelming the user."""

    messages = [{"role": "system", "content": system_prompt}]
    truncated_history = history[-MAX_HISTORY_MESSAGES:]
    messages.extend(truncated_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return jsonify({"error": "AI call failed"}), 500

    updated_history = truncated_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    response_data = {"reply": reply, "history": updated_history, "animations": animations}

    # Include retrieved chunks when requested (for RAG debugging / testing)
    show_chunks = (
        body.get("show_chunks", False)
        or request.args.get("show_chunks", "").lower() in ("1", "true")
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
