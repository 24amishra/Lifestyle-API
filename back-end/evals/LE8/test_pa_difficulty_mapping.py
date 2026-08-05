"""
Deterministic unit tests for LE8-derived exercise difficulty inference.

DELIBERATELY SEPARATE FROM test_le8_scoring.py. This is downstream
exercise-matching logic that CONSUMES an LE8 Physical Activity score — it is
not an LE8 metric and does not compute one. Its cutoffs (70/40) are its own,
and are intentionally different from the LE8 tier cutoffs (80/50) tested in
the other file. Keeping them apart stops anyone from "harmonizing" two
threshold sets that are supposed to differ.

Functions under test:
    _pa_score                  app.py:671  (extract PA score from payload)
    _infer_difficulty_from_le8 app.py:707  (score -> Beginner/Intermediate/Advanced)
    _resolve_exercise_difficulty app.py:726 (stated level beats inferred)

Plain pytest — no LLM judge, no API calls. See README.md.
"""

import pytest

from le8_loader import load_functions

_fns = load_functions(
    "_pa_score",
    "_infer_difficulty_from_le8",
    "_resolve_exercise_difficulty",
    "_build_difficulty_note",
)
_pa_score = _fns["_pa_score"]
_infer_difficulty_from_le8 = _fns["_infer_difficulty_from_le8"]
_resolve_exercise_difficulty = _fns["_resolve_exercise_difficulty"]
_build_difficulty_note = _fns["_build_difficulty_note"]


def payload(score):
    """Minimal le8_data shaped like app.py:1956-1972."""
    return {"metrics": {"physical_activity": {"score": score}}}


# ===========================================================================
# Cutoffs — app.py:719-723
#   pa_score >= 70 -> "Advanced"
#   pa_score >= 40 -> "Intermediate"
#   else           -> "Beginner"
#   no score       -> "Beginner" (documented fallback, app.py:710)
# ===========================================================================

@pytest.mark.parametrize("score,expected", [
    (0,   "Beginner"),
    (20,  "Beginner"),       # squarely Beginner
    (55,  "Intermediate"),   # squarely Intermediate
    (85,  "Advanced"),       # squarely Advanced
    (100, "Advanced"),
])
def test_difficulty_in_tier(score, expected):
    assert _infer_difficulty_from_le8(payload(score)) == expected


@pytest.mark.parametrize("score,expected", [
    (39, "Beginner"),      # boundary: highest Beginner
    (40, "Intermediate"),  # boundary: lowest Intermediate
    (41, "Intermediate"),
    (69, "Intermediate"),  # boundary: highest Intermediate
    (70, "Advanced"),      # boundary: lowest Advanced
    (71, "Advanced"),
])
def test_difficulty_cutoff_boundaries(score, expected):
    """Off-by-one checks at 40 and 70. Both cutoffs are >=, so the cutoff
    value itself belongs to the HIGHER tier."""
    assert _infer_difficulty_from_le8(payload(score)) == expected


def test_difficulty_cutoffs_differ_from_le8_tier_cutoffs():
    """Pins the intentional divergence: LE8 tiers break at 80/50
    (app.py:3117), difficulty breaks at 70/40. A score of 75 is
    'Intermediate' as an LE8 tier but 'Advanced' as a difficulty level.
    If someone aligns these, this fails."""
    assert _infer_difficulty_from_le8(payload(75)) == "Advanced"
    assert _infer_difficulty_from_le8(payload(45)) == "Intermediate"


def test_float_scores_are_accepted():
    """_pa_score explicitly allows float (app.py:697)."""
    assert _infer_difficulty_from_le8(payload(69.9)) == "Intermediate"
    assert _infer_difficulty_from_le8(payload(70.0)) == "Advanced"


# ===========================================================================
# Type / shape guards — ALREADY CORRECT, asserted normally.
#
# _pa_score has an explicit numeric guard (app.py:679-704) added because
# le8_data is client-controlled and a non-numeric score would push a
# TypeError into the caller's `>=` comparisons — "a 500 on demand via
# {"score": "75"}". The guard works: every non-numeric input returns None,
# and the caller falls back to Beginner, the safest possible default.
# ===========================================================================

@pytest.mark.parametrize("bad_score", ["75", None, [], {}, object()])
def test_pa_score_rejects_non_numeric(bad_score):
    assert _pa_score(payload(bad_score)) is None


@pytest.mark.parametrize("bad_score", [True, False])
def test_pa_score_rejects_bool(bad_score):
    """bool is a subclass of int, so it must be excluded explicitly
    (app.py:683). True would otherwise score as 1."""
    assert _pa_score(payload(bad_score)) is None


@pytest.mark.parametrize("bad_score", ["75", None, True, [], {}])
def test_non_numeric_falls_back_to_beginner(bad_score):
    """The safe default: never infer a harder workout from garbage input."""
    assert _infer_difficulty_from_le8(payload(bad_score)) == "Beginner"


@pytest.mark.parametrize("bad_payload", [
    None,
    {},
    {"metrics": None},
    {"metrics": {}},
    {"metrics": {"physical_activity": None}},
    {"metrics": {"physical_activity": {}}},
])
def test_malformed_payload_falls_back_to_beginner(bad_payload):
    """Every degenerate payload shape survives and returns the safe default."""
    assert _pa_score(bad_payload) is None
    assert _infer_difficulty_from_le8(bad_payload) == "Beginner"


# ===========================================================================
# Stated level beats inferred — app.py:756-759
# ===========================================================================

def test_stated_difficulty_wins_over_inferred():
    level, source, pa = _resolve_exercise_difficulty(
        {"difficulty": "Beginner"}, payload(95)
    )
    assert (level, source, pa) == ("Beginner", "stated", None)


def test_inferred_difficulty_carries_provenance_and_raw_score():
    """The raw score travels with the level so the mismatch note can disclose
    WHERE an inferred level came from (app.py:675-677)."""
    level, source, pa = _resolve_exercise_difficulty({}, payload(85))
    assert (level, source, pa) == ("Advanced", "inferred", 85)


def test_inferred_is_behaviour_neutral_when_nothing_stated():
    """app.py:753-754 — with no stated level it must return exactly what
    _infer_difficulty_from_le8 returns."""
    for score in (0, 39, 40, 69, 70, 100):
        level, source, _ = _resolve_exercise_difficulty(None, payload(score))
        assert level == _infer_difficulty_from_le8(payload(score))
        assert source == "inferred"


# ===========================================================================
# Out-of-range input — REGRESSION TESTS. Bugs #3 and #5, FIXED 2026-08-04.
#
# WAS: _pa_score validated that the score IS a number but never that it is a
# PLAUSIBLE one. The LE8 PA score is defined as "(steps / goal) x 100, capped
# at 100" (app.py:3121), so anything above 100 or below 0 is out of range by
# definition — yet it passed the isinstance check untouched. Two consequences
# on the high side:
#
#   - a corrupt score of 101, 500 or 1e9 inferred "Advanced", the most
#     strenuous tier, for a cancer survivor whose actual activity level is
#     unknown. Fatigue, neuropathy and deconditioning are exactly the
#     population risks the prompt warns about (app.py:3179-3181); and
#   - the raw number leaked into the model-facing difficulty note verbatim —
#     "Their LE8 Physical Activity score is 1000000000.0/100"
#     (_build_difficulty_note, app.py:1534-1538) — which the model then
#     relayed to the user. See test_corrupt_score_takes_the_no_score_branch.
#
# NOW: _pa_score requires 0 <= score <= 100 and returns None otherwise.
#
# A SENTINEL, NOT AN EXCEPTION — the opposite of the scoring functions in
# test_le8_scoring.py, and for a concrete reason: this function already
# documents its contract as "the score, or None if unavailable" (app.py:673),
# and its caller already treats None as "fall back to Beginner". An
# out-of-range score is precisely an unavailable score, so it reuses the
# sentinel that already exists and already has a safe handler. Raising would
# have needed a new try/except at every call site to buy nothing.
#
# These are now plain passing assertions. Reverting the guard turns them red.
# ===========================================================================

@pytest.mark.parametrize("score", [101, 500, 1_000_000_000])
def test_pa_score_rejects_above_cap(score):
    """PA score is capped at 100 by definition (app.py:3121)."""
    assert _pa_score(payload(score)) is None


@pytest.mark.parametrize("score", [101, 500, 1_000_000_000])
def test_above_cap_does_not_infer_advanced(score):
    """The consequence that actually matters: an impossible score must not
    prescribe the hardest workout tier."""
    assert _infer_difficulty_from_le8(payload(score)) == "Beginner"


@pytest.mark.parametrize("score", [-1, -50])
def test_pa_score_rejects_negative(score):
    assert _pa_score(payload(score)) is None


def test_negative_score_still_yields_safe_difficulty():
    """The negative case reaches the safe default via the sentinel now, where
    before it reached it by accident (-50 < 40). Same visible outcome, correct
    internal contract — see test_pa_score_rejects_negative."""
    assert _infer_difficulty_from_le8(payload(-50)) == "Beginner"


# --- the leak this fix also closed ---------------------------------------
# The difficulty note interpolates difficulty_pa straight into text the model
# is told to follow, gated only on `is not None` (app.py:1534-1538). Before
# the range guard, a corrupt score reached that branch as a real number and
# the model was handed "Their LE8 Physical Activity score is
# 1000000000.0/100" to relay. This was NOT in the original bug #3 write-up;
# it was found by tracing difficulty_pa end-to-end after the fix.

_VIDEOS = [
    {"difficulty": "Beginner", "category": "Yoga"},
    {"difficulty": "Advanced", "category": "HIIT"},
]


@pytest.mark.parametrize("score", [101, 500, 1_000_000_000, -50])
def test_corrupt_score_takes_the_no_score_branch(score):
    """End-to-end: a corrupt score must reach _build_difficulty_note as None
    so the note says no score was available, instead of printing the corrupt
    number to the model."""
    level, source, difficulty_pa = _resolve_exercise_difficulty({}, payload(score))
    assert difficulty_pa is None
    assert level == "Beginner"

    note = _build_difficulty_note(
        {"categories": ["Yoga"]}, level, source, difficulty_pa, _VIDEOS, False
    )
    assert "No LE8 Physical Activity score was available" in note
    assert str(score) not in note


def test_valid_score_still_reaches_the_note():
    """The other half: a legitimate score must still be disclosed, so the
    guard cannot be 'fixed' by suppressing every score."""
    level, source, difficulty_pa = _resolve_exercise_difficulty({}, payload(85))
    assert (level, difficulty_pa) == ("Advanced", 85)

    note = _build_difficulty_note(
        {"categories": ["Yoga"]}, level, source, difficulty_pa, _VIDEOS, False
    )
    assert "Their LE8 Physical Activity score is 85/100" in note
