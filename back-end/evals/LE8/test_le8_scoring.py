"""
Deterministic unit tests for the LE8 (Life's Essential 8) scoring logic.

Plain pytest — no LLM judge, no API calls, no RAGAS/TruLens/DeepEval. LE8
scoring is fixed-threshold arithmetic with exactly one right answer per
input, so a semantic judge is the wrong instrument for it (see README.md).

Source of truth for every number below: the LE8 SCORING REFERENCE block in
app.py's system prompt, app.py:3109-3176. No threshold here was invented;
each test names the prompt line it came from.

SCOPE — only three of the eight LE8 metrics are scored in Python:

    _score_hba1c             app.py:1802
    _score_fasting_glucose   app.py:1814
    _score_non_hdl           app.py:1822
    _non_hdl_range_for_score app.py:1830  (reverse map, score -> mg/dL range)
    _le8_tier                app.py:1937  (score -> Ideal/Intermediate/Low)

The other five metrics (physical activity, sleep, blood pressure, BMI, diet,
smoking) arrive pre-computed from the frontend in the le8_data payload
(app.py:1956-1972); no Python computes them. Their thresholds exist only as
prose in the system prompt, so the last section here guards that prose text
against accidental edits rather than pretending to test a function.

Downstream difficulty inference (_pa_score and the 70/40 cutoffs) is NOT
here — that is exercise-matching logic that consumes an LE8 score, not an
LE8 metric. It lives in test_pa_difficulty_mapping.py.
"""

import inspect

import pytest

from le8_loader import load_functions, read_app_source

_fns = load_functions(
    "_score_hba1c",
    "_score_fasting_glucose",
    "_score_non_hdl",
    "_non_hdl_range_for_score",
    "_le8_tier",
)
_score_hba1c = _fns["_score_hba1c"]
_score_fasting_glucose = _fns["_score_fasting_glucose"]
_score_non_hdl = _fns["_score_non_hdl"]
_non_hdl_range_for_score = _fns["_non_hdl_range_for_score"]
_le8_tier = _fns["_le8_tier"]


# ===========================================================================
# Tier mapping — app.py:3117
#   "Score tiers: 0-49 = Low | 50-79 = Intermediate | 80-100 = Ideal"
# ===========================================================================

@pytest.mark.parametrize("score,expected", [
    (100, "Ideal"),
    (85,  "Ideal"),
    (80,  "Ideal"),          # boundary: lowest Ideal
    (79,  "Intermediate"),   # boundary: highest Intermediate
    (60,  "Intermediate"),
    (50,  "Intermediate"),   # boundary: lowest Intermediate
    (49,  "Low"),            # boundary: highest Low
    (20,  "Low"),
    (0,   "Low"),
])
def test_le8_tier_boundaries(score, expected):
    assert _le8_tier(score) == expected


def test_le8_tier_accepts_floats():
    """Docstring says int or float; the composite average produces floats."""
    assert _le8_tier(79.9) == "Intermediate"
    assert _le8_tier(80.0) == "Ideal"


# ===========================================================================
# Blood Lipids — Non-HDL cholesterol, mg/dL. app.py:3147
#   "<130 -> 100 | 130-159 -> 60 | 160-189 -> 40 | 190-219 -> 20 | >=220 -> 0"
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (100, 100),   # squarely Ideal
    (145, 60),    # squarely Intermediate
    (175, 40),    # squarely Low
    (200, 20),    # squarely Low
    (260, 0),     # squarely Low
])
def test_non_hdl_in_tier(value, expected):
    assert _score_non_hdl(value) == expected


@pytest.mark.parametrize("value,expected", [
    (129, 100), (129.9, 100), (130, 60),   # 130 boundary
    (159, 60),  (159.9, 60),  (160, 40),   # 160 boundary
    (189, 40),  (189.9, 40),  (190, 20),   # 190 boundary
    (219, 20),  (219.9, 20),  (220, 0),    # 220 boundary
])
def test_non_hdl_boundaries(value, expected):
    """Off-by-one checks: each threshold is exclusive-below, so the threshold
    value itself must land in the WORSE tier."""
    assert _score_non_hdl(value) == expected


def test_non_hdl_tier_jump_has_no_middle_ground():
    """app.py:3150 — going 60 -> 100 requires getting under 130; nothing
    scores between. Guards against someone adding an intermediate band."""
    assert _score_non_hdl(130) == 60
    assert _score_non_hdl(129.99) == 100


# --- reverse map: quoted score -> mg/dL range (app.py:1830) ---------------

@pytest.mark.parametrize("score,expected", [
    (100, "under 130 mg/dL"),
    (60,  "130-159 mg/dL"),
    (40,  "160-189 mg/dL"),
    (20,  "190-219 mg/dL"),
    (0,   "220 mg/dL or higher"),
])
def test_non_hdl_range_for_score_round_trips(score, expected):
    assert _non_hdl_range_for_score(score) == expected


@pytest.mark.parametrize("score", [75, 50, 99, 1, 101, -1, 1000])
def test_non_hdl_range_for_score_rejects_impossible_scores(score):
    """Returns None for any score _score_non_hdl cannot emit. The caller
    (app.py:1919) only builds the note `if range_str`, so None is the
    correct, already-handled 'say nothing' signal."""
    assert _non_hdl_range_for_score(score) is None


def test_non_hdl_reverse_map_covers_every_emittable_score():
    """Every score the forward function can produce must reverse-map. Catches
    a threshold being added to one function but not the other.

    Sweeps from 1, not 0: 0 is not a valid non-HDL value and now raises."""
    emittable = {_score_non_hdl(v) for v in range(1, 400)}
    for score in emittable:
        assert _non_hdl_range_for_score(score) is not None, (
            f"_score_non_hdl can emit {score} but the reverse map has no entry"
        )


# ===========================================================================
# Blood Sugar — fasting glucose, mg/dL. app.py:3138
#   "No diabetes, fasting glucose: <100 -> 100 | 100-125 -> 60 | >=126 -> 0"
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (85, 100),    # squarely Ideal
    (110, 60),    # squarely Intermediate
    (150, 0),     # squarely Low
])
def test_fasting_glucose_in_tier(value, expected):
    assert _score_fasting_glucose(value) == expected


@pytest.mark.parametrize("value,expected", [
    (99, 100), (99.9, 100), (100, 60),   # 100 boundary
    (125, 60), (125.9, 60), (126, 0),    # 126 boundary
])
def test_fasting_glucose_boundaries(value, expected):
    assert _score_fasting_glucose(value) == expected


def test_fasting_glucose_ignores_diabetes_status():
    """app.py:1815-1816 — LE8 defines a diabetic-specific scale for HbA1c
    only, not fasting glucose. This function takes no diabetes argument;
    asserting the signature so nobody 'helpfully' adds one."""
    params = list(inspect.signature(_score_fasting_glucose).parameters)
    assert params == ["value"]


# ===========================================================================
# Blood Sugar — HbA1c %, NON-diabetic scale. app.py:3139
#   "No diabetes, HbA1c: <5.7 -> 100 | 5.7-6.4 -> 60 | >=6.5 -> 0"
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (5.0, 100),   # squarely Ideal
    (6.0, 60),    # squarely Intermediate
    (7.5, 0),     # squarely Low
])
def test_hba1c_nondiabetic_in_tier(value, expected):
    assert _score_hba1c(value, False) == expected


@pytest.mark.parametrize("value,expected", [
    (5.6, 100), (5.69, 100), (5.7, 60),   # 5.7 boundary
    (6.4, 60),  (6.49, 60),  (6.5, 0),    # 6.5 boundary
])
def test_hba1c_nondiabetic_boundaries(value, expected):
    assert _score_hba1c(value, False) == expected


# ===========================================================================
# Blood Sugar — HbA1c %, DIABETIC scale (max 40). app.py:3140-3141
#   "With diabetes: <7 -> 40 | 7-7.9 -> 30 | 8-8.9 -> 20 | 9-9.9 -> 10 |
#    >=10 -> 0"
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (6.5, 40),
    (7.5, 30),
    (8.5, 20),
    (9.5, 10),
    (11.0, 0),
])
def test_hba1c_diabetic_in_tier(value, expected):
    assert _score_hba1c(value, True) == expected


@pytest.mark.parametrize("value,expected", [
    (6.9, 40), (7.0, 30),    # 7 boundary
    (7.9, 30), (8.0, 20),    # 8 boundary
    (8.9, 20), (9.0, 10),    # 9 boundary
    (9.9, 10), (10.0, 0),    # 10 boundary
])
def test_hba1c_diabetic_boundaries(value, expected):
    assert _score_hba1c(value, True) == expected


def test_hba1c_diabetic_scale_caps_at_40_which_is_always_low_tier():
    """Consequence worth pinning: the diabetic scale maxes at 40, and 40 is
    below the 50-point Intermediate cutoff (app.py:3117). So a diabetic user
    with excellent control still scores 'Low' on this metric. That is the
    documented LE8 design, not a bug — but it surprises people, and a future
    edit to either the cap or the tier cutoffs should fail here loudly."""
    best_possible = _score_hba1c(4.0, True)
    assert best_possible == 40
    assert _le8_tier(best_possible) == "Low"


def test_hba1c_same_value_scores_differently_by_diabetes_status():
    """6.0% is Intermediate (60) without a diagnosis but the diabetic scale's
    maximum (40) with one — the branch that app.py:1889-1892 warns the model
    not to silently re-evaluate."""
    assert _score_hba1c(6.0, False) == 60
    assert _score_hba1c(6.0, True) == 40


# ===========================================================================
# Out-of-range input — REGRESSION TESTS. Bugs #1/#4 and #2, FIXED 2026-08-04.
#
# WAS: none of the three scoring functions validated its input. A negative or
# zero value fell through the first `<` comparison and scored a perfect
# 100 = "Ideal", which was then written into an authoritative system note —
# "scores EXACTLY 100/100 (Ideal tier). Report this precisely; do not
# recalculate it" (app.py:1884-1912) — that the model is instructed not to
# second-guess. Values arrive from regexes over free-text user chat
# (app.py:1784-1789), so a typo was enough to trigger it.
#
# NOW: each function raises ValueError on `value <= 0`. A single guard placed
# above the has_diabetes branch in _score_hba1c covers both scales — the
# non-diabetic and diabetic cases were never independent bugs, just two
# consequences of the same missing check.
#
# ValueError rather than a clamp because there is no honest clamp target:
# clamping a negative to 100 IS the original bug, and clamping to 0 would
# fabricate a critical-range reading from a typo.
#
# THE OTHER HALF OF THE FIX (bug #2, same change): the raise is only safe
# because _build_computed_value_note now catches ValueError per metric and
# skips that one note (app.py:1897-1957). One bad value never suppresses the
# notes for the other metrics, and if every value is rejected the function
# degrades to the "" it already returns when it finds nothing (app.py:1863).
# Without that catch, the raise would surface as a 500 on the chat route.
# If you ever remove the try/except, these tests still pass while the app
# breaks — so do not treat them as covering the caller.
#
# These are now plain passing assertions. Reverting either half of the fix
# turns them red.
# ===========================================================================

@pytest.mark.parametrize("value", [-59, -1, 0])
def test_non_hdl_rejects_impossible_values(value):
    """Non-HDL cholesterol of 0 or less does not exist."""
    with pytest.raises(ValueError):
        _score_non_hdl(value)


@pytest.mark.parametrize("value", [-59, -1, 0])
def test_fasting_glucose_rejects_impossible_values(value):
    """A living person's fasting glucose is never 0 or negative."""
    with pytest.raises(ValueError):
        _score_fasting_glucose(value)


@pytest.mark.parametrize("value", [-5, -1, 0])
def test_hba1c_nondiabetic_rejects_impossible_values(value):
    """HbA1c is a percentage of glycated hemoglobin; 0% or negative is
    impossible."""
    with pytest.raises(ValueError):
        _score_hba1c(value, False)


@pytest.mark.parametrize("value", [-5, -1, 0])
def test_hba1c_diabetic_rejects_impossible_values(value):
    """The guard sits above the has_diabetes branch, so the diabetic scale is
    covered by the same check — this is bug #4, fixed by the same line as #1."""
    with pytest.raises(ValueError):
        _score_hba1c(value, True)


# --- high end: NOT a bug, asserted normally ------------------------------
# An implausibly high value (10x the top threshold) falls through every
# branch to the final `return 0`. That is effectively a clamp to the bottom
# of the scale, and it is directionally safe: it reports the worst score
# rather than a falsely reassuring one. Unvalidated, but not a patient-safety
# hazard, so these are plain passing assertions that pin the behavior.

@pytest.mark.parametrize("value,expected", [
    (2200, 0),      # 10x the 220 mg/dL top threshold
    (1_000_000, 0),
])
def test_non_hdl_implausibly_high_clamps_to_zero(value, expected):
    assert _score_non_hdl(value) == expected
    assert _le8_tier(_score_non_hdl(value)) == "Low"


@pytest.mark.parametrize("value,expected", [
    (1260, 0),      # 10x the 126 mg/dL top threshold
    (1_000_000, 0),
])
def test_fasting_glucose_implausibly_high_clamps_to_zero(value, expected):
    assert _score_fasting_glucose(value) == expected


@pytest.mark.parametrize("value,expected", [
    (65, 0),        # 10x the 6.5% top threshold
    (1_000_000, 0),
])
def test_hba1c_implausibly_high_clamps_to_zero(value, expected):
    assert _score_hba1c(value, False) == expected
    assert _score_hba1c(value, True) == expected


# ===========================================================================
# Cross-check: every emittable score maps to a tier without raising.
# ===========================================================================

def test_every_emittable_score_maps_to_a_valid_tier():
    # All sweeps start at 1, not 0 — a value of 0 is physiologically
    # impossible for every one of these metrics and now raises ValueError.
    scores = set()
    scores.update(_score_non_hdl(v) for v in range(1, 400))
    scores.update(_score_fasting_glucose(v) for v in range(1, 400))
    scores.update(_score_hba1c(v / 10, False) for v in range(1, 200))
    scores.update(_score_hba1c(v / 10, True) for v in range(1, 200))
    for score in scores:
        assert _le8_tier(score) in {"Ideal", "Intermediate", "Low"}


# ===========================================================================
# The five metrics with NO Python implementation.
#
# Blood pressure, BMI, sleep, physical activity, diet and smoking are scored
# on the frontend and arrive pre-computed (app.py:1956-1972). Their only
# definition in this repo is prose in the system prompt, which means an
# accidental edit to that prose silently changes what the bot tells users
# and nothing catches it.
#
# Writing Python reference implementations here would be tautological — it
# would test code no caller uses. Instead these assert the prompt text still
# literally contains each threshold table. Not a semantic check; a
# regression guard on the source of truth.
#
# KNOWN TRADEOFF — these are exact substring matches, so they are fragile to
# HARMLESS edits: reflowing a line, changing indentation, or realigning the
# `->` columns will fail them without any threshold actually changing. That
# is the accepted cost of guarding prose with no parser. WHEN ONE FAILS, DIFF
# THE PROMPT FIRST — confirm whether a number moved or only whitespace did,
# before treating it as a real scoring regression.
# ===========================================================================

_APP_SOURCE = read_app_source()


@pytest.mark.parametrize("metric,threshold_line", [
    # app.py:3117
    ("tier cutoffs",
     "Score tiers: 0-49 = Low | 50-79 = Intermediate | 80-100 = Ideal"),
    # app.py:3121
    ("physical activity",
     "Score = (steps / goal) x 100, capped at 100. Default goal: 10,000 steps/day."),
    # app.py:3125
    ("sleep",
     "Score = (hours / 8) x 100, capped at 100."),
    # app.py:3130-3134
    ("blood pressure ideal",    "<120 / <80   -> 100 (Ideal)"),
    ("blood pressure elevated", "120-129 / <80 -> 90"),
    ("blood pressure stage 1",  "130-139 OR 80-89 -> 75"),
    ("blood pressure stage 2",  "140-159 OR 90-99 -> 50"),
    ("blood pressure crisis",   ">=160 OR >=100   -> 0"),
    # app.py:3153
    ("bmi", "<25 -> 100 | 25-29.9 -> 70 | 30-34.9 -> 30 | 35-39.9 -> 15 | >=40 -> 0"),
    # app.py:3160
    ("diet mepa", "8-10 pts -> 100 | 6-7 -> 80 | 4-5 -> 50 | 2-3 -> 25 | 0-1 -> 0"),
    # app.py:3165-3171
    ("smoking never",           "Never smoked                -> 100"),
    ("smoking quit 5+",         "Quit 5+ years ago           -> 100"),
    ("smoking quit 1-4",        "Quit 1-4 years ago          -> 75"),
    ("smoking quit <1",         "Quit under 1 year ago       -> 50"),
    ("smoking current rare",    "Current smoker (rarely)     -> 25"),
    ("smoking current regular", "Current smoker (regularly)  -> 0"),
    ("smoking secondhand", "Secondhand exposure in home -> deduct 20 pts, floor at 0."),
])
def test_system_prompt_still_defines_threshold_table(metric, threshold_line):
    """These five metrics have no Python implementation — the prompt IS the
    implementation. If this fails, someone edited a threshold table and the
    chatbot may now be quoting different numbers to users."""
    assert threshold_line in _APP_SOURCE, (
        f"The {metric} threshold table in the system prompt no longer matches "
        "this exact string.\n"
        "DIFF THE PROMPT BEFORE ASSUMING A REAL REGRESSION: this is a substring "
        "guard, so reformatting/whitespace alone can trip it. If only formatting "
        "changed, update the expected string here. If a NUMBER changed, confirm "
        "that was intentional — it changes what the bot tells users."
    )


@pytest.mark.parametrize("prompt_line,fn,cases", [
    # Blood sugar, app.py:3138-3141
    ("No diabetes, fasting glucose (mg/dL): <100 -> 100 | 100-125 -> 60 | >=126 -> 0",
     _score_fasting_glucose, [(99, 100), (100, 60), (126, 0)]),
    ("No diabetes, HbA1c (%):              <5.7 -> 100 | 5.7-6.4 -> 60 | >=6.5 -> 0",
     lambda v: _score_hba1c(v, False), [(5.6, 100), (5.7, 60), (6.5, 0)]),
    # Blood lipids, app.py:3147
    ("<130  -> 100 | 130-159 -> 60 | 160-189 -> 40 | 190-219 -> 20 | >=220 -> 0",
     _score_non_hdl, [(129, 100), (130, 60), (220, 0)]),
])
def test_prompt_table_and_python_agree(prompt_line, fn, cases):
    """For the three metrics that exist BOTH as prompt prose and as Python,
    assert the two have not drifted apart. The model is told to trust the
    prompt table; the COMPUTED VALUE note is generated from the Python. If
    they disagree, the bot contradicts itself."""
    assert prompt_line in _APP_SOURCE, (
        "Prompt threshold table changed — Python may now disagree with it. "
        "Diff the prompt before assuming a real regression (see note above)."
    )
    for value, expected in cases:
        assert fn(value) == expected, (
            f"Python scores {value} as {fn(value)}, prompt table says {expected}"
        )
