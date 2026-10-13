"""
Rule-based sarcasm and mocking detection for social media comments.

This module combines signals from VADER sentiment, emoji analysis, and simple
text patterns. It does NOT use machine learning.

Important:
- A single emoji or phrase is rarely enough to prove sarcasm or mocking.
- sarcasm_detected and mocking_detected are True only when several signals
  together provide reasonably strong evidence.
- Sarcasm (ironic meaning) and mocking (ridicule) are tracked separately because
  they can overlap but are not the same thing.
"""

import re


# ---------------------------------------------------------------------------
# Mocking emojis — signals only, not automatic proof of sarcasm.
# These match the set used in emoji_analysis.py for consistency.
# ---------------------------------------------------------------------------
MOCKING_EMOJIS = ("🙄", "🤡", "😏", "😒")

# Laughter emojis that can amplify "positive words + ridicule" patterns.
LAUGHTER_EMOJIS = ("😂", "🤣", "😆")

# Phrases that often carry sarcastic tone when used in the right context.
# Multi-word phrases are safer than single words like "sure" on their own.
SARCASTIC_PHRASES = [
    "yeah right",
    "oh great",
    "great job",
    "brilliant idea",
    "what a genius",
    "how wonderful",
    "thanks for nothing",
    "love that for you",
]

# "sure" appears in sincere replies too — only count in clearer sarcastic patterns.
AMBIGUOUS_SARCASTIC_PHRASES = [
    "oh sure",
    "yeah sure",
    "sure buddy",
    "sure pal",
]

# Positive-sounding words that clash with mocking/laughter signals.
POSITIVE_MOCKING_WORDS = [
    "great",
    "brilliant",
    "nice",
    "fantastic",
    "wonderful",
    "genius",
    "love",
    "amazing",
    "perfect",
    "beautiful",
    "awesome",
    "lovely",
    "excellent",
    "update",
    "idea",
    "job",
]

# Phrases that lean more toward direct ridicule than subtle irony.
MOCKING_PHRASES = [
    "what a genius",
    "what a joke",
    "clown",
    "nice one",
    "good one",
    "well done",
    "slow clap",
]

# Minimum sarcasm_score before sarcasm_detected can be True.
SARCASM_SCORE_THRESHOLD = 0.55

# Minimum internal mocking score before mocking_detected can be True.
MOCKING_SCORE_THRESHOLD = 0.50

# Weights for sarcasm-related signals (capped total at 1.0).
SARCASM_WEIGHTS = {
    "positive sentiment conflicts with sarcasm/mocking emoji": 0.45,
    "positive sentiment conflicts with negative emoji tone": 0.35,
    "mocking emoji detected": 0.30,
    "sarcastic phrase detected": 0.30,
    "ambiguous sarcastic phrase detected": 0.15,
    "positive language with laughter or mocking emoji": 0.35,
    "excessive exclamation marks (supporting signal)": 0.10,
    "excessive question marks (supporting signal)": 0.08,
}

# Weights for mocking-specific signals.
MOCKING_WEIGHTS = {
    "mocking emoji detected": 0.35,
    "mocking phrase detected": 0.30,
    "positive praise paired with laughter emoji": 0.35,
    "positive praise paired with clown or eye-roll emoji": 0.40,
    "ridicule pattern with laughter emoji": 0.30,
}

# Punctuation counts above these values add a small supporting bump only.
EXCESSIVE_EXCLAMATION_THRESHOLD = 2
EXCESSIVE_QUESTION_THRESHOLD = 2

# Word token pattern for finding positive words in text.
WORD_PATTERN = re.compile(r"[a-z']+", re.IGNORECASE)


def empty_sarcasm_result() -> dict:
    """Default result when there is nothing meaningful to analyze."""
    return {
        "sarcasm_score": 0.0,
        "sarcasm_detected": False,
        "sarcasm_reasons": [],
        "mocking_detected": False,
        "mocking_reasons": [],
    }


def _normalize_text(text) -> str:
    """Convert input to a safe string (None becomes empty)."""
    if text is None:
        return ""
    return str(text)


def _normalize_sentiment_label(label) -> str:
    """Normalize sentiment labels to uppercase strings."""
    if label is None:
        return "NEUTRAL"
    return str(label).strip().upper()


def _normalize_emoji_sentiment(emoji_sentiment) -> str:
    """Normalize emoji sentiment labels to uppercase strings."""
    if emoji_sentiment is None:
        return "NEUTRAL"
    return str(emoji_sentiment).strip().upper()


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Convert a value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _find_phrases(text_lower: str, phrases: list[str]) -> list[str]:
    """
    Return which phrases from the list appear in the lowercased text.

    We use simple substring matching for multi-word phrases because social
    media comments are short and informal.
    """
    return [phrase for phrase in phrases if phrase in text_lower]


def _find_mocking_emojis_in_text(text: str) -> list[str]:
    """Return mocking emojis present in the raw comment text."""
    return [emoji for emoji in MOCKING_EMOJIS if emoji in text]


def _find_laughter_emojis_in_text(text: str) -> list[str]:
    """Return laughter emojis present in the raw comment text."""
    return [emoji for emoji in LAUGHTER_EMOJIS if emoji in text]


def _has_positive_language(text_lower: str) -> bool:
    """
    Check whether the comment contains praise-like words.

    Used for "Great idea 😂" style combinations — praise plus ridicule signals.
    """
    words = set(WORD_PATTERN.findall(text_lower))
    return any(word in POSITIVE_MOCKING_WORDS for word in words)


def _has_strong_positive_sentiment(sentiment_label: str, compound_score: float) -> bool:
    """
    Treat VADER output as strongly positive when the label or compound says so.

    We accept moderately positive compound scores because sarcasm often still
    reads as positive to lexicon-based tools.
    """
    if sentiment_label == "POSITIVE":
        return True
    return compound_score >= 0.35


def _compute_sarcasm_signals(
    text_lower: str,
    sentiment_label: str,
    compound_score: float,
    emoji_sentiment: str,
    sarcasm_emoji_detected: bool,
    exclamation_count: int,
    question_count: int,
    has_positive_language: bool,
    has_laughter_emoji: bool,
    has_mocking_emoji: bool,
) -> tuple[float, list[str]]:
    """
    Collect sarcasm reasons and compute a score from 0.0 to 1.0.

    Each rule adds weight only when it fires. The total is capped at 1.0.
    """
    reasons: list[str] = []
    score = 0.0
    strong_positive = _has_strong_positive_sentiment(sentiment_label, compound_score)

    # --- Rule 1: Sentiment vs emoji contradiction ---
    # Positive words from VADER clash with eye-roll/clown-style emojis.
    # Example: "Fantastic update 🙄"
    if strong_positive and (sarcasm_emoji_detected or has_mocking_emoji):
        reason = "positive sentiment conflicts with sarcasm/mocking emoji"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    # --- Rule 2: Mocking emojis as sarcasm-supporting signals ---
    # A 🙄 or 🤡 alone is not enough for sarcasm_detected, but it adds evidence
    # when combined with contradiction or phrase rules.
    if sarcasm_emoji_detected or has_mocking_emoji:
        reason = "mocking emoji detected"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    # Positive VADER label but emojis read negative — another irony hint.
    if strong_positive and emoji_sentiment == "NEGATIVE":
        reason = "positive sentiment conflicts with negative emoji tone"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    # --- Rule 3: Sarcastic phrases ---
    # Clear multi-word phrases get more weight than ambiguous ones.
    sarcastic_hits = _find_phrases(text_lower, SARCASTIC_PHRASES)
    if sarcastic_hits:
        reason = "sarcastic phrase detected"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    ambiguous_hits = _find_phrases(text_lower, AMBIGUOUS_SARCASTIC_PHRASES)
    if ambiguous_hits:
        reason = "ambiguous sarcastic phrase detected"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    # --- Rule 4: Positive language + laughter/mocking combination ---
    # "Brilliant 😂" or "Great idea 🤡" — praise plus ridicule cues.
    # Skip when Rule 1 already captured the same emoji contradiction so we
    # do not double-count one emoji twice in the sarcasm score.
    contradiction_fired = (
        "positive sentiment conflicts with sarcasm/mocking emoji" in reasons
    )
    if (
        has_positive_language
        and (has_laughter_emoji or has_mocking_emoji)
        and not contradiction_fired
    ):
        reason = "positive language with laughter or mocking emoji"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    # --- Rule 5: Excessive punctuation (supporting only) ---
    # Never enough on its own; only nudges the score when other cues exist.
    has_other_signals = len(reasons) > 0 or has_mocking_emoji or has_laughter_emoji

    if has_other_signals and exclamation_count >= EXCESSIVE_EXCLAMATION_THRESHOLD:
        reason = "excessive exclamation marks (supporting signal)"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    if has_other_signals and question_count >= EXCESSIVE_QUESTION_THRESHOLD:
        reason = "excessive question marks (supporting signal)"
        if reason not in reasons:
            reasons.append(reason)
            score += SARCASM_WEIGHTS[reason]

    score = min(1.0, round(score, 3))
    return score, reasons


def _compute_mocking_signals(
    text_lower: str,
    has_positive_language: bool,
    has_laughter_emoji: bool,
    has_mocking_emoji: bool,
    mocking_emojis_found: list[str],
    sarcasm_emoji_detected: bool,
) -> tuple[float, list[str]]:
    """
    Collect mocking-specific reasons and compute a separate mocking score.

    Mocking focuses on ridicule cues (clown emoji, laugh-at-someone patterns)
    rather than subtle ironic praise alone.
    """
    reasons: list[str] = []
    score = 0.0

    # --- Rule 2: Mocking emojis as signals ---
    # Presence of 🙄 🤡 😏 😒 increases mocking evidence but is not proof alone.
    if sarcasm_emoji_detected or has_mocking_emoji:
        reason = "mocking emoji detected"
        if reason not in reasons:
            reasons.append(reason)
            score += MOCKING_WEIGHTS[reason]

    # --- Direct ridicule phrases ---
    mocking_hits = _find_phrases(text_lower, MOCKING_PHRASES)
    if mocking_hits:
        reason = "mocking phrase detected"
        if reason not in reasons:
            reasons.append(reason)
            score += MOCKING_WEIGHTS[reason]

    # --- Rule 4 (mocking angle): praise + laughter ---
    if has_positive_language and has_laughter_emoji:
        reason = "positive praise paired with laughter emoji"
        if reason not in reasons:
            reasons.append(reason)
            score += MOCKING_WEIGHTS[reason]

    # Praise paired with classic mocking emojis (not just any emoji).
    if has_positive_language and (has_mocking_emoji or sarcasm_emoji_detected):
        clown_or_eyeroll = any(
            emoji in mocking_emojis_found for emoji in ("🙄", "🤡", "😏", "😒")
        )
        if clown_or_eyeroll or sarcasm_emoji_detected:
            reason = "positive praise paired with clown or eye-roll emoji"
            if reason not in reasons:
                reasons.append(reason)
                score += MOCKING_WEIGHTS[reason]

    # Short ridicule comments that are mostly laughter.
    if has_laughter_emoji and (
        mocking_hits or has_mocking_emoji or sarcasm_emoji_detected
    ):
        reason = "ridicule pattern with laughter emoji"
        if reason not in reasons:
            reasons.append(reason)
            score += MOCKING_WEIGHTS[reason]

    score = min(1.0, round(score, 3))
    return score, reasons


def _is_sarcasm_detected(sarcasm_score: float, sarcasm_reasons: list[str]) -> bool:
    """
    Decide whether sarcasm_detected should be True.

    We require:
    1. A score at or above SARCASM_SCORE_THRESHOLD, and
    2. At least two distinct reasons, OR one strong contradiction reason plus
       another supporting reason.

    This stops a lone weak phrase or a single emoji from flipping the flag.
    """
    if sarcasm_score < SARCASM_SCORE_THRESHOLD:
        return False

    if len(sarcasm_reasons) >= 2:
        return True

    strong_reasons = {
        "positive sentiment conflicts with sarcasm/mocking emoji",
        "positive sentiment conflicts with negative emoji tone",
        "positive language with laughter or mocking emoji",
        "sarcastic phrase detected",
    }

    return any(reason in strong_reasons for reason in sarcasm_reasons)


def _is_mocking_detected(mocking_score: float, mocking_reasons: list[str]) -> bool:
    """
    Decide whether mocking_detected should be True.

    Mocking needs a decent score and either multiple reasons or one strong
    ridicule pattern (emoji + praise, or phrase + laughter).
    """
    if mocking_score < MOCKING_SCORE_THRESHOLD:
        return False

    if len(mocking_reasons) >= 2:
        return True

    strong_mocking_reasons = {
        "positive praise paired with clown or eye-roll emoji",
        "positive praise paired with laughter emoji",
        "ridicule pattern with laughter emoji",
        "mocking phrase detected",
    }

    return any(reason in strong_mocking_reasons for reason in mocking_reasons)


def analyze_sarcasm(
    text,
    sentiment_label,
    compound_score,
    emoji_sentiment,
    sarcasm_emoji_detected,
    exclamation_count,
    question_count,
) -> dict:
    """
    Analyze a comment for sarcasm and mocking signals.

    Parameters
    ----------
    text : str
        The comment text (usually original or cleaned text from preprocessing).
    sentiment_label : str
        VADER label: POSITIVE, NEGATIVE, or NEUTRAL.
    compound_score : float
        VADER compound score from -1.0 to 1.0.
    emoji_sentiment : str
        Overall emoji tone from emoji analysis (e.g. POSITIVE, LAUGHTER).
    sarcasm_emoji_detected : bool
        True when emoji analysis found a sarcasm/mocking emoji.
    exclamation_count : int
        Number of '!' characters from preprocessing.
    question_count : int
        Number of '?' characters from preprocessing.

    Returns
    -------
    dict
        sarcasm_score     — float from 0.0 to 1.0
        sarcasm_detected  — True when sarcasm evidence is reasonably strong
        sarcasm_reasons   — human-readable sarcasm explanation strings
        mocking_detected  — True when ridicule evidence is reasonably strong
        mocking_reasons   — human-readable mocking explanation strings
    """
    safe_text = _normalize_text(text)

    if safe_text.strip() == "":
        return empty_sarcasm_result()

    text_lower = safe_text.lower()
    normalized_label = _normalize_sentiment_label(sentiment_label)
    normalized_emoji_sentiment = _normalize_emoji_sentiment(emoji_sentiment)
    compound = _safe_float(compound_score)
    exclamations = _safe_int(exclamation_count)
    questions = _safe_int(question_count)
    sarcasm_emoji_flag = bool(sarcasm_emoji_detected)

    mocking_emojis_found = _find_mocking_emojis_in_text(safe_text)
    laughter_emojis_found = _find_laughter_emojis_in_text(safe_text)
    has_mocking_emoji = len(mocking_emojis_found) > 0
    has_laughter_emoji = len(laughter_emojis_found) > 0
    has_positive_language = _has_positive_language(text_lower)

    sarcasm_score, sarcasm_reasons = _compute_sarcasm_signals(
        text_lower=text_lower,
        sentiment_label=normalized_label,
        compound_score=compound,
        emoji_sentiment=normalized_emoji_sentiment,
        sarcasm_emoji_detected=sarcasm_emoji_flag,
        exclamation_count=exclamations,
        question_count=questions,
        has_positive_language=has_positive_language,
        has_laughter_emoji=has_laughter_emoji,
        has_mocking_emoji=has_mocking_emoji,
    )

    mocking_score, mocking_reasons = _compute_mocking_signals(
        text_lower=text_lower,
        has_positive_language=has_positive_language,
        has_laughter_emoji=has_laughter_emoji,
        has_mocking_emoji=has_mocking_emoji,
        mocking_emojis_found=mocking_emojis_found,
        sarcasm_emoji_detected=sarcasm_emoji_flag,
    )

    return {
        "sarcasm_score": sarcasm_score,
        "sarcasm_detected": _is_sarcasm_detected(sarcasm_score, sarcasm_reasons),
        "sarcasm_reasons": sarcasm_reasons,
        "mocking_detected": _is_mocking_detected(mocking_score, mocking_reasons),
        "mocking_reasons": mocking_reasons,
    }
