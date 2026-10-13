"""
Rule-based spam and hashtag-hijacking analysis for social media comments.

This module looks for common spam patterns using simple, transparent rules.
It does NOT use machine learning.

Important:
- A URL alone does not mean spam — many legitimate comments include links.
- Hashtag "unrelatedness" cannot be judged perfectly with rules alone.
  The hashtag_hijacking_signal is a heuristic hint, not a final verdict.
"""

import re


# ---------------------------------------------------------------------------
# Promotional phrases often seen in spam comments (matched case-insensitively).
# ---------------------------------------------------------------------------
PROMOTIONAL_PHRASES = [
    "buy now",
    "click here",
    "limited offer",
    "discount",
    "dm me",
    "message me",
    "follow me",
    "check my profile",
    "visit my page",
    "subscribe",
    "giveaway",
]

# Hashtag terms that often appear in promotional or trend-chasing posts.
# We compare the hashtag text (without #) against this list.
PROMOTIONAL_HASHTAG_TERMS = {
    "giveaway",
    "free",
    "discount",
    "sale",
    "promo",
    "promotion",
    "crypto",
    "bitcoin",
    "nft",
    "follow",
    "followme",
    "subscribe",
    "subscribers",
    "linkinbio",
    "viral",
    "trending",
    "win",
    "contest",
    "sweepstakes",
    "dm",
    "shop",
    "buynow",
}

# How many hashtags start to look excessive on a typical comment.
EXCESSIVE_HASHTAG_THRESHOLD = 5
VERY_EXCESSIVE_HASHTAG_THRESHOLD = 8

# Minimum spam_score needed before we label a comment as spam.
# URL alone (0.15) stays below this bar.
SPAM_SCORE_THRESHOLD = 0.55

# Weights for each spam reason — kept in one place so scoring is easy to read.
SPAM_WEIGHTS = {
    "promotional phrase detected": 0.35,
    "URL detected": 0.15,
    "excessive hashtags detected": 0.25,
    "very excessive hashtags detected": 0.15,  # extra bump on top of the above
    "excessive repetition detected": 0.15,
    "promotional hashtags detected": 0.25,
    "multiple promotional phrases detected": 0.15,
}

# Token pattern: words made of letters and digits (used for topic matching).
WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def empty_spam_result() -> dict:
    """Default result when there is nothing meaningful to analyze."""
    return {
        "spam": False,
        "spam_score": 0.0,
        "spam_reasons": [],
        "hashtag_hijacking_signal": False,
        "hashtag_hijacking_reasons": [],
    }


def _normalize_text(text) -> str:
    """Convert input to a safe string (None becomes empty)."""
    if text is None:
        return ""
    return str(text)


def _normalize_hashtags(hashtags) -> list[str]:
    """
    Convert hashtags input into a clean list of strings.

    Accepts the list from preprocessing, None, or an empty list.
    """
    if hashtags is None:
        return []
    if isinstance(hashtags, str):
        return [hashtags] if hashtags.strip() else []
    if isinstance(hashtags, list):
        return [str(tag) for tag in hashtags if tag is not None and str(tag).strip()]
    return []


def _find_promotional_phrases(text: str) -> list[str]:
    """
    Return which promotional phrases appear in the text.

    We search the lowercased text so "Buy Now" and "buy now" both match.
    """
    lower_text = text.lower()
    found = []
    for phrase in PROMOTIONAL_PHRASES:
        if phrase in lower_text:
            found.append(phrase)
    return found


def _hashtag_body(tag: str) -> str:
    """Strip the leading # and lowercase a hashtag for comparisons."""
    return tag.lstrip("#").lower()


def _find_promotional_hashtags(hashtags: list[str]) -> list[str]:
    """
    Return hashtags whose text matches known promotional/trending terms.

    Example: "#giveaway" matches because "giveaway" is in PROMOTIONAL_HASHTAG_TERMS.
    """
    found = []
    for tag in hashtags:
        body = _hashtag_body(tag)
        if body in PROMOTIONAL_HASHTAG_TERMS:
            found.append(tag)
    return found


def _text_without_hashtags(text: str) -> str:
    """Remove hashtag tokens so we can look at the comment topic separately."""
    return re.sub(r"#\w+", " ", text)


def _extract_topic_words(text: str) -> set[str]:
    """
    Pull out simple word tokens from the comment (excluding hashtags).

    These are used only as a rough hint — not perfect topic understanding.
    """
    text_without_tags = _text_without_hashtags(text)
    words = WORD_PATTERN.findall(text_without_tags.lower())
    # Ignore very short tokens (like "a", "ok") — they are not useful for matching.
    return {word for word in words if len(word) >= 3}


def _hashtag_relates_to_topic(hashtag_body: str, topic_words: set[str]) -> bool:
    """
    Guess whether a hashtag might relate to the comment text.

    A hashtag is considered "related" if:
    - its text equals a topic word, or
    - a topic word appears inside the hashtag, or vice versa.

    This is a simple heuristic — unrelated hashtags can still slip through.
    """
    if not topic_words:
        return False

    if hashtag_body in topic_words:
        return True

    for word in topic_words:
        if word in hashtag_body or hashtag_body in word:
            return True

    return False


def _count_unrelated_hashtags(hashtags: list[str], topic_words: set[str]) -> int:
    """Count hashtags that do not appear related to the comment topic words."""
    unrelated = 0
    for tag in hashtags:
        body = _hashtag_body(tag)
        if not _hashtag_relates_to_topic(body, topic_words):
            unrelated += 1
    return unrelated


def _compute_spam_signals(
    text: str,
    hashtags: list[str],
    has_url_flag: bool,
    excessive_repetition_flag: bool,
) -> tuple[float, list[str]]:
    """
    Collect spam reasons and compute a score from 0.0 to 1.0.

    Each reason adds a fixed weight (see SPAM_WEIGHTS).
    The total is capped at 1.0.
    """
    reasons: list[str] = []
    score = 0.0

    # --- Promotional phrases (strong spam signal) ---
    phrases_found = _find_promotional_phrases(text)
    if len(phrases_found) >= 1:
        reasons.append("promotional phrase detected")
        score += SPAM_WEIGHTS["promotional phrase detected"]
    if len(phrases_found) >= 2:
        reasons.append("multiple promotional phrases detected")
        score += SPAM_WEIGHTS["multiple promotional phrases detected"]

    # --- URL (weak signal on its own) ---
    if has_url_flag:
        reasons.append("URL detected")
        score += SPAM_WEIGHTS["URL detected"]

    # --- Hashtag volume ---
    hashtag_count = len(hashtags)
    if hashtag_count >= EXCESSIVE_HASHTAG_THRESHOLD:
        reasons.append("excessive hashtags detected")
        score += SPAM_WEIGHTS["excessive hashtags detected"]
    if hashtag_count >= VERY_EXCESSIVE_HASHTAG_THRESHOLD:
        reasons.append("very excessive hashtags detected")
        score += SPAM_WEIGHTS["very excessive hashtags detected"]

    # --- Character repetition (from preprocessing) ---
    if excessive_repetition_flag:
        reasons.append("excessive repetition detected")
        score += SPAM_WEIGHTS["excessive repetition detected"]

    # --- Promotional-looking hashtags ---
    promo_tags = _find_promotional_hashtags(hashtags)
    if len(promo_tags) >= 1:
        reasons.append("promotional hashtags detected")
        score += SPAM_WEIGHTS["promotional hashtags detected"]

    # Cap the score at 1.0 so it stays on a 0–1 scale.
    score = min(1.0, round(score, 3))
    return score, reasons


def _is_spam(spam_score: float, spam_reasons: list[str]) -> bool:
    """
    Decide whether spam is True.

    We require both:
    1. A score at or above SPAM_SCORE_THRESHOLD, and
    2. At least one "strong" reason (not URL or repetition alone).

    This stops innocent link-sharing from being labeled spam.
    """
    if spam_score < SPAM_SCORE_THRESHOLD:
        return False

    strong_reasons = {
        "promotional phrase detected",
        "multiple promotional phrases detected",
        "promotional hashtags detected",
        "excessive hashtags detected",
        "very excessive hashtags detected",
    }

    has_strong_signal = any(reason in strong_reasons for reason in spam_reasons)

    # URL + repetition together can also be strong enough at a high score.
    has_url_and_repetition = (
        "URL detected" in spam_reasons and "excessive repetition detected" in spam_reasons
    )

    return has_strong_signal or (has_url_and_repetition and spam_score >= 0.6)


def _compute_hashtag_hijacking_signals(
    text: str,
    hashtags: list[str],
) -> tuple[bool, list[str]]:
    """
    Look for signs that hashtags may be hijacking unrelated trends.

    We cannot know the "true" topic with rules alone — these are hints only.
    """
    reasons: list[str] = []
    hashtag_count = len(hashtags)

    if hashtag_count == 0:
        return False, reasons

    topic_words = _extract_topic_words(text)
    unrelated_count = _count_unrelated_hashtags(hashtags, topic_words)
    promo_tags = _find_promotional_hashtags(hashtags)

    # Signal 1: many hashtags where most do not overlap with comment words.
    if hashtag_count >= 4 and unrelated_count >= 3:
        share_unrelated = unrelated_count / hashtag_count
        if share_unrelated >= 0.6:
            reasons.append(
                "many hashtags appear unrelated to comment text (heuristic)"
            )

    # Signal 2: short comment body but many hashtags (common hijacking pattern).
    text_without_tags = _text_without_hashtags(text).strip()
    if hashtag_count >= 3 and len(text_without_tags) < 30:
        reasons.append("many hashtags in a very short comment")

    # Signal 3: several promotional/trending hashtags that do not match topic words.
    unrelated_promo_tags = [
        tag
        for tag in promo_tags
        if not _hashtag_relates_to_topic(_hashtag_body(tag), topic_words)
    ]
    if len(unrelated_promo_tags) >= 2:
        reasons.append("promotional or trending hashtags unrelated to comment text")

    # Signal 4: very high hashtag count regardless of topic match.
    if hashtag_count >= VERY_EXCESSIVE_HASHTAG_THRESHOLD:
        reasons.append("very high number of hashtags")

    # The signal is True when we collected at least one hijacking hint.
    return len(reasons) > 0, reasons


def analyze_spam(
    text,
    hashtags,
    has_url,
    excessive_repetition,
) -> dict:
    """
    Analyze a comment for spam and hashtag-hijacking signals.

    Parameters
    ----------
    text : str
        The comment text (usually original_text or cleaned_text from preprocessing).
    hashtags : list[str]
        Hashtags extracted by preprocessing (e.g. ["#python", "#AI"]).
    has_url : bool
        True when preprocessing found a URL in the comment.
    excessive_repetition : bool
        True when preprocessing found a character repeated 4+ times in a row.

    Returns
    -------
    dict
        spam                      — True only when spam evidence is reasonably strong
        spam_score                — float from 0.0 to 1.0
        spam_reasons              — list of human-readable reason strings
        hashtag_hijacking_signal  — heuristic True/False (not a final verdict)
        hashtag_hijacking_reasons — list of hijacking hint strings
    """
    safe_text = _normalize_text(text)
    safe_hashtags = _normalize_hashtags(hashtags)

    if safe_text.strip() == "" and len(safe_hashtags) == 0:
        return empty_spam_result()

    spam_score, spam_reasons = _compute_spam_signals(
        text=safe_text,
        hashtags=safe_hashtags,
        has_url_flag=bool(has_url),
        excessive_repetition_flag=bool(excessive_repetition),
    )

    hijacking_signal, hijacking_reasons = _compute_hashtag_hijacking_signals(
        text=safe_text,
        hashtags=safe_hashtags,
    )

    return {
        "spam": _is_spam(spam_score, spam_reasons),
        "spam_score": spam_score,
        "spam_reasons": spam_reasons,
        "hashtag_hijacking_signal": hijacking_signal,
        "hashtag_hijacking_reasons": hijacking_reasons,
    }
