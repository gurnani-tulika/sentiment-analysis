"""
Rule-based emoji analysis for social media comments.

This module looks at emojis already extracted by preprocessing and groups them
into simple categories. It does NOT use machine learning.

Important: emojis like 🙄 or 🤡 can appear in sarcastic comments, but seeing
them alone does not prove sarcasm. The sarcasm_emoji_detected flag only means
"a commonly associated emoji was present" — not a final sarcasm verdict.
"""

import re


# Match one emoji character at a time (no "+" at the end).
# Used to split clusters such as "😂😂" into individual emojis.
INDIVIDUAL_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F680-\U0001F6FF"  # transport and map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]",
    flags=re.UNICODE,
)

# Category names returned in emoji_categories.
CATEGORY_POSITIVE = "positive"
CATEGORY_NEGATIVE = "negative"
CATEGORY_LAUGHTER = "laughter"
CATEGORY_SARCASM_OR_MOCKING = "sarcasm_or_mocking"
CATEGORY_SURPRISE = "surprise"
CATEGORY_NEUTRAL_OR_OTHER = "neutral_or_other"

ALL_CATEGORIES = [
    CATEGORY_POSITIVE,
    CATEGORY_NEGATIVE,
    CATEGORY_LAUGHTER,
    CATEGORY_SARCASM_OR_MOCKING,
    CATEGORY_SURPRISE,
    CATEGORY_NEUTRAL_OR_OTHER,
]

# Map each known emoji to exactly one category.
# Include common variants (with and without variation selectors) where helpful.
EMOJI_TO_CATEGORY = {
    # Positive
    "😍": CATEGORY_POSITIVE,
    "❤": CATEGORY_POSITIVE,
    "❤️": CATEGORY_POSITIVE,
    "❤️‍🔥": CATEGORY_POSITIVE,
    "😊": CATEGORY_POSITIVE,
    "😄": CATEGORY_POSITIVE,
    "😁": CATEGORY_POSITIVE,
    "👍": CATEGORY_POSITIVE,
    "👏": CATEGORY_POSITIVE,
    "🎉": CATEGORY_POSITIVE,
    # Negative
    "😡": CATEGORY_NEGATIVE,
    "😠": CATEGORY_NEGATIVE,
    "😞": CATEGORY_NEGATIVE,
    "😢": CATEGORY_NEGATIVE,
    "😭": CATEGORY_NEGATIVE,
    "💔": CATEGORY_NEGATIVE,
    "👎": CATEGORY_NEGATIVE,
    # Laughter
    "😂": CATEGORY_LAUGHTER,
    "🤣": CATEGORY_LAUGHTER,
    "😆": CATEGORY_LAUGHTER,
    # Sarcasm / mocking (signal only — not proof of sarcasm)
    "🙄": CATEGORY_SARCASM_OR_MOCKING,
    "🤡": CATEGORY_SARCASM_OR_MOCKING,
    "😏": CATEGORY_SARCASM_OR_MOCKING,
    "😒": CATEGORY_SARCASM_OR_MOCKING,
    # Surprise
    "😮": CATEGORY_SURPRISE,
    "😲": CATEGORY_SURPRISE,
    "😳": CATEGORY_SURPRISE,
}

# Check longer emoji sequences first (e.g. ❤️‍🔥 before ❤️).
MULTI_CHAR_EMOJIS = sorted(EMOJI_TO_CATEGORY.keys(), key=len, reverse=True)


def empty_emoji_result() -> dict:
    """Default result when there are no emojis to analyze."""
    return {
        "emoji_sentiment": "NEUTRAL",
        "emoji_categories": [],
        "sarcasm_emoji_detected": False,
        "emoji_count": 0,
    }


def _normalize_emoji_input(emojis) -> list[str]:
    """
    Convert the input into a flat list of individual emoji strings.

    Accepts the emojis list from preprocessing, an empty list, None, or "".
    """
    if emojis is None:
        return []

    if isinstance(emojis, str):
        if emojis.strip() == "":
            return []
        source_text = emojis
    elif isinstance(emojis, list):
        if len(emojis) == 0:
            return []
        # Join list items in case preprocessing grouped adjacent emojis together.
        source_text = "".join(str(item) for item in emojis if item is not None)
        if source_text.strip() == "":
            return []
    else:
        # Unknown input type — treat as empty for safety.
        return []

    return _extract_individual_emojis(source_text)


def _extract_individual_emojis(text: str) -> list[str]:
    """
    Pull individual emojis out of a string.

    Step 1: look for known multi-character emojis (e.g. ❤️‍🔥).
    Step 2: use regex to find remaining single emojis one at a time.
    """
    remaining = text
    found: list[str] = []

    # Pass 1 — match longer known sequences first so they are not split apart.
    for emoji in MULTI_CHAR_EMOJIS:
        if len(emoji) <= 1:
            continue
        while emoji in remaining:
            found.append(emoji)
            remaining = remaining.replace(emoji, "", 1)

    # Pass 2 — anything left is split into one emoji per match.
    found.extend(INDIVIDUAL_EMOJI_PATTERN.findall(remaining))
    return found


def _categorize_emoji(emoji: str) -> str:
    """Return the category for one emoji, or neutral_or_other if unknown."""
    if emoji in EMOJI_TO_CATEGORY:
        return EMOJI_TO_CATEGORY[emoji]

    # Some platforms store ❤️ as "❤" plus a variation selector — treat as positive.
    if emoji.strip("️") in EMOJI_TO_CATEGORY:
        return EMOJI_TO_CATEGORY[emoji.strip("️")]

    return CATEGORY_NEUTRAL_OR_OTHER


def _count_by_category(emoji_list: list[str]) -> dict[str, int]:
    """Count how many emojis fall into each category."""
    counts = {category: 0 for category in ALL_CATEGORIES}
    for emoji in emoji_list:
        category = _categorize_emoji(emoji)
        counts[category] += 1
    return counts


def _categories_present(counts: dict[str, int]) -> list[str]:
    """
    Build the emoji_categories list.

    Only includes categories that had at least one emoji.
    Order follows ALL_CATEGORIES so results are predictable.
    """
    return [category for category in ALL_CATEGORIES if counts[category] > 0]


def _determine_emoji_sentiment(counts: dict[str, int]) -> str:
    """
    Pick one overall emoji_sentiment label using simple rules.

    This is a rough summary — emoji_categories keeps the full picture when
    multiple types of emojis appear together.
    """
    positive_signal = counts[CATEGORY_POSITIVE] + counts[CATEGORY_LAUGHTER]
    negative_signal = counts[CATEGORY_NEGATIVE]

    has_positive = counts[CATEGORY_POSITIVE] > 0
    has_laughter = counts[CATEGORY_LAUGHTER] > 0
    has_negative = counts[CATEGORY_NEGATIVE] > 0
    has_surprise = counts[CATEGORY_SURPRISE] > 0
    has_neutral = counts[CATEGORY_NEUTRAL_OR_OTHER] > 0

    # Conflicting strong signals → MIXED.
    if has_positive and has_negative:
        return "MIXED"
    if has_laughter and has_negative:
        return "MIXED"

    if has_negative and negative_signal >= positive_signal:
        return "NEGATIVE"

    if has_positive or (has_laughter and not has_negative):
        return "POSITIVE"

    if has_laughter and not has_negative and not has_positive:
        return "LAUGHTER"

    if has_surprise and not has_positive and not has_negative and not has_laughter:
        return "NEUTRAL"

    if has_neutral and not any(
        counts[category] > 0
        for category in ALL_CATEGORIES
        if category != CATEGORY_NEUTRAL_OR_OTHER
    ):
        return "NEUTRAL"

    # Sarcasm/mocking emojis alone do not change sentiment to "sarcastic".
    return "NEUTRAL"


def analyze_emojis(emojis) -> dict:
    """
    Analyze a list (or string) of emojis and return category summaries.

    Parameters
    ----------
    emojis : list[str] | str | None
        Usually the "emojis" field from preprocessing.analyze_text().
        Also safely handles None, "", and [].

    Returns
    -------
    dict
        emoji_sentiment       — overall label (POSITIVE, NEGATIVE, NEUTRAL, etc.)
        emoji_categories      — all categories found (may be more than one)
        sarcasm_emoji_detected — True if a sarcasm/mocking emoji was seen
        emoji_count           — total number of individual emojis
    """
    emoji_list = _normalize_emoji_input(emojis)

    if len(emoji_list) == 0:
        return empty_emoji_result()

    counts = _count_by_category(emoji_list)
    categories = _categories_present(counts)

    return {
        "emoji_sentiment": _determine_emoji_sentiment(counts),
        "emoji_categories": categories,
        "sarcasm_emoji_detected": counts[CATEGORY_SARCASM_OR_MOCKING] > 0,
        "emoji_count": len(emoji_list),
    }
