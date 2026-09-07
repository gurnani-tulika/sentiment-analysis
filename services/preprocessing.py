"""
Text preprocessing utilities for social media comments.

Each function handles one small task. The main entry point is analyze_text(),
which combines everything into a single result dictionary.
"""

import re


# Matches common emoji ranges (faces, symbols, flags, etc.)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F680-\U0001F6FF"  # transport and map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)

# Matches hashtags like #python or #DataScience
HASHTAG_PATTERN = re.compile(r"#\w+", flags=re.UNICODE)

# Matches http/https links and www. links
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)

# Matches a character repeated 4 or more times in a row (e.g. "soooo")
EXCESSIVE_REPETITION_PATTERN = re.compile(r"(.)\1{3,}")


def is_missing_value(comment) -> bool:
    """Return True when the comment is None or a floating-point NaN."""
    if comment is None:
        return True
    if isinstance(comment, float) and comment != comment:
        return True
    return False


def to_standard_string(comment) -> str:
    """
    Convert a comment to a safe string.

    None and NaN become an empty string. Other values are converted with str().
    """
    if is_missing_value(comment):
        return ""
    return str(comment)


def is_empty_comment(comment) -> bool:
    """Return True when the comment is missing or contains only whitespace."""
    return to_standard_string(comment).strip() == ""


def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace into a single space and trim the ends.

    Example: "  hello   world  " -> "hello world"
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_emojis(text: str) -> list[str]:
    """
    Find all emojis in the text without changing the text itself.

    Returns a list of emoji characters found in order.
    """
    return EMOJI_PATTERN.findall(text)


def extract_hashtags(text: str) -> list[str]:
    """
    Find all hashtags in the text (including the # symbol).

    Example: "Love #python and #AI" -> ["#python", "#AI"]
    """
    return HASHTAG_PATTERN.findall(text)


def count_hashtags(hashtags: list[str]) -> int:
    """Return how many hashtags were found."""
    return len(hashtags)


def has_url(text: str) -> bool:
    """Return True if the text contains at least one URL."""
    return URL_PATTERN.search(text) is not None


def has_excessive_repetition(text: str) -> bool:
    """
    Return True if any character is repeated 4+ times in a row.

    Example: "sooooo good" -> True
    """
    return EXCESSIVE_REPETITION_PATTERN.search(text) is not None


def count_exclamation_marks(text: str) -> int:
    """Count how many '!' characters appear in the text."""
    return text.count("!")


def count_question_marks(text: str) -> int:
    """Count how many '?' characters appear in the text."""
    return text.count("?")


def clean_text(text: str) -> str:
    """
    Build a cleaned version of the comment for later analysis.

    - Normalizes whitespace
    - Collapses excessive character repetition down to two characters
      (e.g. "sooooo" -> "soo")

    Emojis and hashtags are kept. URLs are kept. This is not sentiment analysis.
    """
    cleaned = normalize_whitespace(text)
    cleaned = EXCESSIVE_REPETITION_PATTERN.sub(r"\1\1", cleaned)
    return cleaned


def empty_analysis_result(original_text: str = "") -> dict:
    """Return a default result dictionary for missing or empty comments."""
    return {
        "original_text": original_text,
        "cleaned_text": "",
        "emojis": [],
        "emoji_count": 0,
        "hashtags": [],
        "hashtag_count": 0,
        "has_url": False,
        "excessive_repetition": False,
        "exclamation_count": 0,
        "question_count": 0,
    }


def analyze_text(comment) -> dict:
    """
    Analyze a single comment and return structured preprocessing results.

    The original comment is preserved as-is (except None/NaN, which become "").
    A separate cleaned_text field is provided for downstream use.
    """
    if is_missing_value(comment):
        return empty_analysis_result()

    # Keep the original text unchanged for strings; convert other types safely.
    if isinstance(comment, str):
        original_text = comment
    else:
        original_text = str(comment)

    if original_text.strip() == "":
        return empty_analysis_result(original_text)

    emojis = extract_emojis(original_text)
    hashtags = extract_hashtags(original_text)

    return {
        "original_text": original_text,
        "cleaned_text": clean_text(original_text),
        "emojis": emojis,
        "emoji_count": len(emojis),
        "hashtags": hashtags,
        "hashtag_count": count_hashtags(hashtags),
        "has_url": has_url(original_text),
        "excessive_repetition": has_excessive_repetition(original_text),
        "exclamation_count": count_exclamation_marks(original_text),
        "question_count": count_question_marks(original_text),
    }
