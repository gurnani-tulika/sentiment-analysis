"""
Sentiment analysis for social media comments using VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) works well on
short, informal text like social media comments, including emojis and slang.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create one analyzer instance and reuse it for every comment.
# This is faster than creating a new analyzer each time.
analyzer = SentimentIntensityAnalyzer()

# Classification thresholds for the compound score.
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


def is_missing_value(text) -> bool:
    """Return True when the value is None or a floating-point NaN."""
    if text is None:
        return True
    if isinstance(text, float) and text != text:
        return True
    return False


def empty_sentiment_result() -> dict:
    """Return a default result for missing or empty comments."""
    return {
        "sentiment_label": "NEUTRAL",
        "compound_score": 0.0,
        "positive_score": 0.0,
        "neutral_score": 0.0,
        "negative_score": 0.0,
    }


def classify_sentiment(compound_score: float) -> str:
    """
    Convert a compound score into a sentiment label.

    compound >= 0.5  -> POSITIVE
    compound <= -0.5 -> NEGATIVE
    otherwise        -> NEUTRAL
    """
    if compound_score >= POSITIVE_THRESHOLD:
        return "POSITIVE"
    if compound_score <= NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def analyze_sentiment(text) -> dict:
    """
    Analyze the sentiment of a single comment.

    Passes the original text to VADER unchanged so emojis, punctuation,
    capitalization, and other social-media signals are preserved.

    Handles None, NaN, empty strings, and non-string input safely.
    """
    if is_missing_value(text):
        return empty_sentiment_result()

    # Keep strings unchanged; convert other types (e.g. numbers) to text.
    if isinstance(text, str):
        original_text = text
    else:
        original_text = str(text)

    if original_text.strip() == "":
        return empty_sentiment_result()

    # VADER returns scores from -1 (most negative) to +1 (most positive).
    scores = analyzer.polarity_scores(original_text)

    compound_score = scores["compound"]

    return {
        "sentiment_label": classify_sentiment(compound_score),
        "compound_score": compound_score,
        "positive_score": scores["pos"],
        "neutral_score": scores["neu"],
        "negative_score": scores["neg"],
    }
