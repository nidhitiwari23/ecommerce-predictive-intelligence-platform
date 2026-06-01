"""
models/sentiment_analyzer.py
=============================
Sentiment Analysis on Customer Reviews using TextBlob + NLTK.

What is Sentiment Analysis?
  Sentiment analysis reads a piece of text and determines whether
  the emotion/opinion expressed is POSITIVE, NEGATIVE, or NEUTRAL.

  Example:
    "Great product, arrived early!"        → Positive (score: +0.78)
    "Product was okay but delivery slow"   → Slightly Negative (score: -0.12)
    "DO NOT BUY — completely broken"       → Very Negative (score: -0.90)
    "Received the item on Tuesday"         → Neutral (score: 0.00)

Why do we need this for e-commerce?
  1. The 5-star review score is a number but the TEXT tells us WHY.
     A customer who writes "terrible quality, will never buy again"
     needs immediate attention — even if they gave 2 stars instead of 1.

  2. Sentiment scores become FEATURES for our churn model.
     Customers who write very negative reviews are significantly more
     likely to churn. Adding sentiment polarity to the churn model
     improves its accuracy.

  3. Topic detection tells the business WHAT customers complain about most —
     is it delivery? product quality? customer service? packaging?

How TextBlob works:
  TextBlob uses a pre-built dictionary of words with known sentiment scores.
  It analyses each word in the text, combines their scores, and returns:

  - polarity:     -1.0 (very negative) to +1.0 (very positive)
  - subjectivity:  0.0 (factual/objective) to 1.0 (personal/emotional)

  "The delivery was extremely slow and the box was damaged."
   → polarity: -0.45, subjectivity: 0.65

TextBlob works WITHOUT any training data — it uses the built-in lexicon.
This makes it ideal for quick deployment with no labelled review data needed.
"""

import logging
import re
import warnings
from typing import Dict, List, Optional
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Try importing TextBlob — graceful fallback if not installed
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not installed. Run: pip install textblob")

# Try importing NLTK for text preprocessing
try:
    import nltk
    # Download required NLTK data files (only needed once)
    for resource in ["punkt", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words("portuguese") + stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = set()
    logger.warning("NLTK not installed. Run: pip install nltk")


class SentimentAnalyzer:
    """
    Analyses customer review text and extracts sentiment features.

    Outputs per review:
      - polarity_score      : -1.0 to +1.0 (negative to positive)
      - subjectivity_score  : 0.0 to 1.0 (objective to subjective)
      - sentiment_label     : 'Positive' / 'Neutral' / 'Negative'
      - review_length       : number of characters in the review text
      - word_count          : number of words
      - has_exclamation     : 1 if review contains ! (strong emotion)
      - has_caps            : 1 if review has CAPITALS (shouting / anger)
      - complaint_keywords  : number of complaint words found
      - praise_keywords     : number of positive words found

    All these become additional features in the churn prediction model.
    """

    # Common complaint words in Portuguese (Olist dataset) and English
    COMPLAINT_WORDS = [
        # Portuguese
        "ruim", "horrível", "péssimo", "quebrado", "atrasado", "errado",
        "defeituoso", "danificado", "fraude", "enganação", "nunca", "cancelar",
        "reembolso", "decepcionado", "frustrado", "problema", "reclamação",
        # English
        "terrible", "awful", "broken", "damaged", "late", "wrong", "defective",
        "fraud", "scam", "never", "cancel", "refund", "disappointed", "frustrated",
        "horrible", "poor", "worst", "bad", "useless", "fake", "cheap", "ugly"
    ]

    # Common praise words in Portuguese and English
    PRAISE_WORDS = [
        # Portuguese
        "ótimo", "excelente", "perfeito", "maravilhoso", "rápido", "lindo",
        "adorei", "amei", "recomendo", "satisfeito", "feliz", "qualidade",
        # English
        "excellent", "perfect", "amazing", "wonderful", "fast", "great",
        "love", "recommend", "satisfied", "happy", "quality", "best",
        "beautiful", "awesome", "fantastic", "superb", "outstanding"
    ]

    def __init__(self):
        self._summary_stats: Optional[Dict] = None

    # ── Public methods ─────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame,
                text_column: str = "review_comment_message") -> pd.DataFrame:
        """
        Run sentiment analysis on all reviews and add feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Master feature table containing review text column.
        text_column : str
            Name of the column containing review text strings.

        Returns
        -------
        pd.DataFrame
            DataFrame with all original columns PLUS sentiment feature columns.
        """
        if text_column not in df.columns:
            logger.warning(f"  Column '{text_column}' not found. "
                           f"Returning DataFrame with default sentiment scores.")
            return self._add_default_sentiment(df)

        logger.info(f"  Analysing sentiment for {len(df):,} reviews...")

        # Fill missing reviews with empty string (some orders have no review text)
        texts = df[text_column].fillna("").astype(str)

        # Analyse each review
        results = [self._analyse_single(text) for text in texts]
        sentiment_df = pd.DataFrame(results)

        # Add all sentiment columns to the original DataFrame
        result = df.copy()
        for col in sentiment_df.columns:
            result[col] = sentiment_df[col].values

        # Calculate and log summary statistics
        self._summary_stats = self.get_summary(result)
        logger.info(f"  Average polarity:     {self._summary_stats['avg_polarity']:.3f}")
        logger.info(f"  Positive reviews:     {self._summary_stats['pct_positive']:.1%}")
        logger.info(f"  Negative reviews:     {self._summary_stats['pct_negative']:.1%}")
        logger.info(f"  Reviews with text:    {self._summary_stats['pct_with_text']:.1%}")

        return result

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Return summary statistics of the sentiment analysis results.
        Used for the dashboard and MLflow logging.
        """
        if "polarity_score" not in df.columns:
            return {"avg_polarity": 0.0, "pct_positive": 0.0,
                    "pct_negative": 0.0, "pct_with_text": 0.0}

        total = len(df)
        with_text = (df.get("review_length", pd.Series([0]*total)) > 0).sum()

        return {
            "avg_polarity":   round(float(df["polarity_score"].mean()), 4),
            "avg_subjectivity": round(float(df["subjectivity_score"].mean()), 4),
            "pct_positive":   round((df["sentiment_label"] == "Positive").sum() / total, 4),
            "pct_neutral":    round((df["sentiment_label"] == "Neutral").sum() / total, 4),
            "pct_negative":   round((df["sentiment_label"] == "Negative").sum() / total, 4),
            "pct_with_text":  round(with_text / total, 4),
            "avg_word_count": round(float(df.get("word_count", pd.Series([0])).mean()), 1),
            "total_reviews":  total
        }

    def get_top_complaints(self, df: pd.DataFrame,
                           text_column: str = "review_comment_message",
                           top_n: int = 10) -> pd.DataFrame:
        """
        Find the most common complaint keywords across negative reviews.

        This tells the business WHAT customers are unhappy about —
        is it delivery? product quality? packaging? customer service?

        Returns
        -------
        pd.DataFrame
            Top N complaint keywords with their frequency count.
        """
        if text_column not in df.columns or "sentiment_label" not in df.columns:
            return pd.DataFrame(columns=["keyword", "count"])

        # Only look at negative reviews
        negative_reviews = df[df["sentiment_label"] == "Negative"][text_column]
        negative_reviews = negative_reviews.fillna("").astype(str)

        # Count complaint keyword occurrences
        keyword_counts = {}
        for text in negative_reviews:
            text_lower = text.lower()
            for word in self.COMPLAINT_WORDS:
                if word in text_lower:
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1

        if not keyword_counts:
            return pd.DataFrame(columns=["keyword", "count"])

        result = pd.DataFrame(
            list(keyword_counts.items()),
            columns=["keyword", "count"]
        ).sort_values("count", ascending=False).head(top_n).reset_index(drop=True)

        return result

    def get_churn_sentiment_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Calculate how strongly sentiment correlates with churn.

        This is the key insight: customers who write negative reviews
        are much more likely to churn. Quantifying this helps justify
        including sentiment as a feature in the churn model.

        Returns
        -------
        dict
            Average polarity for churned vs. non-churned customers,
            and the correlation coefficient.
        """
        if "churn" not in df.columns or "polarity_score" not in df.columns:
            return {}

        churned     = df[df["churn"] == 1]["polarity_score"]
        not_churned = df[df["churn"] == 0]["polarity_score"]

        correlation = df["polarity_score"].corr(df["churn"])

        return {
            "avg_polarity_churned":     round(float(churned.mean()), 4),
            "avg_polarity_not_churned": round(float(not_churned.mean()), 4),
            "polarity_churn_correlation": round(float(correlation), 4),
            "insight": (
                "Customers who churn write significantly more negative reviews "
                f"(avg polarity {churned.mean():.2f}) compared to loyal customers "
                f"(avg polarity {not_churned.mean():.2f})"
            )
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _analyse_single(self, text: str) -> Dict:
        """
        Analyse a single review text string and return all features.

        This is the core function called for every review.
        Returns a dictionary that becomes one row in the results DataFrame.
        """
        # Handle empty or very short reviews
        if not text or len(text.strip()) < 3:
            return self._default_sentiment_row()

        # ── TextBlob sentiment scoring ──────────────────────────────────
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity      = float(blob.sentiment.polarity)
                subjectivity  = float(blob.sentiment.subjectivity)
            except Exception:
                polarity, subjectivity = 0.0, 0.0
        else:
            # Fallback: simple keyword-based scoring if TextBlob not available
            polarity, subjectivity = self._keyword_sentiment(text)

        # ── Sentiment label ─────────────────────────────────────────────
        # We use 0.05 and -0.05 as thresholds rather than 0.0
        # because very small values are essentially neutral
        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        # ── Text features ───────────────────────────────────────────────
        text_clean    = text.strip()
        review_length = len(text_clean)
        word_count    = len(text_clean.split())

        # Detect emotional indicators
        has_exclamation = int("!" in text_clean)
        # Has CAPS: any word fully in uppercase (3+ letters) indicates shouting
        has_caps = int(bool(re.search(r'\b[A-Z]{3,}\b', text_clean)))

        # Count complaint and praise keywords
        text_lower = text_clean.lower()
        complaint_count = sum(1 for w in self.COMPLAINT_WORDS if w in text_lower)
        praise_count    = sum(1 for w in self.PRAISE_WORDS    if w in text_lower)

        return {
            "polarity_score":      round(polarity, 4),
            "subjectivity_score":  round(subjectivity, 4),
            "sentiment_label":     label,
            "review_length":       review_length,
            "word_count":          word_count,
            "has_exclamation":     has_exclamation,
            "has_caps":            has_caps,
            "complaint_keywords":  complaint_count,
            "praise_keywords":     praise_count,
            # Composite score: combines polarity with keyword signals
            # More negative complaint keywords = even lower score
            "composite_sentiment": round(
                polarity - (complaint_count * 0.1) + (praise_count * 0.05), 4
            )
        }

    def _keyword_sentiment(self, text: str) -> Tuple:
        """
        Simple fallback sentiment scoring using keyword counting.
        Used when TextBlob is not installed.

        Returns (polarity, subjectivity) tuple.
        """
        text_lower = text.lower()
        praise_count    = sum(1 for w in self.PRAISE_WORDS    if w in text_lower)
        complaint_count = sum(1 for w in self.COMPLAINT_WORDS if w in text_lower)

        total = praise_count + complaint_count
        if total == 0:
            return 0.0, 0.0

        polarity     = (praise_count - complaint_count) / total
        subjectivity = min(total / 5.0, 1.0)
        return round(polarity, 4), round(subjectivity, 4)

    def _default_sentiment_row(self) -> Dict:
        """Return zero/neutral values for empty or very short reviews."""
        return {
            "polarity_score":      0.0,
            "subjectivity_score":  0.0,
            "sentiment_label":     "Neutral",
            "review_length":       0,
            "word_count":          0,
            "has_exclamation":     0,
            "has_caps":            0,
            "complaint_keywords":  0,
            "praise_keywords":     0,
            "composite_sentiment": 0.0
        }

    def _add_default_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all sentiment columns with default values when no text available."""
        result = df.copy()
        defaults = self._default_sentiment_row()
        for col, val in defaults.items():
            result[col] = val
        return result


# ── Type hint fix ──────────────────────────────────────────────────────────────
from typing import Tuple  # noqa: E402 — needed for _keyword_sentiment return type
