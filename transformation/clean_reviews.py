"""
clean_reviews.py
----------------
Cleans and enriches the Olist order reviews dataset by:
  1. Parsing date columns to datetime
  2. Computing review response time
  3. Deduplicating orders with multiple reviews (keep most recent)
  4. Adding helper flags for text presence and sentiment bucket

Input files:
    data/raw/olist_order_reviews_dataset.csv

Output:
    data/processed/clean_reviews.parquet   (one row per order_id)

Usage:
    python transformation/clean_reviews.py
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load and parse dates
# ---------------------------------------------------------------------------

def load_reviews(path: Path) -> pd.DataFrame:
    log.info("Loading reviews from %s", path)
    reviews = pd.read_csv(
        path,
        dtype={
            "review_id": str,
            "order_id":  str,
        },
    )
    log.info("Raw shape: %s", reviews.shape)

    for col in ["review_creation_date", "review_answer_timestamp"]:
        reviews[col] = pd.to_datetime(reviews[col], errors="coerce")
        n_null = reviews[col].isnull().sum()
        if n_null > 0:
            log.warning("  %s → %d null values after parsing", col, n_null)

    return reviews


# ---------------------------------------------------------------------------
# Step 2 — Validate review scores
# ---------------------------------------------------------------------------

def validate_scores(reviews: pd.DataFrame) -> pd.DataFrame:
    invalid = ~reviews["review_score"].isin([1, 2, 3, 4, 5])
    if invalid.any():
        log.warning(
            "%d rows have invalid review_score values: %s",
            invalid.sum(),
            reviews.loc[invalid, "review_score"].unique().tolist(),
        )

    log.info(
        "Review score distribution:\n%s",
        reviews["review_score"].value_counts().sort_index().to_string(),
    )
    log.info(
        "Mean review score: %.3f", reviews["review_score"].mean()
    )

    return reviews


# ---------------------------------------------------------------------------
# Step 3 — Compute response time
# ---------------------------------------------------------------------------

def compute_response_time(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews["review_response_time_hours"] = (
        (reviews["review_answer_timestamp"] - reviews["review_creation_date"])
        .dt.total_seconds() / 3600
    ).round(2)

    # Negative response times = data issues, flag but keep
    n_negative = (reviews["review_response_time_hours"] < 0).sum()
    if n_negative > 0:
        log.warning(
            "%d rows have negative review_response_time_hours", n_negative
        )

    log.info(
        "Response time (hours) — median: %.1f | mean: %.1f | max: %.1f",
        reviews["review_response_time_hours"].median(),
        reviews["review_response_time_hours"].mean(),
        reviews["review_response_time_hours"].max(),
    )

    return reviews


# ---------------------------------------------------------------------------
# Step 4 — Add text and sentiment flags
# ---------------------------------------------------------------------------

def add_text_flags(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    has_comment_message: used to filter text-based NLP analysis
    sentiment_bucket: coarse label for quick aggregations in notebooks
    """
    reviews["review_comment_title"]   = reviews["review_comment_title"].str.strip()
    reviews["review_comment_message"] = reviews["review_comment_message"].str.strip()

    reviews["has_comment_title"]   = reviews["review_comment_title"].notna() & \
                                     (reviews["review_comment_title"] != "")
    reviews["has_comment_message"] = reviews["review_comment_message"].notna() & \
                                     (reviews["review_comment_message"] != "")

    log.info(
        "Reviews with comment message: %d (%.1f%%)",
        reviews["has_comment_message"].sum(),
        100 * reviews["has_comment_message"].mean(),
    )

    # Coarse sentiment bucket: negative (1-2), neutral (3), positive (4-5)
    reviews["sentiment_bucket"] = pd.cut(
        reviews["review_score"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        right=True,
    )

    log.info(
        "Sentiment bucket distribution:\n%s",
        reviews["sentiment_bucket"].value_counts().sort_index().to_string(),
    )

    return reviews


# ---------------------------------------------------------------------------
# Step 5 — Deduplicate: one review per order_id
# ---------------------------------------------------------------------------

def deduplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Some orders have multiple review rows. Keep the most recent by
    review_creation_date so each order_id maps to exactly one review.
    """
    n_before = len(reviews)
    n_dup_orders = reviews.duplicated(subset="order_id").sum()

    if n_dup_orders > 0:
        log.info(
            "%d duplicate order_id rows found — keeping most recent review",
            n_dup_orders,
        )
        reviews = (
            reviews.sort_values("review_creation_date", ascending=False)
            .drop_duplicates(subset="order_id", keep="first")
        )
        log.info("Dropped %d duplicate rows (%d → %d)", n_before - len(reviews), n_before, len(reviews))
    else:
        log.info("No duplicate order_ids — no deduplication needed")

    return reviews


# ---------------------------------------------------------------------------
# Step 6 — Finalize
# ---------------------------------------------------------------------------

def finalize(reviews: pd.DataFrame) -> pd.DataFrame:
    col_order = [
        "review_id", "order_id",
        "review_score", "sentiment_bucket",
        "review_creation_date", "review_answer_timestamp",
        "review_response_time_hours",
        "review_comment_title", "review_comment_message",
        "has_comment_title", "has_comment_message",
    ]
    reviews = reviews[col_order].reset_index(drop=True)
    log.info("Final reviews shape: %s", reviews.shape)

    log.info(
        "\n%s",
        reviews.describe(include="all").T[["count", "unique", "top"]].to_string(),
    )
    return reviews


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reviews = load_reviews(RAW_DIR / "olist_order_reviews_dataset.csv")
    reviews = validate_scores(reviews)
    reviews = compute_response_time(reviews)
    reviews = add_text_flags(reviews)
    reviews = deduplicate_reviews(reviews)
    reviews = finalize(reviews)

    out = PROCESSED_DIR / "clean_reviews.parquet"
    reviews.to_parquet(out, index=False)
    log.info("Saved clean_reviews → %s", out)


if __name__ == "__main__":
    main()