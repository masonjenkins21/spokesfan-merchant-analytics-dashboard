import pandas as pd
from pathlib import Path



# File paths
INPUT_FILE = Path(
    "data/processed/reviews_with_roberta_sentiment.csv"
)

OUTPUT_DIR = Path(
    "data/processed/dashboard_metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# Load data
print("Loading sentiment dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Loaded {len(df)} reviews")



# Convert review dates and create month field
df["review_date"] = pd.to_datetime(
    df["review_date"],
    format="ISO8601",
    errors="coerce",
    utc=True
)



df["month"] = (
    df["review_date"]
    .dt.to_period("M")
    .astype(str)
)



# 1. Merchant Summary
print("\nCreating merchant summary...")


merchant_summary = (
    df.groupby("merchant")
    .agg(
        total_reviews=("review_description", "count"),
        positive_reviews=(
            "roberta_label",
            lambda x: (x == "positive").sum()
        ),
        neutral_reviews=(
            "roberta_label",
            lambda x: (x == "neutral").sum()
        ),
        negative_reviews=(
            "roberta_label",
            lambda x: (x == "negative").sum()
        ),
        average_rating=("rating", "mean"),
        average_confidence=(
            "roberta_confidence",
            "mean"
        )
    )
    .reset_index()
)


merchant_summary["positive_percent"] = (
    merchant_summary["positive_reviews"]
    /
    merchant_summary["total_reviews"]
    *
    100
)

merchant_summary["negative_percent"] = (
    merchant_summary["negative_reviews"]
    /
    merchant_summary["total_reviews"]
    *
    100
)


merchant_summary.to_csv(
    OUTPUT_DIR / "merchant_sentiment_summary.csv",
    index=False
)



# 2. Product Sentiment
print("Creating product summary...")


product_summary = (
    df.groupby(
        [
            "merchant",
            "product_name"
        ]
    )
    .agg(
        total_reviews=("review_description", "count"),
        average_rating=("rating", "mean"),
        average_confidence=(
            "roberta_confidence",
            "mean"
        ),
        positive_percent=(
            "roberta_label",
            lambda x:
            (x == "positive").mean() * 100
        ),
        negative_percent=(
            "roberta_label",
            lambda x:
            (x == "negative").mean() * 100
        )
    )
    .reset_index()
)


product_summary.to_csv(
    OUTPUT_DIR / "product_sentiment_summary.csv",
    index=False
)



# 3. Rating vs Sentiment
print("Creating rating sentiment analysis...")


rating_sentiment = (
    df.groupby("rating")
    .agg(
        total_reviews=("review_description", "count"),
        positive_percent=(
            "roberta_label",
            lambda x:
            (x == "positive").mean() * 100
        ),
        negative_percent=(
            "roberta_label",
            lambda x:
            (x == "negative").mean() * 100
        )
    )
    .reset_index()
)


rating_sentiment.to_csv(
    OUTPUT_DIR / "rating_sentiment_analysis.csv",
    index=False
)



# 4. Monthly Trends

print("Creating monthly trends...")

monthly_trends = (
    df.groupby(
        [
            "merchant",
            "month"
        ]
    )
    .agg(
        total_reviews=("review_description", "count"),
        positive_percent=(
            "roberta_label",
            lambda x:
            (x == "positive").mean() * 100
        ),
        negative_percent=(
            "roberta_label",
            lambda x:
            (x == "negative").mean() * 100
        )
    )
    .reset_index()
)


monthly_trends.to_csv(
    OUTPUT_DIR / "monthly_sentiment_trends.csv",
    index=False
)


print("\n========== COMPLETE ==========")

print(
    "Dashboard files saved to:"
)

print(
    OUTPUT_DIR
)