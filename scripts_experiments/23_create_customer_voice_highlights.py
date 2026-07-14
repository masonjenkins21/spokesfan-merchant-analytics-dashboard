"""
Script 23 - Create Customer Voice Highlights

Purpose:
    Create representative customer feedback examples
    for Power BI customer insights dashboards.

Creates:
    - customer_voice_highlights.csv

Includes:
    - product
    - sentiment
    - theme
    - customer rating
    - customer feedback
    - sentiment confidence
    - feedback priority
"""


from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

processed_path = project_root / "data" / "processed"

dashboard_path = processed_path / "data" / "processed" / "dashboard_metrics"

sentiment_file = (
    processed_path /
    "reviews_with_roberta_sentiment.csv"
)

theme_file = (
    processed_path /
    "dashboard_metrics" /
    "product_negative_themes.csv"
)

output_file = (
    processed_path /
    "dashboard_metrics" /
    "customer_voice_highlights.csv"
)


print("\nCreating customer voice highlights...\n")


# ---------------------------------------------------------
# Load sentiment data
# ---------------------------------------------------------

print("Loading sentiment data...")

df = pd.read_csv(
    sentiment_file,
    low_memory=False
)

print(f"Loaded {len(df)} reviews")


# ---------------------------------------------------------
# Remove non-product items
# ---------------------------------------------------------

print("\nRemoving non-product items...")


exclude_products = [
    "Gift Card",
    "Gift Box",
    "Mug",
    "T-Shirt"
]


df = df[
    ~df["product_name"]
    .str.contains(
        "|".join(exclude_products),
        case=False,
        na=False
    )
]


print(
    f"Remaining reviews after product cleanup: {len(df)}"
)


# ---------------------------------------------------------
# Remove neutral and empty reviews
# ---------------------------------------------------------

print("\nRemoving neutral and empty reviews...")


df = df[
    df["roberta_label"]
    .isin(
        [
            "negative",
            "positive"
        ]
    )
]


df = df[
    df["review_text"]
    .notna()
]


df = df[
    df["review_text"]
    .str.strip()
    .ne("")
]


# ---------------------------------------------------------
# Load themes
# ---------------------------------------------------------

print("\nLoading themes...")


themes = pd.read_csv(
    theme_file
)


# ---------------------------------------------------------
# Assign themes correctly
# ---------------------------------------------------------

print("\nAssigning themes...")


df = df.merge(
    themes[
        [
            "merchant",
            "product_name",
            "theme"
        ]
    ],
    on=[
        "merchant",
        "product_name"
    ],
    how="left"
)


df["theme"] = (
    df["theme"]
    .fillna(
        "General Feedback"
    )
)


# ---------------------------------------------------------
# Rank reviews
# ---------------------------------------------------------

print("\nRanking reviews...")


df["sentiment_priority"] = (
    df["roberta_label"]
    .map(
        {
            "negative": 1,
            "positive": 2
        }
    )
)


df = df.sort_values(
    [
        "merchant",
        "product_name",
        "sentiment_priority",
        "roberta_confidence"
    ],
    ascending=[
        True,
        True,
        True,
        False
    ]
)


# ---------------------------------------------------------
# Select representative feedback
# ---------------------------------------------------------

print("\nSelecting representative feedback...")


highlights = (
    df
    .groupby(
        [
            "merchant",
            "product_name",
            "roberta_label"
        ]
    )
    .head(3)
)


# Remove duplicate reviews

highlights = (
    highlights
    .drop_duplicates(
        subset=[
            "merchant",
            "product_name",
            "review_text"
        ]
    )
)


# ---------------------------------------------------------
# Rename columns
# ---------------------------------------------------------

# Remove original sentiment column to avoid duplicate names
highlights = highlights.drop(
    columns=[
        "sentiment"
    ],
    errors="ignore"
)


# Rename model output columns
highlights = highlights.rename(
    columns={
        "roberta_label": "sentiment",
        "rating": "customer_star_rating",
        "review_text": "customer_feedback",
        "roberta_confidence": "sentiment_confidence"
    }
)


# ---------------------------------------------------------
# Add dashboard priority
# ---------------------------------------------------------

print("\nAdding feedback priority...")


highlights["feedback_priority"] = (
    highlights["sentiment"]
    .map(
        {
            "negative": "High",
            "positive": "Low"
        }
    )
)


# ---------------------------------------------------------
# Final columns
# ---------------------------------------------------------

highlights = highlights[
    [
        "merchant",
        "product_name",
        "sentiment",
        "customer_star_rating",
        "customer_feedback",
        "sentiment_confidence",
        "feedback_priority"
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving customer voice highlights...")


highlights.to_csv(
    output_file,
    index=False
)


print("\n========== COMPLETE ==========")

print(
    f"Saved:\n{output_file}"
)


print("\nPreview:")

print(
    highlights.head(20)
)