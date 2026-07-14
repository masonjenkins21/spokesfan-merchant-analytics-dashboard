"""
Script 21 - Create Dashboard Metrics

Purpose:
    Create BI-ready dashboard metrics from sentiment analysis outputs.

Creates:
    - product_negative_themes.csv with customer impact metrics

Metrics added:
    - negative_review_count
    - theme_prevalence_in_negative_reviews
    - total_reviews
    - overall_customer_impact
    - impact_rank
    - customer_risk_score
"""


from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

processed_path = project_root / "data" / "processed"

dashboard_path = processed_path / "dashboard_metrics"

sentiment_file = (
    processed_path /
    "reviews_with_roberta_sentiment.csv"
)

product_theme_file = dashboard_path / "product_negative_themes.csv"


print("\nCreating dashboard metrics...\n")


# ---------------------------------------------------------
# Load sentiment dataset
# ---------------------------------------------------------

print("Loading sentiment dataset...")

df = pd.read_csv(
    sentiment_file,
    low_memory=False
)

print(f"Loaded {len(df)} reviews")


print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# Load existing product themes
# ---------------------------------------------------------

print("\nLoading product negative themes...")

product_negative_themes = pd.read_csv(
    product_theme_file
)

print("\nExisting theme columns:")
print(product_negative_themes.columns.tolist())


print(
    f"Loaded {len(product_negative_themes)} theme records"
)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nRoBERTa sentiment distribution:")
print(df["roberta_label"].value_counts())


# ---------------------------------------------------------
# Calculate negative review counts per product
# ---------------------------------------------------------

print("\nCalculating negative review counts...")


negative_reviews = (
    df[
        df["roberta_label"]
        .str.lower()
        .eq("negative")
    ]
    .groupby(
        [
            "merchant",
            "product_name"
        ]
    )
    .size()
    .reset_index(
        name="negative_review_count"
    )
)


# ---------------------------------------------------------
# Calculate total reviews per product
# ---------------------------------------------------------

print("Calculating total review counts...")


total_reviews = (
    df.groupby(
        [
            "merchant",
            "product_name"
        ]
    )
    .size()
    .reset_index(
        name="total_reviews"
    )
)


# ---------------------------------------------------------
# Merge metrics
# ---------------------------------------------------------

print("\nMerging metrics...")


# Remove old calculated columns if rerunning script
product_negative_themes = product_negative_themes.drop(
    columns=[
        "negative_review_count",
        "theme_severity",
        "total_reviews",
        "theme_frequency_within_negative_reviews",
        "overall_customer_impact",
        "impact_rank"
    ],
    errors="ignore"
)


product_negative_themes = (
    product_negative_themes
    .merge(
        negative_reviews,
        on=[
            "merchant",
            "product_name"
        ],
        how="left"
    )
    .merge(
        total_reviews,
        on=[
            "merchant",
            "product_name"
        ],
        how="left"
    )
)

product_negative_themes[
    [
        "negative_review_count",
        "total_reviews"
    ]
] = (
    product_negative_themes[
        [
            "negative_review_count",
            "total_reviews"
        ]
    ]
    .fillna(0)
    .astype(int)
)


print("\nColumns after merge:")
print(product_negative_themes.columns.tolist())


# ---------------------------------------------------------
# Create BI metrics
# ---------------------------------------------------------

print("Creating BI metrics...")


# Frequency of theme mentions within negative reviews
product_negative_themes[
    "theme_prevalence_in_negative_reviews"
] = (
    product_negative_themes["mentions"]
    /
    product_negative_themes["negative_review_count"]
    .replace(0, 1)
)

# Overall customer impact
product_negative_themes[
    "overall_customer_impact"
] = (
    product_negative_themes["mentions"]
    /
    product_negative_themes["total_reviews"]
    .replace(0, 1)
)

product_negative_themes[
    [
        "theme_prevalence_in_negative_reviews",
        "overall_customer_impact"
    ]
] = (
    product_negative_themes[
        [
            "theme_prevalence_in_negative_reviews",
            "overall_customer_impact"
        ]
    ]
    .round(4)
)


# Rank biggest customer impact issues by merchant
product_negative_themes[
    "impact_rank"
] = (
    product_negative_themes
    .groupby("merchant")
    ["overall_customer_impact"]
    .rank(
        ascending=False,
        method="dense"
    )
)


# Customer risk score
product_negative_themes[
    "customer_risk_score"
] = (
    product_negative_themes[
        "overall_customer_impact"
    ]
    *
    product_negative_themes[
        "impact_rank"
    ]
)


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

product_negative_themes = (
    product_negative_themes
    .sort_values(
        [
            "merchant",
            "impact_rank"
        ]
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving updated product themes...")


product_negative_themes.to_csv(
    product_theme_file,
    index=False
)


print("\n========== COMPLETE ==========")

print(
    f"Saved:\n{product_theme_file}"
)


print("\nPreview:")

print(
    product_negative_themes.head(20)
)