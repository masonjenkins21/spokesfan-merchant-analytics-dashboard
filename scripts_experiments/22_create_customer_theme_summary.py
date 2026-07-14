"""
Script 22 - Create Customer Theme Summary

Purpose:
    Aggregate product-level negative themes into
    customer insight metrics for Power BI.

Creates:
    - customer_theme_summary.csv

Metrics:
    - total_theme_mentions
    - products_affected
    - reviews_affected
    - average_customer_impact
    - risk_level
    - theme_rank
"""


from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

dashboard_path = (
    project_root
    / "data"
    / "processed"
    / "dashboard_metrics"
)

input_file = (
    dashboard_path
    / "product_negative_themes.csv"
)

output_file = (
    dashboard_path
    / "customer_theme_summary.csv"
)


print("\nCreating customer theme summary...\n")


# ---------------------------------------------------------
# Load product theme metrics
# ---------------------------------------------------------

print("Loading product negative themes...")

df = pd.read_csv(
    input_file
)

print(f"Loaded {len(df)} theme records")


print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# Aggregate themes
# ---------------------------------------------------------

print("\nAggregating themes...")


theme_summary = (
    df.groupby("theme")
    .agg(
        total_theme_mentions=(
            "mentions",
            "sum"
        ),

        products_affected=(
            "product_name",
            "nunique"
        ),

        negative_reviews_affected=(
            "negative_review_count",
            "sum"
        ),

        average_customer_impact=(
            "overall_customer_impact",
            "mean"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# Create risk categories
# ---------------------------------------------------------

print("\nCreating risk levels...")


def assign_risk(score):

    if score >= 0.05:
        return "High"

    elif score >= 0.02:
        return "Medium"

    else:
        return "Low"


theme_summary["risk_level"] = (
    theme_summary[
        "average_customer_impact"
    ]
    .apply(assign_risk)
)


# ---------------------------------------------------------
# Rank themes
# ---------------------------------------------------------

theme_summary["theme_rank"] = (
    theme_summary[
        "total_theme_mentions"
    ]
    .rank(
        ascending=False,
        method="dense"
    )
    .astype(int)
)


# ---------------------------------------------------------
# Format output
# ---------------------------------------------------------

theme_summary = (
    theme_summary
    .sort_values(
        "theme_rank"
    )
)


theme_summary[
    "average_customer_impact"
] = (
    theme_summary[
        "average_customer_impact"
    ]
    .round(4)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving customer theme summary...")


theme_summary.to_csv(
    output_file,
    index=False
)


print("\n========== COMPLETE ==========")

print(
    f"Saved:\n{output_file}"
)


print("\nPreview:")

print(
    theme_summary.head(10)
)