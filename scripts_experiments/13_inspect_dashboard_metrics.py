import pandas as pd
from pathlib import Path


DATA_DIR = Path(
    "data/processed/dashboard_metrics"
)


files = [
    "merchant_sentiment_summary.csv",
    "product_sentiment_summary.csv",
    "rating_sentiment_analysis.csv",
    "monthly_sentiment_trends.csv"
]


for file in files:

    print("\n" + "=" * 50)
    print(file)
    print("=" * 50)

    df = pd.read_csv(DATA_DIR / file)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nPreview:")
    print(df.head())