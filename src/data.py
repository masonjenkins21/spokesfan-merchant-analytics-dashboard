from pathlib import Path
import pandas as pd


# Load data
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "raw"


def load_data():
    products = pd.read_csv(data_dir / "cheekbonebeauty_products_export.csv")
    reviews = pd.read_csv(
        data_dir / "cheekbonebeauty.com_YOTPO_all_product_reviews.csv"
    )

    reviews["review_description"] = reviews["review_description"].fillna("")

    return products, reviews