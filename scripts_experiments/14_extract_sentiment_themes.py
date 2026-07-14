import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)



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


OUTPUT_FILE = (
    OUTPUT_DIR /
    "negative_sentiment_themes.csv"
)



# Load data
print("Loading sentiment dataset...")


df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


# Create product identifier from URL
df["product_name"] = (
    df["product_url"]
    .str.extract(r"/products/([^/?]+)")
)


df["product_name"] = (
    df["product_name"]
    .str.replace("-", " ", regex=False)
    .str.title()
)


print(
    f"Loaded {len(df)} reviews"
)



# Filter negative reviews
negative_reviews = df[
    df["roberta_label"] == "negative"
].copy()


print(
    f"Negative reviews found: {len(negative_reviews)}"
)


negative_reviews = negative_reviews[
    negative_reviews["review_text"].notna()
]



# TF-IDF setup
custom_stop_words = list(
    ENGLISH_STOP_WORDS.union(
        {
            # Generic review words
            "great",
            "amazing",
            "excellent",
            "perfect",
            "beautiful",

            # Cosmetic nouns
            "skin",
            "colour",
            "color",
            "lip",
            "lips",
            "lipstick",
            "eyeliner",
            "pencil",
            "pencils",
            "cream",
            "serum",
            "mask",

            # Generic commerce
            "received",
            "receive",
            "receiving",
            "item",
            "items",
            "purchase",
            "purchased",
            "buy",
            "bought",
            "ordered",
            "order",

            # Generic adjectives
            "little",
            "small",
            "large",
            "pretty",
            "best"
        }
    )
)


def assign_theme(keyword):
    keyword = keyword.lower()

    if any(word in keyword for word in [
        "ship", "deliver", "order", "package", "receive"
    ]):
        return "Shipping & Delivery"

    if any(word in keyword for word in [
        "dry", "sticky", "greasy", "oily", "thick", "thin"
    ]):
        return "Texture"

    if any(word in keyword for word in [
        "shade", "colour", "color", "pigment"
    ]):
        return "Color Accuracy"

    if any(word in keyword for word in [
        "smudge", "fade", "lasting", "wear"
    ]):
        return "Wear Performance"

    if any(word in keyword for word in [
        "refund", "return", "support", "service"
    ]):
        return "Customer Service"

    if any(word in keyword for word in [
        "irritation", "burn", "itch", "rash", "allergic"
    ]):
        return "Skin Reaction"

    if any(word in keyword for word in [
        "acne", "breakout", "breakouts", "pimple", "blemish"
    ]):
        return "Skin Concerns"

    if any(word in keyword for word in [
        "work", "working", "results",
        "effective", "advertised"
    ]):
        return "Product Effectiveness"

    return None


# TF-IDF extraction function
def extract_theme_mentions(texts):

    texts = (
        texts
        .dropna()
        .astype(str)
        .tolist()
    )


    if len(texts) == 0:

        return pd.DataFrame(
            columns=[
                "theme",
                "mentions"
            ]
        )


    vectorizer = TfidfVectorizer(

        stop_words=custom_stop_words,

        max_features=500,

        ngram_range=(1, 2),

        min_df=1,

        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )


    try:

        tfidf_matrix = vectorizer.fit_transform(
            texts
        )


    except ValueError:

        return pd.DataFrame(
            columns=[
                "theme",
                "mentions"
            ]
        )


    keywords = pd.DataFrame(
        {
            "keyword":
                vectorizer.get_feature_names_out()
        }
    )

    keywords["theme"] = keywords["keyword"].apply(assign_theme)


    theme_scores = (
        keywords
        .dropna(subset=["theme"])
        .groupby("theme")
        .size()
        .reset_index(name="mentions")
        .sort_values("mentions", ascending=False)
    )


    return theme_scores




# Merchant themes
print("\nExtracting merchant themes...")


merchant_results = []


for merchant, group in (
    negative_reviews
    .groupby("merchant")
):

    print(
        f"Processing {merchant}"
    )

    themes = extract_theme_mentions(
        group["review_text"]
    )

    if not themes.empty:
        themes["merchant"] = merchant

        merchant_results.append(
            themes
        )



if merchant_results:

    merchant_theme_results = pd.concat(
        merchant_results,
        ignore_index=True
    )

    merchant_theme_results = merchant_theme_results[
        [
            "merchant",
            "theme",
            "mentions"
        ]
    ]


else:

    merchant_theme_results = pd.DataFrame(
        columns=[
            "merchant",
            "theme",
            "mentions"
        ]
    )




# Product themes
print("\nExtracting product themes...")


product_results = []


for (merchant, product), group in (
    negative_reviews
    .groupby(
        [
            "merchant",
            "product_name"
        ]
    )
):


    # Ignore products with too few negative reviews
    if len(group) < 5:

        continue

    themes = extract_theme_mentions(
        group["review_text"]
    )

    if not themes.empty:
        themes["merchant"] = merchant
        themes["product_name"] = product
        themes["negative_review_count"] = len(group)

        product_results.append(
            themes
        )



if product_results:

    product_theme_results = pd.concat(
        product_results,
        ignore_index=True
    )

    product_theme_results = (
        product_theme_results[
            [
                "merchant",
                "product_name",
                "theme",
                "mentions",
                "negative_review_count"
            ]
        ]
    )

    product_theme_results["theme_frequency_within_negative_reviews"] = (
            product_theme_results["mentions"]
            /
            product_theme_results["negative_review_count"]
    )


    product_theme_results.to_csv(
        OUTPUT_DIR /
        "product_negative_themes.csv",
        index=False
    )


else:

    print(
        "No products had enough negative reviews for theme extraction."
    )




# Save merchant themes
merchant_theme_results.to_csv(
    OUTPUT_FILE,
    index=False
)



print(
    "\n========== COMPLETE =========="
)


print(
    "Saved:"
)


print(
    OUTPUT_FILE
)


print(
    "\nPreview:"
)


print(
    merchant_theme_results.head(20)
)