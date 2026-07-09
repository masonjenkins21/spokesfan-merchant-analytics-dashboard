import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)


# -----------------------------
# File paths
# -----------------------------

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
    "negative_sentiment_keywords.csv"
)


# -----------------------------
# Load data
# -----------------------------

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


# -----------------------------
# Filter negative reviews
# -----------------------------

negative_reviews = df[
    df["roberta_label"] == "negative"
].copy()


print(
    f"Negative reviews found: {len(negative_reviews)}"
)


negative_reviews = negative_reviews[
    negative_reviews["review_text"].notna()
]


# -----------------------------
# TF-IDF setup
# -----------------------------

custom_stop_words = list(
    ENGLISH_STOP_WORDS.union(
        {

            # Common filler words
            "like",
            "love",
            "product",
            "products",
            "use",
            "used",
            "using",
            "really",
            "good",
            "nice",
            "just",
            "thing",
            "things",
            "get",
            "got",
            "make",
            "made",
            "also",
            "one",
            "two",
            "much",
            "very",

            # Contraction fragments
            "didn",
            "doesn",
            "don",
            "isn",
            "wasn",
            "weren",
            "haven",
            "hasn",
            "hadn",
            "couldn",
            "wouldn",
            "shouldn",
            "ive",
            "im",
            "dont",
            "cant",

            # Common review language
            "would",
            "could",
            "want",
            "feel",
            "felt",

            # Remaining low-value fragments
            "did",
            "does",
            "bit"

        }
    )
)


# -----------------------------
# TF-IDF extraction function
# -----------------------------

def extract_keywords(texts, top_n=20):

    texts = (
        texts
        .dropna()
        .astype(str)
        .tolist()
    )


    if len(texts) == 0:

        return pd.DataFrame(
            columns=[
                "keyword",
                "score"
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
                "keyword",
                "score"
            ]
        )


    scores = (
        tfidf_matrix
        .sum(axis=0)
        .A1
    )


    keywords = pd.DataFrame(
        {
            "keyword":
                vectorizer.get_feature_names_out(),

            "score":
                scores
        }
    )


    keywords = keywords.sort_values(
        "score",
        ascending=False
    )


    return keywords.head(top_n)



# -----------------------------
# Merchant themes
# -----------------------------

print("\nExtracting merchant themes...")


merchant_results = []


for merchant, group in (
    negative_reviews
    .groupby("merchant")
):

    print(
        f"Processing {merchant}"
    )


    keywords = extract_keywords(
        group["review_text"]
    )


    if not keywords.empty:

        keywords["merchant"] = merchant

        merchant_results.append(
            keywords
        )



if merchant_results:

    merchant_theme_results = pd.concat(
        merchant_results,
        ignore_index=True
    )


    merchant_theme_results = (
        merchant_theme_results[
            [
                "merchant",
                "keyword",
                "score"
            ]
        ]
    )


else:

    merchant_theme_results = pd.DataFrame(
        columns=[
            "merchant",
            "keyword",
            "score"
        ]
    )



# -----------------------------
# Product themes
# -----------------------------

print("\nExtracting product themes...")


product_results = []


for product, group in (
    negative_reviews
    .groupby("product_name")
):


    # Ignore products with too few negative reviews

    if len(group) < 5:

        continue



    keywords = extract_keywords(
        group["review_text"]
    )


    if not keywords.empty:

        keywords["product_name"] = product

        product_results.append(
            keywords
        )



if product_results:

    product_theme_results = pd.concat(
        product_results,
        ignore_index=True
    )


    product_theme_results = (
        product_theme_results[
            [
                "product_name",
                "keyword",
                "score"
            ]
        ]
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



# -----------------------------
# Save merchant themes
# -----------------------------

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