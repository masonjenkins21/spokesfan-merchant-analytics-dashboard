from src.data import load_data
from src.preprocess import apply_preprocessing
from src.models import load_model, load_bert_model, bert_score
from sklearn.metrics import classification_report

from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
import joblib



# Load data
project_root = Path(__file__).resolve().parent
_, reviews = load_data()

reviews = apply_preprocessing(reviews)



# Load trained TF-IDF + Logistic Regression

tfidf = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
model = load_model(project_root / "models" / "logreg_model.pkl")
X = tfidf.transform(reviews["clean_text"])

reviews["ml_pred_class"] = model.predict(X)



# VADER
sia = SentimentIntensityAnalyzer()

reviews["vader_score"] = reviews["clean_text"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)



# DISTILBERT
bert_pipe = load_bert_model()

sample = reviews.sample(
    500,
    random_state=42
).copy()

sample["bert_score"] = sample["clean_text"].apply(
    lambda x: bert_score(bert_pipe, x)
)



# Final dataset
final_df = sample.copy()

final_df["final_sentiment"] = final_df["bert_score"].apply(
    lambda x:
        "positive"
        if x >= 0.3
        else "negative"
        if x <= -0.3
        else "neutral"
)



# Agreement
agreement = (
    reviews["rating"] ==
    reviews["ml_pred_class"]
).mean()

print("\n--- MODEL AGREEMENT RATE ---")
print(f"TF-IDF vs Actual Rating Agreement: {agreement:.2%}")



# Error analysis
errors = reviews[
    reviews["rating"] != reviews["ml_pred_class"]
]

print("\n--- ERROR SAMPLE ---")

print(errors[[
    "review_description",
    "clean_text",
    "rating",
    "ml_pred_class"
]].head(10))



# OUTPUTS
print("\n--- FINAL DATAFRAME PREVIEW ---")
print(final_df.head())

print("\n--- DATAFRAME INFO ---")
print(final_df.info())

print("\n--- FINAL SENTIMENT DISTRIBUTION ---")
print(final_df["final_sentiment"].value_counts())

print("\n--- SAMPLE COMPARISON ---")
print(final_df[[
    "rating",
    "ml_pred_class",
    "bert_score",
    "vader_score",
    "final_sentiment"
]].head(10))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    reviews["rating"],
    reviews["ml_pred_class"]
))


# OPTIONAL EXPORT
# final_df.to_csv(
#     "data/processed/sentiment_output.csv",
#     index=False
# )