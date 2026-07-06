import joblib
from transformers import pipeline


def train_logreg(X, y):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def load_bert_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


def bert_score(pipeline, text):
    result = pipeline(text[:512])[0]
    return result["score"] if result["label"] == "POSITIVE" else -result["score"]