
---

## Methodology

### 1. Data Preprocessing
- Lowercasing
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization

Output:
- `clean_text` column used for all models

---

### 2. Feature Engineering
Two representations were tested:

- **Bag of Words (CountVectorizer)**
- **TF-IDF (final model input)**

---

### 3. Models

#### A. Logistic Regression (TF-IDF)
- Multiclass classification (ratings 1–5)
- Class-weight balanced to handle imbalance
- Primary ML model used in evaluation

#### B. VADER (Rule-based model)
- Lexicon-based sentiment scoring
- Outputs compound score (-1 to +1)

#### C. DistilBERT (Transformer model)
- Pretrained sentiment classifier
- Converts text into contextual sentiment scores

---

## Evaluation Strategy

The main evaluation metric:

- **Agreement Rate**
  - Percentage where predicted rating == actual star rating

Additional evaluation:

- Classification report (precision, recall, F1)
- Error analysis (misclassified reviews)
- Sentiment distribution comparison

---

## Key Results

### Logistic Regression Performance
- **Accuracy / Agreement Rate:** ~86%

### Class-wise Performance
- Strong performance on **5-star reviews**
- Weakest performance on **2–4 star ambiguity classes**
- Lower performance reflects class imbalance and subjective reviews

### Sentiment Distribution (BERT-based)
- Majority positive (~80%+)
- Small portion negative (~15–20%)
- Very few neutral classifications

---

## Key Insights

- TF-IDF + Logistic Regression performs well for clear sentiment (1-star vs 5-star)
- Performance drops significantly for mid-range ratings (2–4 stars)
- VADER performs well on short, emotional reviews but struggles with nuance
- BERT provides the most context-aware sentiment but is computationally heavier

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
