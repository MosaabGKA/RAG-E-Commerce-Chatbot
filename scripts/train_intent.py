"""Train the intent -> routing-category classifier on the Bitext customer-support data.

Uses the gold 'intent' column, condensed to the 7 routing categories from the
assignment, with TF-IDF (word + char n-grams) + a linear model.
"""
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BITEXT_DATASET, INTENT_TO_CATEGORY, INTENT_DIR
from src.preprocessing import normalize_text


def main():
    ds = load_dataset(BITEXT_DATASET)
    df = ds["train"].to_pandas()[["instruction", "intent"]].copy()
    df["category"] = df["intent"].map(INTENT_TO_CATEGORY)

    unmapped = sorted(df.loc[df["category"].isna(), "intent"].unique())
    if unmapped:
        raise SystemExit(f"Unmapped intents: {unmapped}")
    df["category"] = df["category"].astype(str)

    print(df["category"].value_counts())
    df["clean"] = df["instruction"].apply(lambda x: normalize_text(x, keep_chars=True))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])
    print(f"Train={len(train_df)}  Test={len(test_df)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)
    X_train = vectorizer.fit_transform(train_df["clean"])
    X_test = vectorizer.transform(test_df["clean"])
    y_train, y_test = train_df["category"], test_df["category"]

    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nAccuracy = {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    joblib.dump(vectorizer, INTENT_DIR / "vectorizer.joblib")
    joblib.dump(model, INTENT_DIR / "model.joblib")
    joblib.dump({"classes": list(model.classes_)}, INTENT_DIR / "meta.joblib")
    print("\nSaved artifacts to", INTENT_DIR)
    return cm


if __name__ == "__main__":
    main()