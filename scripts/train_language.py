"""Train a language-identification classifier (char n-gram TF-IDF + LinearSVC).

Data : papluca/language-identification  (90k samples / 20 languages / pre-split)
Enhancements over a plain BoW baseline:
  * character n-grams (2..5) which capture language-specific morphology/scripts
    far better than raw words for short, noisy messages
  * a confidence threshold -> low-confidence predictions are flagged as "unknown"
  * per-language evaluation (accuracy + macro-F1 + confusion matrix)
"""
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LANG_DATASET, LANG_DIR, LANG_CONFIDENCE_THRESHOLD
from src.preprocessing import normalize_text

tqdm.pandas()


def load_lang_data():
    ds = load_dataset(LANG_DATASET)  # train / validation / test
    return ds


def build_df(split):
    df = pd.DataFrame({"text": split["text"], "label": split["labels"]})
    df["clean"] = df["text"].apply(lambda x: normalize_text(x, keep_chars=True))
    return df


def main():
    print("Loading dataset ...")
    ds = load_lang_data()
    train_df = build_df(ds["train"])
    val_df = build_df(ds["validation"])
    test_df = build_df(ds["test"])
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}  languages={train_df['label'].nunique()}")

    # -- feature engineering: char n-grams 2..5 ---------------------------------
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["clean"])
    X_val = vectorizer.transform(val_df["clean"])
    X_test = vectorizer.transform(test_df["clean"])
    y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]

    print("Training LinearSVC ...")
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"\n[{name}] accuracy = {acc:.4f}")
        print(classification_report(y, preds, zero_division=0))

    # -- save artifacts -----------------------------------------------------------
    joblib.dump(vectorizer, LANG_DIR / "vectorizer.joblib")
    joblib.dump(model, LANG_DIR / "model.joblib")
    joblib.dump({"threshold": LANG_CONFIDENCE_THRESHOLD,
                 "classes": list(model.classes_)}, LANG_DIR / "meta.joblib")

    # -- confusion matrix ---------------------------------------------------------
    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, test_preds, labels=model.classes_)

    # low-confidence rate
    conf = test_proba.max(axis=1)
    unknown_rate = (conf < LANG_CONFIDENCE_THRESHOLD).mean()
    print(f"\nUnknown-flagging rate (conf<{LANG_CONFIDENCE_THRESHOLD}): {unknown_rate:.3%}")
    print("Unsure examples:")
    unsure = np.where(conf < LANG_CONFIDENCE_THRESHOLD)[0][:5]
    for i in unsure:
        print("  ", repr(test_df["text"].iloc[i])[:80], "->", y_test.iloc[i], f"(conf={conf[i]:.2f})")
    return cm, test_df, model, vectorizer


if __name__ == "__main__":
    main()