"""Language detection wrapper (TF-IDF char n-grams + logistic regression)."""
import joblib

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LANG_DIR
from src.preprocessing import normalize_text


class LanguageDetector:
    def __init__(self, lang_dir=LANG_DIR):
        self.vectorizer = joblib.load(lang_dir / "vectorizer.joblib")
        self.model = joblib.load(lang_dir / "model.joblib")
        meta = joblib.load(lang_dir / "meta.joblib")
        self.threshold = meta["threshold"]
        self.classes = self.model.classes_

    def detect(self, text):
        clean = normalize_text(text, keep_chars=True)
        X = self.vectorizer.transform([clean])
        probs = self.model.predict_proba(X)[0]
        idx = int(probs.argmax())
        lang = self.model.classes_[idx]
        return {
            "language": lang if probs[idx] >= self.threshold else "unknown",
            "confidence": float(probs[idx]),
            "all_probs": {c: float(p) for c, p in zip(self.model.classes_, probs)},
        }