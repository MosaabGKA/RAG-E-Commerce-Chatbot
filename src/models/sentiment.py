"""Sentiment / tone classifier wrapper (BiLSTM/GRU + learned embeddings)."""
import joblib
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import SENTIMENT_DIR
from src.preprocessing import normalize_text
from src.models.sentiment_model import RNNSentimentModel

MAX_LEN = 64


class SentimentClassifier:
    def __init__(self, sentiment_dir=SENTIMENT_DIR):
        self.vocab = joblib.load(sentiment_dir / "vocab.joblib")
        self.buckets = joblib.load(sentiment_dir / "buckets.joblib")
        ckpt = torch.load(sentiment_dir / "sentiment.pt", map_location="cpu", weights_only=False)
        if "config" in ckpt:
            cfg = ckpt["config"]
        else:  # backward compat with pre-config checkpoints
            cfg = {"vocab_size": ckpt["vocab_size"], "cell": "lstm",
                   "hidden": 128, "embed": 128, "num_classes": len(self.buckets)}
        self.model = RNNSentimentModel(cfg["vocab_size"], embedding_dim=cfg["embed"],
                                       hidden_dim=cfg["hidden"],
                                       num_classes=cfg["num_classes"], cell=cfg["cell"],
                                       padding_idx=0)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        toks = [self.vocab.get(w, 1) for w in normalize_text(text).split()][:MAX_LEN]
        x = torch.tensor([toks], dtype=torch.long)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return {
            "sentiment": self.buckets[idx],
            "confidence": float(probs[idx]),
            "probabilities": {b: float(p) for b, p in zip(self.buckets, probs)},
        }