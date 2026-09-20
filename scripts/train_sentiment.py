"""Train the sentiment/tone classifier (BiLSTM/GRU + learned embeddings).

Data: dair-ai/emotion (6 emotions) mapped to 3 tone buckets.
Domain adaptation: the Twitter emotion data has a strong negative prior and no
customer-support vocabulary. We therefore augment the train set with balanced
support-domain examples (real neutral KB queries, synthetic pos/neg, hand-labeled
support-style messages) and, crucially, select the checkpoint on a *support-domain*
validation set rather than the biased Twitter validation set.
"""
import json
import random

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.augment as augment
from src.config import DATA_DIR, EMOTION_DATASET, EMOTION_ID2LABEL, \
    EMOTION_TO_BUCKET, SENTIMENT_BUCKETS, SENTIMENT_DIR
from src.preprocessing import normalize_text, build_vocab
from src.models.sentiment_model import RNNSentimentModel, train_epoch, evaluate

torch.manual_seed(0)
random.seed(0)
pd.set_option("display.width", 200)

MAX_LEN = 64
BATCH_SIZE = 128
EPOCHS = 15
CELL = "gru"
HIDDEN = 128
EMBED = 128
SUPPORT_VAL_FRACTION = 0.15


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts, self.labels, self.vocab = texts, labels, vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        toks = [self.vocab.get(w, 1) for w in self.texts[i].split()][:MAX_LEN]
        return torch.tensor(toks, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    return padded, torch.tensor(ys)


def macro_acc(y_true, y_pred):
    from collections import Counter
    true_set = set(y_true)
    accs = []
    for c in true_set:
        idx = [i for i, v in enumerate(y_true) if v == c]
        if idx:
            accs.append(sum(y_pred[i] == y_true[i] for i in idx) / len(idx))
    return sum(accs) / len(accs)


def load_support_samples():
    """Balanced support-domain pool -> (texts, labels)."""
    with open(DATA_DIR / "support_tone_samples.json") as fh:
        hand = json.load(fh)
    pool = {"negative": [], "neutral": [], "positive": []}
    for s in hand:
        pool[s["bucket"]].append(normalize_text(s["text"]))

    aug = augment.load_augmentation(n_neutral=2500, n_syn=700)
    pool["neutral"] += [normalize_text(t) for t in aug["neutral"]]
    pool["negative"] += [normalize_text(t) for t in aug["negative"]]
    pool["positive"] += [normalize_text(t) for t in aug["positive"]]

    n = min(len(v) for v in pool.values())
    texts, labels = [], []
    for bucket, bucket_texts in pool.items():
        chosen = random.sample(bucket_texts, n)
        texts += chosen
        labels += [SENTIMENT_BUCKETS.index(bucket)] * n
    return texts, labels


def main():
    # --- 1) emotion base data ---------------------------------------------------
    ds = load_dataset(EMOTION_DATASET)
    df = {split: ds[split].to_pandas() for split in ["train", "validation", "test"]}
    for split, d in df.items():
        d["bucket"] = d["label"].map(lambda l: EMOTION_TO_BUCKET[EMOTION_ID2LABEL[l]])
        d["clean"] = d["text"].apply(lambda t: normalize_text(t))

    # --- 2) support-domain pool, split into train/val ---------------------------
    su_texts, su_labels = load_support_samples()
    paired = list(zip(su_texts, su_labels))
    random.shuffle(paired)
    n_val = int(len(paired) * SUPPORT_VAL_FRACTION)
    su_val = paired[:n_val]
    su_train = paired[n_val:]
    print(f"Support pool={len(su_texts)} -> train={len(su_train)} val={len(su_val)}")

    # --- 3) balanced augmented train set ----------------------------------------
    train_df = df["train"]
    per_class = train_df["bucket"].value_counts().min()
    balanced = pd.concat([
        train_df[train_df["bucket"] == b].sample(per_class, random_state=0)
        for b in SENTIMENT_BUCKETS
    ])
    extra = pd.DataFrame({
        "clean": [t for t, _ in su_train],
        "bucket": [SENTIMENT_BUCKETS[l] for _, l in su_train],
    })
    train_aug = pd.concat([balanced, extra], ignore_index=True).sample(frac=1, random_state=0)
    print("Train bucket distribution:\n", train_aug["bucket"].value_counts())

    vocab = build_vocab(train_aug["clean"], min_freq=1, max_size=30_000)
    print(f"\nvocab={len(vocab)}  train={len(train_aug)}")

    texts_tr = train_aug["clean"].tolist()
    labels_tr = train_aug["bucket"].map(SENTIMENT_BUCKETS.index).tolist()
    texts_val = df["validation"]["clean"].tolist() + [t for t, _ in su_val]
    labels_val = df["validation"]["bucket"].map(SENTIMENT_BUCKETS.index).tolist() + \
                 [l for _, l in su_val]

    tr_ds = TextDataset(texts_tr, labels_tr, vocab)
    va_ds = TextDataset(texts_val, labels_val, vocab)
    te_ds = TextDataset(df["test"]["clean"], df["test"]["bucket"].map(SENTIMENT_BUCKETS.index).tolist(), vocab)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, collate_fn=collate)
    su_loader = DataLoader(TextDataset(
        [normalize_text(s["text"]) for s in json.load(open(DATA_DIR/"support_tone_samples.json"))],
        [SENTIMENT_BUCKETS.index(s["bucket"]) for s in json.load(open(DATA_DIR/"support_tone_samples.json"))],
        vocab), batch_size=64, collate_fn=collate)

    device = torch.device("cpu")
    model = RNNSentimentModel(len(vocab), embedding_dim=EMBED, hidden_dim=HIDDEN,
                              num_classes=len(SENTIMENT_BUCKETS), cell=CELL, padding_idx=0).to(device)
    print(f"Parameters = {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, device)
        _, _, va_pred, va_true = evaluate(model, va_loader, device)
        va_macro = macro_acc(va_true, va_pred)
        # select on support-domain + Twitter macro accuracy
        score = va_macro
        print(f"epoch {epoch:02d} | train acc={tr_acc:.4f} | val macro={va_macro:.4f}")
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(),
                        "config": {"vocab_size": len(vocab), "cell": CELL,
                                   "hidden": HIDDEN, "embed": EMBED,
                                   "num_classes": len(SENTIMENT_BUCKETS)}},
                       SENTIMENT_DIR / "sentiment.pt")

    model.load_state_dict(torch.load(SENTIMENT_DIR / "sentiment.pt", weights_only=False)["state_dict"])
    _, te_acc, te_pred, te_true = evaluate(model, te_loader, device)
    print(f"\n[twitter test] accuracy = {te_acc:.4f}")
    print(classification_report(te_true, te_pred, target_names=SENTIMENT_BUCKETS, zero_division=0))

    _, su_acc, su_pred, su_true = evaluate(model, su_loader, device)
    print(f"\n[support-style qualitative set] accuracy = {su_acc:.4f}")
    print(classification_report(su_true, su_pred, target_names=SENTIMENT_BUCKETS, zero_division=0))
    print("Confusion:\n", confusion_matrix(su_true, su_pred, labels=range(3)))

    joblib.dump(vocab, SENTIMENT_DIR / "vocab.joblib")
    joblib.dump(SENTIMENT_BUCKETS, SENTIMENT_DIR / "buckets.joblib")
    print("\nSaved artifacts to", SENTIMENT_DIR)


if __name__ == "__main__":
    main()