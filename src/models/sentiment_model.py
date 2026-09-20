"""Reusable BiLSTM/GRU text classifier used for sentiment/tone detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNSentimentModel(nn.Module):
    """Embedding -> (Bi)LSTM/GRU -> concat(mean-pool, max-pool) -> classifier head."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_classes=3, num_layers=1, cell="lstm", bidirectional=True,
                 dropout=0.3, padding_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        cell_cls = nn.LSTM if cell == "lstm" else nn.GRU
        rnn_input = embedding_dim
        self.rnn = cell_cls(rnn_input, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        direction = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * direction * 2, hidden_dim * direction),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction, num_classes),
        )

    def forward(self, x, lengths=None):
        emb = self.embed(x)                     # B x T x E
        out, _ = self.rnn(emb)
        mask = (x != self.embed.padding_idx).unsqueeze(-1)
        mask = mask.expand_as(out).float()
        mean_pool = (out * mask).sum(1) / (mask.sum(1) + 1e-9)
        max_pool = (out * mask).max(1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.fc(pooled)
        return logits


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += yb.numel()
        correct += (logits.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        preds = logits.argmax(1)
        total += yb.numel()
        correct += (preds == yb).sum().item()
        loss_sum += loss.item() * yb.numel()
        all_preds.extend(preds.cpu().tolist())
        all_y.extend(yb.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_y