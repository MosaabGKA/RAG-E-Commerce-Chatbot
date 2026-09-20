"""Shared text pre-processing helpers used across all modules."""
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#\w+")
MONEY_RE = re.compile(r"\$\d[\d,.]*|[€£]\s?\d[\d,.]*|\d+\s?(usd|eur|gbp)")
NUMBER_RE = re.compile(r"\d[_.,]?\d*")
MULTI_SPACE_RE = re.compile(r"\s{2,}")
# unicode-aware: keep letters/digits of every script, drop punctuation/symbols
NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str, keep_chars: bool = True) -> str:
    """Basic normalization: strip urls/emails/mentions/hashtags, mask numbers,
    lowercase, then collapse whitespace.

    keep_chars=True  -> drop punctuation / symbols but keep letters of ALL
                        scripts (needed for e.g. Arabic on a language model)."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASH_RE.sub(" ", text)
    text = MONEY_RE.sub(" <currency> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    if keep_chars:
        text = NON_WORD_RE.sub(" ", text)
    # collapse whitespace
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_vocab(texts, min_freq: int = 2, max_size: int = 20_000):
    """Build a word->id vocab from a sequence of tokenized texts."""
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab