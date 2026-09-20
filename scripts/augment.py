"""Domain-adaptation augmentation for the sentiment/tone model.

The dair-ai/emotion set is Twitter text. To make tone detection work on
customer-support writing we augment the train split with:
  * real neutral support queries sampled from the Bitext KB (the support corpus)
  * synthetic support-style negative (complaint) and positive (gratitude) messages
"""
import random

from datasets import load_dataset

from src.config import BITEXT_DATASET

NEG_TEMPLATES = [
    "i am very unhappy with {}",
    "{} is a total disaster and i am furious",
    "why is {} taking so long, this is unacceptable",
    "{} never works and no one helps me, i want a human",
    "i have been waiting for {} for weeks, ridiculous",
    "i got charged twice for {}, this is outrageous",
    "{} arrived damaged, i am extremely frustrated",
    "my {} was cancelled without warning, i am angry",
    "i will complain about {} everywhere, terrible service",
    "{} is a rip off, i demand a refund right now",
    "stop ignoring my messages about {}, it is so annoying",
    "this {} situation is driving me crazy",
    "i am so fed up with {}, worst experience ever",
    "{} came in the wrong colour again, unbelievable",
    "i want {} sorted immediately, i am not happy",
]
POS_TEMPLATES = [
    "thank you so much for {}, really appreciate it",
    "i am delighted with {}, great job",
    "{} was handled perfectly, i love it",
    "amazing service with {}, five stars",
    "very happy with {}, thanks a lot",
    "{} exceeded my expectations, fantastic",
    "great support team, solved {} quickly",
    "i am thrilled about {}, wonderful",
    "{} arrived early, brilliant",
    "such a pleasant experience with {}, thank you",
    "i really like {}, keep up the good work",
    "{} is perfect, i am satisfied",
    "best experience ever with {}, impressive",
    "{} worked flawlessly, lucky to have this service",
    "super pleased with {}, would recommend",
]
TOPICS = [
    "my order", "the delivery", "my refund", "this product", "the customer service",
    "my account", "the payment", "the invoice", "the return process", "my package",
    "the exchange", "my subscription", "the tracking", "the quality", "the support",
]


def _expand(templates, n):
    out = []
    while len(out) < n:
        t = random.choice(templates)
        places = t.count("{}")
        texts = [t.format(*([random.choice(TOPICS)] * places))]
        out.extend(texts)
    return out[:n]


def load_augmentation(n_neutral=6000, n_syn=900, seed=0):
    random.seed(seed)
    neutral = _neutral_support_queries(n_neutral)
    negative = _expand(NEG_TEMPLATES, n_syn)
    positive = _expand(POS_TEMPLATES, n_syn)
    return {"neutral": neutral, "negative": negative, "positive": positive}


def _neutral_support_queries(n):
    ds = load_dataset(BITEXT_DATASET)
    df = ds["train"].to_pandas()[["instruction", "intent"]].dropna()
    # skip complaints & reviews — those carry negative tone
    df = df[~df["intent"].isin(["complaint", "review"])]
    texts = df["instruction"].tolist()
    random.seed(0)
    return random.sample(texts, min(n, len(texts)))