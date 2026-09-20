"""RAG subsystem: local FAISS retriever (hybrid BM25+cosine) + Groq grounded generation."""
import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import GROQ_MODEL, RAG_DIR, TOP_K

_WORD_RE = re.compile(r"\w+")

SYSTEM_TEMPLATE = (
    "You are a helpful, professional customer support assistant for an online retailer. "
    "Answer the customer's question using ONLY the information in the retrieved support "
    "responses below. If the customer sounds {sentiment} ({sentiment_explanation}), "
    "acknowledge that before answering in an empathetic tone. If the retrieved context "
    "does NOT cover the question, say so honestly and offer to escalate to a human agent "
    "rather than guessing. Reply in the same language the customer used ({language})."
)

CONTEXT_TEMPLATE = (
    "Context (retrieved past support responses):\n"
    "{context}\n\n"
    "Customer question: \"{question}\""
)

SENTIMENT_EXPLANATION = {
    "negative": "frustrated or upset",
    "neutral": "neither happy nor upset",
    "positive": "satisfied and happy",
}


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


class Retriever:
    def __init__(self, rag_dir=RAG_DIR, hybrid=True, bm25_pool=120, bm25_k1=1.2, bm25_b=0.75):
        self.hybrid = hybrid
        self.bm25_pool = bm25_pool
        self.bm25_k1, self.bm25_b = bm25_k1, bm25_b
        self.index = faiss.read_index(str(rag_dir / "faiss.index"))
        with open(rag_dir / "metadata.pkl", "rb") as fh:
            payload = pickle.load(fh)
        self.metadata = payload["metadata"]
        with open(rag_dir / "embedder_path.txt") as fh:
            model_name = fh.read().strip()
        self.embedder = SentenceTransformer(model_name, device="cpu")
        if self.hybrid:
            self._build_bm25()

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        corpus = [_tokenize(m["instruction"]) for m in self.metadata]
        self.bm25 = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)

    def retrieve(self, query, k=TOP_K):
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        if self.hybrid:
            bm_scores = self.bm25.get_scores(_tokenize(query))
            candidates = np.argsort(bm_scores)[-self.bm25_pool:][::-1]
            cand_vec = None
            if len(candidates) > k:
                cand_vec = self.embedder.encode(
                    [self.metadata[int(i)]["instruction"] for i in candidates],
                    normalize_embeddings=True,
                ).astype("float32")
                sims = cand_vec @ q_vec.T
                order = candidates[np.argsort(sims[:, 0])[::-1][:k]]
            else:
                order = candidates
            hits = []
            for i in order:
                meta = self.metadata[int(i)]
                entries = {"bm25": float(bm_scores[int(i)]), **meta}
                if cand_vec is not None:
                    idx_in_cand = int(np.where(candidates == i)[0][0])
                    entries["score"] = float(cand_vec[idx_in_cand] @ q_vec.T)
                else:
                    entries["score"] = float(q_vec @ self.embedder.encode(
                        [meta["instruction"]], normalize_embeddings=True).T)
                hits.append(entries)
            return hits
        scores, idx = self.index.search(q_vec, k)
        hits = []
        for score, i in zip(scores[0], idx[0]):
            hits.append({"score": float(score), **self.metadata[int(i)]})
        return hits


class RAGGenerator:
    def __init__(self, model=None, api_key=None):
        from groq import Groq
        self.model = model or os.getenv("GROQ_MODEL", GROQ_MODEL)
        self.key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.key) if self.key else None

    @property
    def available(self):
        return self.client is not None

    def _build_messages(self, question, hits, sentiment, language="en"):
        context = "\n\n".join(
            f"{i + 1}) Q: {h['instruction']}\n   A: {h['response']}" for i, h in enumerate(hits)
        )
        system = SYSTEM_TEMPLATE.format(
            sentiment=sentiment,
            sentiment_explanation=SENTIMENT_EXPLANATION[sentiment],
            language=language,
        )
        user = CONTEXT_TEMPLATE.format(context=context, question=question)
        return system, user

    def generate(self, question, hits, sentiment="neutral", language="en"):
        system, user = self._build_messages(question, hits, sentiment, language)
        if not self.available:
            # offline fallback: grounded template answer from top retrieved response
            top = hits[0] if hits else None
            if not top:
                return ("I don't have enough information to answer that. "
                        "Let me escalate this to a human agent for you."), []
            return (
                f"Regarding your question about \"{top['instruction']}\": "
                f"{top['response']}",
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content, [system, user]


if __name__ == "__main__":
    q = "I ordered a jacket two weeks ago, where is it?"
    r = Retriever()
    hits = r.retrieve(q, k=3)
    print("Top hits:")
    for h in hits:
        print(f"  [{h['score']:.3f}] ({h['intent']}) {h['instruction'][:70]}")
    gen = RAGGenerator()
    print("\nGenerated answer:")
    print(gen.generate(q, hits, sentiment="negative")[0])