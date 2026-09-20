# RAG-Based E-Commerce Customer Support Chatbot

End-to-end chatbot that processes every customer message through **four stages** before answering:

1. **Language detection** (traditional NLP: char n-gram TF-IDF + logistic regression — ~99% test accuracy, 20 languages)
2. **Sentiment / tone** (BiLSTM/GRU + learned embeddings, 3 buckets: negative/neutral/positive — 89% on a customer-support-style set)
3. **Intent classification / routing** (TF-IDF + logistic regression on gold Bitext intents condensed to routing categories — ~99.8% test accuracy)
4. **Grounded Q&A (RAG)** (local FAISS index + hybrid BM25/dense retrieval + Groq LLM generation, offline template fallback)

## Project layout

```
project/
  data/                 datasets cache + hand-labeled tone samples
  artifacts/            trained artifacts (vectorizers, models, FAISS index)
  src/
    config.py           constants, intent & sentiment mappings
    preprocessing.py    shared text cleaning
    models/             per-module inference wrappers (language, sentiment, intent, rag)
    router.py           SupportChatbot: 4-stage orchestration + routing policy
  notebooks/            4 deliverable notebooks (all executed with outputs)
  scripts/              training + build + notebook executor scripts
  deploy/
    app.py              FastAPI server
    run_pipeline.py     CLI demo (no server needed)
    .env.example        copy to .env and set GROQ_API_KEY
```

## Setup

```bash
python -m venv venv && source venv/bin/activate     # or reuse an existing venv
pip install -r requirements.txt
cp deploy/.env.example .env                          # then set GROQ_API_KEY=...
```

## Run the full pipeline

```bash
# Offline demo (works without a Groq key — template fallback)
CUDA_VISIBLE_DEVICES="" python deploy/run_pipeline.py
CUDA_VISIBLE_DEVICES="" python deploy/run_pipeline.py "where is my order"

# Live LLM answers (with GROQ_API_KEY)
GROQ_API_KEY=sk-... python deploy/run_pipeline.py

# FastAPI server
CUDA_VISIBLE_DEVICES="" uvicorn deploy.app:app --app-dir . --port 8000
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Where is my order? It is two weeks late!"}'
```

API: `GET /health` · `POST /chat` (returns language, sentiment, intent, priority flag,
response type and the grounded answer).

## Train / build again

```bash
CUDA_VISIBLE_DEVICES="" python scripts/train_language.py
CUDA_VISIBLE_DEVICES="" python scripts/train_sentiment.py
CUDA_VISIBLE_DEVICES="" python scripts/train_intent.py
CUDA_VISIBLE_DEVICES="" python scripts/build_rag_index.py
```

The four `notebooks/*.ipynb` reproduce each module with evaluation + saved artifacts.

## Datasets

| Module      | Dataset |
|-------------|---------|
| Language    | `papluca/language-identification` (90k, 20 langs, pre-split) |
| Sentiment   | `dair-ai/emotion` (6 emotions → 3 buckets) + support-domain augmentation |
| Intent + RAG| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 pairs) |


