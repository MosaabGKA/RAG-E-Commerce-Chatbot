"""FastAPI deployment for the RAG customer-support chatbot.

Run from the project root:
    RAG_DEBUG=1 CUDA_VISIBLE_DEVICES="" uvicorn deploy.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /            -> API info
    GET  /health      -> model availability check
    POST /chat        -> {message: str} -> full 4-stage pipeline result
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.router import SupportChatbot


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    user_message: str
    language: str
    sentiment: str
    intent: str
    intent_confidence: float
    priority_flag: bool
    escalate: bool
    response_type: str
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = SupportChatbot()
    yield


app = FastAPI(title="NLP RAG Customer-Support Chatbot", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name": "NLP RAG Customer-Support Chatbot",
        "endpoints": {"/health": "GET", "/chat": "POST"},
        "llm": "available" if app.state.bot.generator.available else "offline (template fallback)",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "language_model": "loaded",
        "sentiment_model": "loaded",
        "intent_model": "loaded",
        "rag_index_vectors": app.state.bot.retriever.index.ntotal,
        "llm": "available" if app.state.bot.generator.available else "offline",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    out = app.state.bot.handle(req.message)
    return ChatResponse(
        user_message=out["user_message"],
        language=out["language"],
        sentiment=out["sentiment"]["sentiment"],
        intent=out["intent_route"]["category"],
        intent_confidence=round(out["intent_route"]["confidence"], 4),
        priority_flag=out["priority_flag"],
        escalate=out["escalate"],
        response_type=out["response_type"],
        response=out["response"],
    )