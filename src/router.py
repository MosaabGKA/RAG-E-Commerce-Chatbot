"""End-to-end orchestrator: the 4-stage customer-support chatbot pipeline.

Stage flow per incoming message:
  1. Language detection   -> know which KB language / reply language
  2. Sentiment / tone     -> adjust empathy & priority flagging
  3. Intent classification-> route to the best response path
  4. Q&A RAG              -> grounded answer from the support KB

Routing policy (documented in docs/design_decisions.md):
  * greeting            -> canned small-talk, NO retrieval
  * order_* / billing   -> grounded RAG answer
  * account_management  -> grounded RAG answer
  * complaint           -> empathy prefix + priority flag + still a grounded answer,
                           plus an explicit offer to escalate to a human agent
  * negative sentiment  -> (regardless of intent) empathy acknowledgment + priority flag
  * out_of_scope        -> honest "can't help" + escalate to a human agent
"""
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TOP_K
from src.models.language import LanguageDetector
from src.models.sentiment import SentimentClassifier
from src.models.intent import IntentClassifier, SMALLTALK_RESPONSES, OUT_OF_SCOPE_RESPONSE
from src.models.rag import Retriever, RAGGenerator

_LANGUAGE_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian", "ar": "Arabic",
    "tr": "Turkish", "hi": "Hindi", "ur": "Urdu", "vi": "Vietnamese", "th": "Thai",
    "ja": "Japanese", "zh": "Chinese", "el": "Greek", "bg": "Bulgarian", "sw": "Swahili",
    "unknown": "English",
}

_NEGATIVE_PREFIX = (
    "I'm really sorry to hear you're having a frustrating experience — "
    "that's not the level of service we aim for, and I'll take this seriously. "
)

ESCALATION_SUFFIX = (
    " If this doesn't resolve things, I can also pass your case to a human agent right away."
)

COMPLAINT_SUFFIX = (
    " I've flagged your case for priority review by a human support specialist."
)


class SupportChatbot:
    def __init__(self, api_key=None, model=None, hybrid=True):
        self.lang_detector = LanguageDetector()
        self.sentiment_model = SentimentClassifier()
        self.intent_model = IntentClassifier()
        self.retriever = Retriever(hybrid=hybrid)
        self.generator = RAGGenerator(model=model, api_key=api_key)

    # ------------------------------------------------------------------
    def handle(self, message):
        result = {"user_message": message}

        # 1) language ---------------------------------------------------
        lang = self.lang_detector.detect(message)
        result["language"] = lang["language"]
        result["language_confidence"] = lang["confidence"]
        language_name = _LANGUAGE_NAME.get(result["language"], "English")

        # 2) sentiment / tone --------------------------------------------
        tone = self.sentiment_model.predict(message)
        result["sentiment"] = tone
        negative = tone["sentiment"] == "negative"

        # 3) intent -------------------------------------------------------
        intent = self.intent_model.predict(message, language=result["language"])
        result["intent_route"] = intent
        category = intent["category"]

        # 4) route ---------------------------------------------------------
        priority_flag = (category == "complaint") or negative
        result["priority_flag"] = bool(priority_flag)

        if category == "greeting":
            canned = SMALLTALK_RESPONSES.get(intent.get("intent", "greeting"),
                                             SMALLTALK_RESPONSES["greeting"])
            result.update({"response_type": "smalltalk", "retrieved_chunks": [],
                           "response": canned, "escalate": False})
            return result

        if category == "out_of_scope":
            result.update({"response_type": "out_of_scope", "retrieved_chunks": [],
                           "response": OUT_OF_SCOPE_RESPONSE, "escalate": True})
            return result

        # RAG path ----------------------------------------------------------
        hits = self.retriever.retrieve(message, k=TOP_K)
        answer, messages = self.generator.generate(
            question=message, hits=hits, sentiment=tone["sentiment"], language=language_name)

        prefix = ""
        suffix = ""
        if negative:
            prefix = _NEGATIVE_PREFIX
        if category == "complaint":
            suffix = COMPLAINT_SUFFIX + ESCALATION_SUFFIX
        elif negative:
            suffix = ESCALATION_SUFFIX

        result.update({
            "response_type": "rag",
            "retrieved_chunks": [{
                "score": h["score"],
                "instruction": h["instruction"],
                "response": h["response"],
                "intent": h.get("intent"),
            } for h in hits],
            "response": prefix + answer + suffix,
            "escalate": bool(priority_flag),
            "prompt_messages": messages if os.getenv("RAG_DEBUG") else None,
        })
        return result


if __name__ == "__main__":
    bot = SupportChatbot()
    demo = [
        "Where is my order? I ordered two weeks ago and it never arrived. This is ridiculous!",
        "hello there",
        "I want a refund for the shoes I returned last month",
        "Can you tell me your delivery options?",
        "what is the meaning of life",
        "Merci, ma commande est arrivee parfaitement",
    ]
    for msg in demo:
        print("=" * 100)
        print("USER:", msg)
        out = bot.handle(msg)
        print("lang:", out["language"], "| sentiment:", out["sentiment"]["sentiment"],
              "| intent:", out["intent_route"]["category"], "| priority:", out["priority_flag"])
        print("BOT:", out["response"][:300])