"""Command-line demo of the full chatbot pipeline (works without a server).

Usage:
    cd NLP/project
    CUDA_VISIBLE_DEVICES="" python deploy/run_pipeline.py                  # built-in demos
    CUDA_VISIBLE_DEVICES="" python deploy/run_pipeline.py "my own question"  # single query
    GROQ_API_KEY=... python deploy/run_pipeline.py "where is my order"     # live LLM answers
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.router import SupportChatbot

DEMO_QUERIES = [
    "hello there",
    "Where is my order? I ordered two weeks ago and it never arrived. This is ridiculous!",
    "I want a refund for the shoes I returned last month",
    "Can you tell me your delivery options?",
    "My password is not working, I can't log into my account",
    "I would like to complain about the delivery service",
    "what is the meaning of life",
    "Merci, ma commande est arrivee parfaitement",
]


def pretty(out):
    print("\n  [language]    ", out["language"], f"(conf={out['language_confidence']:.2f})")
    print("  [sentiment]   ", out["sentiment"]["sentiment"],
          f"(conf={out['sentiment']['confidence']:.2f})")
    print("  [intent route]", out["intent_route"]["category"],
          f"(conf={out['intent_route']['confidence']:.2f}, src={out['intent_route']['source']})")
    print("  [priority]    ", out["priority_flag"], "| [escalate]", out["escalate"])
    print("  [type]        ", out["response_type"])
    if out.get("retrieved_chunks"):
        print("  [top chunk]   ", out["retrieved_chunks"][0]["instruction"][:90])
    print("  [answer]      ", out["response"][:400])


def main():
    bot = SupportChatbot()
    llm_state = "Groq LLM (live)" if bot.generator.available else "offline template fallback"
    print(f"LLM backend: {llm_state}\n{'='*90}")

    if len(sys.argv) > 1:
        query = sys.argv[1]
        out = bot.handle(query)
        pretty(out)
        print(json.dumps(out["retrieved_chunks"][:2], indent=2, ensure_ascii=False))
        return

    for q in DEMO_QUERIES:
        print("=" * 90)
        print("USER:", q)
        pretty(bot.handle(q))


if __name__ == "__main__":
    main()