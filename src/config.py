"""Central configuration and constants for the NLP RAG chatbot project."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASETS_DIR = DATA_DIR / "datasets"

LANG_DIR = ARTIFACTS_DIR / "language"
SENTIMENT_DIR = ARTIFACTS_DIR / "sentiment"
INTENT_DIR = ARTIFACTS_DIR / "intent"
RAG_DIR = ARTIFACTS_DIR / "rag"

for _d in (DATA_DIR, ARTIFACTS_DIR, DATASETS_DIR, LANG_DIR, SENTIMENT_DIR, INTENT_DIR, RAG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets (HF)
# ---------------------------------------------------------------------------
LANG_DATASET = "papluca/language-identification"          # 90k / 20 langs / pre-split
EMOTION_DATASET = "dair-ai/emotion"                        # 20k / 6 emotions
BITEXT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Embedding model for RAG retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM used for grounded generation
GROQ_MODEL = "gpt-oss-20b"   # swap to "gpt-oss-120b" if desired

# RAG retrieval settings
TOP_K = 4

# ---------------------------------------------------------------------------
# Sentiment: map the 6 emotions from dair-ai/emotion -> 3 tone buckets
# ---------------------------------------------------------------------------
EMOTION_ID2LABEL = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

EMOTION_TO_BUCKET = {
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
}
SENTIMENT_BUCKETS = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# Intent: Bitext fine-grained intents -> routing categories
# (condensed from the 27 fine-grained intents described in the assignment)
# ---------------------------------------------------------------------------
INTENT_TO_CATEGORY = {
    # small talk - no retrieval needed
    "greeting": "greeting",
    "goodbye": "greeting",
    "thank_you": "greeting",
    "appreciation": "greeting",
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    "contact_customer_service": "order_status",
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "refund_not_received": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    "payment_methods": "billing_and_refunds",
    "cancel_subscription": "billing_and_refunds",
    "change_subscription": "billing_and_refunds",
    "check_cancellation_fee": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "edit_account": "account_management",
    "create_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    "registration_problems": "account_management",
    "change_shipping_address": "account_management",
    "set_up_shipping_address": "account_management",
    "newsletter_subscription": "account_management",
    "complaint": "complaint",
    "review": "complaint",
    "customer_service": "complaint",
    "contact_human_agent": "complaint",
    "out_of_scope": "out_of_scope",
}

ROUTING_CATEGORIES = [
    "greeting",
    "order_status",
    "order_management",
    "billing_and_refunds",
    "account_management",
    "complaint",
    "out_of_scope",
]

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
LANG_CONFIDENCE_THRESHOLD = 0.6  # below this -> treat as unknown language

SUPPORTED_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]