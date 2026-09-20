"""Intent -> routing category classifier wrapper (TF-IDF + logistic regression)."""
import re

import joblib

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import INTENT_DIR, ROUTING_CATEGORIES
from src.preprocessing import normalize_text

# lightweight small-talk detection for intents absent from the Bitext training data
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s?(morning|afternoon|evening)|howdy|yo|greetings|hola"
    r"|bonjour|salut|hallo|guten\s?tag|ciao|merhaba|konnichiwa|안녕|مرحبا|こんにちは)\b"
    r"[!.?\s,]*(how are you.*)?$", re.I | re.UNICODE)
_THANKS_RE = re.compile(r"\b(thanks|thank you|thx|ty|i appreciate( it| you)|cheers"
                        r"|merci|gracias|danke|grazie)\b", re.I)
_GOODBYE_RE = re.compile(r"^(bye|goodbye|good bye|see you|later|ciao|au revoir"
                         r"|auf wiedersehen|adios)\b.*$", re.I)

SMALLTALK_RESPONSES = {
    "greeting": "Hello! Welcome to our customer support. How can I help you today? "
                "You can ask me about your orders, deliveries, refunds, invoices or your account.",
    "thank_you": "You're welcome! It was my pleasure. Is there anything else I can help you with?",
    "goodbye": "Thanks for reaching out! Have a great day, and don't hesitate to contact us again if you need anything.",
}

OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, but that's outside the topics I can assist with "
    "(orders, deliveries, refunds, invoices, accounts). I'll escalate your request "
    "to a human agent so they can help you properly."
)

# Topic gate: if the user's message mentions any of these retail-support terms we
# trust the classifier even at lower confidence; otherwise a low-confidence
# prediction is treated as out-of-scope (prevents absurd grounded answers).
TOPIC_KEYWORDS = [
    "order", "orders", "delivery", "deliver", "ship", "shipping", "shipped", "arrive",
    "arrived", "arrives", "package", "parcel", "track", "tracking", "return", "returns",
    "returning", "refund", "refunds", "invoice", "bill", "billing", "payment", "pay",
    "paying", "price", "charge", "charged", "charges", "account", "login", "password",
    "email", "coupon", "cancel", "cancelled", "cancellation", "complaint", "complain",
    "product", "item", "subscription", "newsletter", "warranty", "size", "color",
    "colour", "discount", "courier", "agent", "human", "contact", "checkout",
]

OUT_OF_SCOPE_CONFIDENCE = 0.55
_TOPIC_RE = re.compile(r"\b(" + "|".join(TOPIC_KEYWORDS) + r")\b", re.I)


class IntentClassifier:
    def __init__(self, intent_dir=INTENT_DIR):
        self.vectorizer = joblib.load(intent_dir / "vectorizer.joblib")
        self.model = joblib.load(intent_dir / "model.joblib")
        self.classes = self.model.classes_

    def _smalltalk_route(self, text):
        lowered = text.lower().strip()
        if _GOODBYE_RE.match(lowered) or re.search(r"\bbye\b", lowered):
            return "greeting", "goodbye"
        if _THANKS_RE.search(lowered) and len(lowered) < 60:
            return "greeting", "thank_you"
        if _GREETING_RE.match(lowered):
            return "greeting", "greeting"
        return None

    def _out_of_scope(self, text):
        return {"category": "out_of_scope", "intent": "out_of_scope",
                "confidence": 0.0, "source": "relevance_gate"}

    def predict(self, text, language=None):
        small = self._smalltalk_route(text)
        if small:
            return {"category": small[0], "intent": small[1], "confidence": 1.0,
                    "source": "heuristic"}
        clean = normalize_text(text, keep_chars=True)
        X = self.vectorizer.transform([clean])
        probs = self.model.predict_proba(X)[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        category = str(self.model.classes_[idx])
        # Relevance gate only applies to English/unknown text; for other detected
        # languages the English-trained classifier is unreliable, so we let the
        # retrieval/generation step handle the message instead of rejecting it.
        non_english = language not in (None, "en", "unknown")
        if not non_english and not _TOPIC_RE.search(text) and confidence < OUT_OF_SCOPE_CONFIDENCE:
            return self._out_of_scope(text)
        return {
            "category": category,
            "intent": category,
            "confidence": confidence,
            "source": "classifier",
        }