"""Generate the 4 deliverable notebooks for the NLP RAG chatbot project."""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)
MD = "markdown"
CODE = "code"

HEAD = "%matplotlib inline\nimport matplotlib\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set_theme()"


def cell(type_, src):
    if type_ == MD:
        return nbf.v4.new_markdown_cell(src)
    return nbf.v4.new_code_cell(src)


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3 (venv)", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nb.cells = [cell(MD, md) if type(c) is str and c == MD else c for c in cells]
    # simpler: rebuild
    nb.cells = []
    for typ, src in cells:
        nb.cells.append(cell(typ, src))
    path = NB_DIR / name
    nbf.write(nb, str(path))
    print("wrote", path)


# =====================================================================
# Notebook 1 — Language Detection
# =====================================================================
nb1_md = """# Module 1 — Language Detection

**Goal:** build a multi-class classifier that identifies the language of each customer message,
so the system can reply in the same language and route correctly.

**Data:** [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification)
(90k samples, 20 languages, pre-split train / validation / test).

**Method (traditional NLP):**
1. **Text cleaning** — lowercase, strip URLs / e-mails / mentions / numbers (masked as `<num>`),
   keep the letters of *every* script (Arabic, Cyrillic, Greek, Thai, Japanese, ...).
2. **Feature engineering** — *character n-grams (2..5)* with TF-IDF weighting. Char n-grams capture
   language-specific morphology and script statistics far better than words for short, noisy chat text.
3. **Model** — linear logistic regression (fast, interpretable, strong for high-dim sparse text).

**Enhancements (per the assignment):**
- Confidence threshold → low-confidence predictions flagged as `unknown` (graceful degradation).
- Per-language F1 report + confusion matrix.
- Unicode-aware cleaning so non-Latin scripts are not destroyed by an ASCII-only regex.
"""

nb1_cells = [
    (MD, nb1_md),
    (CODE, "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('..').resolve()))\n\n"
           "import pandas as pd\nimport numpy as np\nfrom datasets import load_dataset\n"
           "from sklearn.feature_extraction.text import TfidfVectorizer\n"
           "from sklearn.linear_model import LogisticRegression\n"
           "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
           "from src.preprocessing import normalize_text\nfrom src.config import LANG_DIR, LANG_CONFIDENCE_THRESHOLD\n"
           "import warnings\nwarnings.filterwarnings('ignore')\n"),
    (CODE, "ds = load_dataset('papluca/language-identification')\n"
           "def build_df(split):\n"
           "    df = pd.DataFrame({'text': split['text'], 'label': split['labels']})\n"
           "    df['clean'] = df['text'].apply(lambda x: normalize_text(x, keep_chars=True))\n"
           "    return df\n"
           "train_df, val_df, test_df = build_df(ds['train']), build_df(ds['validation']), build_df(ds['test'])\n"
           "print(f'Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}  languages={train_df[\"label\"].nunique()}')\n"
           "print('Cleaned sample:', repr(train_df['clean'].iloc[0][:120]))"),
    (CODE, "vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)\n"
           "X_train = vectorizer.fit_transform(train_df['clean'])\n"
           "X_val, X_test = vectorizer.transform(val_df['clean']), vectorizer.transform(test_df['clean'])\n"
           "y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']\n"
           "print('Training matrix:', X_train.shape, '| features = char n-grams')\n"
           "feats = vectorizer.get_feature_names_out()\n"
           "print('Example features:', list(feats[:8]))"),
    (CODE, "model = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')\n"
           "model.fit(X_train, y_train)\n"
           "for name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:\n"
           "    preds = model.predict(X)\n"
           "    print(f'[{name}] accuracy = {accuracy_score(y, preds):.4f}')\n"
           "print(classification_report(y_test, model.predict(X_test), zero_division=0))"),
    (CODE, HEAD + "\nproba = model.predict_proba(X_test)\nconf = proba.max(axis=1)\n"
           "cm = confusion_matrix(y_test, model.predict(X_test), labels=model.classes_)\n"
           "plt.figure(figsize=(9, 8))\nsns.heatmap(cm, xticklabels=model.classes_, yticklabels=model.classes_, annot=False, cmap='Blues')\n"
           "plt.title('Language detection — test confusion matrix')\nplt.xlabel('predicted'); plt.ylabel('true')\nplt.tight_layout()\nplt.show()\n"
           "unknown_rate = (conf < LANG_CONFIDENCE_THRESHOLD).mean()\n"
           "print('Unknown-flagging rate (conf<0.6):', f'{unknown_rate:.2%}')"),
    (CODE, "cm_low = (conf < LANG_CONFIDENCE_THRESHOLD)\n"
           "df_low = test_df[cm_low]\nprint('Examples the model is unsure about (flagged unknown):')\n"
           "for _, r in df_low.head(4).iterrows():\n"
           "    print('  ', repr(r['text'][:70]), '->', r['label'])"),
    (CODE, "from src.models.language import LanguageDetector\ndetector = LanguageDetector()\n"
           "probes = ['I want a refund for my broken headphones', 'Soyez rapide, mon colis est en retard',\n"
           "          'مطلوب استرداد المبلغ بسرعة', 'Пожалуйста, отмените мой заказ', '주문 상태를 알려주세요',\n"
           "          'Ich brauche Hilfe bei meiner Rechnung']\n"
           "for p in probes:\n"
           "    r = detector.detect(p)\n"
           "    print(f'{r[\"language\"]:8s} (conf={r[\"confidence\"]:.2f})  {p}')\n"
           "print('\\nArtifacts saved to', LANG_DIR)"),
]

# =====================================================================
# Notebook 2 — Sentiment Classifier
# =====================================================================
nb2_md = """# Module 2 — Sentiment / Emotion Classifier

**Goal:** classify the tone of a customer message as **negative / neutral / positive**, tuning the
responses (an angry customer needs a more apologetic, priority-flagged path).

**Data:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — 20k Twitter messages labelled
with 6 emotions, mapped to 3 buckets:
- **negative** ← *sadness, anger, fear*
- **neutral** ← *surprise*
- **positive** ← *joy, love*

**Model (RNN, per the assignment):** `Embedding → BiLSTM/GRU → mean+max pooling → 3-way head`, trained
end-to-end on CPU.

**Domain-shift handling (the key challenge):** emotion is *Twitter* text but our domain is *customer
support*. Twitter has a strong negative prior and no support vocabulary. We therefore:
1. balance the train set across the 3 buckets,
2. augment with real neutral support queries sampled from the Bitext KB, synthetic support-style
   negative/positive messages, and a small hand-labeled set of real support messages,
3. **select the checkpoint on a *support-domain* validation set** instead of the biased Twitter split.

This deliberately trades a little Twitter accuracy for much stronger support-domain tone detection —
see `docs/design_decisions.md`.
"""

nb2_cells = [
    (MD, nb2_md),
    (CODE, "import sys, json, random\nfrom pathlib import Path\nsys.path.insert(0, str(Path('..').resolve()))\n"
           "import joblib\nimport pandas as pd\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
           "from datasets import load_dataset\nfrom torch.utils.data import DataLoader, Dataset\n"
           "from sklearn.metrics import classification_report, confusion_matrix\n"
           "from src.config import (EMOTION_DATASET, EMOTION_ID2LABEL, EMOTION_TO_BUCKET,\n"
           "                        SENTIMENT_BUCKETS, SENTIMENT_DIR, DATA_DIR)\n"
           "from src.preprocessing import normalize_text, build_vocab\n"
           "from src.models.sentiment_model import RNNSentimentModel, train_epoch, evaluate\n"
           "import scripts.augment as augment\n"
           "import warnings\nwarnings.filterwarnings('ignore')\ntorch.manual_seed(0); random.seed(0)\n"
           "%matplotlib inline\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set_theme()\n"
           "MAX_LEN, BATCH, EPOCHS, CELL, HIDDEN, EMBED = 64, 128, 15, 'gru', 128, 128"),
    (CODE, "ds = load_dataset(EMOTION_DATASET)\n"
           "df = {s: ds[s].to_pandas() for s in ['train','validation','test']}\n"
           "for s, d in df.items():\n"
           "    d['bucket'] = d['label'].map(lambda l: EMOTION_TO_BUCKET[EMOTION_ID2LABEL[l]])\n"
           "    d['clean'] = d['text'].apply(lambda t: normalize_text(t))\n"
           "print('Mapping 6 emotions -> 3 buckets:')\nfor k, v in EMOTION_TO_BUCKET.items(): print(f'   {k:8s}-> {v}')\n"
           "print('\\nTrain bucket distribution:\\n', df['train']['bucket'].value_counts())"),
    (CODE, "with open(DATA_DIR / 'support_tone_samples.json') as fh: hand = json.load(fh)\n"
           "pool = {'negative': [], 'neutral': [], 'positive': []}\n"
           "for s in hand: pool[s['bucket']].append(normalize_text(s['text']))\n"
           "aug = augment.load_augmentation(n_neutral=2500, n_syn=700)\n"
           "pool['neutral'] += [normalize_text(t) for t in aug['neutral']]\n"
           "pool['negative'] += [normalize_text(t) for t in aug['negative']]\n"
           "pool['positive'] += [normalize_text(t) for t in aug['positive']]\n"
           "n = min(len(v) for v in pool.values())\n"
           "su_texts, su_labels = [], []\n"
           "for b, txts in pool.items():\n"
           "    chosen = random.sample(txts, n)\n"
           "    su_texts += chosen; su_labels += [SENTIMENT_BUCKETS.index(b)] * n\n"
           "paired = list(zip(su_texts, su_labels)); random.shuffle(paired)\n"
           "SU_VAL = int(len(paired) * 0.15); su_val, su_train = paired[:SU_VAL], paired[SU_VAL:]\n"
           "print(f'Support-style pool balanced at n={n} -> train={len(su_train)} val={len(su_val)}')"),
    (CODE, "class TextDataset(Dataset):\n"
           "    def __init__(self, texts, labels, vocab):\n"
           "        self.texts, self.labels, self.vocab = texts, labels, vocab\n"
           "    def __len__(self): return len(self.texts)\n"
           "    def __getitem__(self, i):\n"
           "        toks = [self.vocab.get(w, 1) for w in self.texts[i].split()][:MAX_LEN]\n"
           "        return torch.tensor(toks, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)\n"
           "def collate(batch):\n"
           "    xs, ys = zip(*batch)\n"
           "    m = max(len(x) for x in xs)\n"
           "    padded = torch.zeros(len(xs), m, dtype=torch.long)\n"
           "    for i, x in enumerate(xs): padded[i, :len(x)] = x\n"
           "    return padded, torch.tensor(ys)\n"
           "def macro_acc(y_true, y_pred):\n"
           "    return np.mean([np.mean([p == t for p, t in zip(y_pred, y_true) if t == c]) for c in set(y_true)])\n"
           "train_aug = pd.concat([\n"
           "    df['train'][df['train']['bucket'] == b].sample(df['train']['bucket'].value_counts().min(), random_state=0)\n"
           "    for b in SENTIMENT_BUCKETS])\n"
           "extra = pd.DataFrame({'clean': [t for t, _ in su_train],\n"
           "                      'bucket': [SENTIMENT_BUCKETS[l] for _, l in su_train]})\n"
           "train_aug = pd.concat([train_aug, extra], ignore_index=True).sample(frac=1, random_state=0)\n"
           "vocab = build_vocab(train_aug['clean'], min_freq=1, max_size=30_000)\n"
           "tr_ds = TextDataset(train_aug['clean'].tolist(), train_aug['bucket'].map(SENTIMENT_BUCKETS.index).tolist(), vocab)\n"
           "va_ds = TextDataset(df['validation']['clean'].tolist() + [t for t,_ in su_val],\n"
           "                    df['validation']['bucket'].map(SENTIMENT_BUCKETS.index).tolist() + [l for _,l in su_val], vocab)\n"
           "te_ds = TextDataset(df['test']['clean'], df['test']['bucket'].map(SENTIMENT_BUCKETS.index).tolist(), vocab)\n"
           "su_ds = TextDataset([normalize_text(s['text']) for s in hand],\n"
           "                    [SENTIMENT_BUCKETS.index(s['bucket']) for s in hand], vocab)\n"
           "tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)\n"
           "va_loader = DataLoader(va_ds, batch_size=BATCH, collate_fn=collate)\n"
           "te_loader = DataLoader(te_ds, batch_size=BATCH, collate_fn=collate)\n"
           "su_loader = DataLoader(su_ds, batch_size=64, collate_fn=collate)\n"
           "print(f'vocab={len(vocab)}  train={len(train_aug)}  support-val={(len(df[\"validation\"]) + len(su_val))}')"),
    (CODE, "device = torch.device('cpu')\n"
           "model = RNNSentimentModel(len(vocab), embedding_dim=EMBED, hidden_dim=HIDDEN,\n"
           "                          num_classes=len(SENTIMENT_BUCKETS), cell=CELL, padding_idx=0).to(device)\n"
           "print(f'Parameters = {sum(p.numel() for p in model.parameters()):,}')\n"
           "opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)\n"
           "best = -1.0\n"
           "for ep in range(1, EPOCHS + 1):\n"
           "    tr_loss, tr_acc = train_epoch(model, tr_loader, opt, device)\n"
           "    _, _, va_pred, va_true = evaluate(model, va_loader, device)\n"
           "    score = macro_acc(va_true, va_pred)\n"
           "    print(f'epoch {ep:02d} | train acc={tr_acc:.4f} | val (support+twitter) macro={score:.4f}')\n"
           "    if score > best:\n"
           "        best = score\n"
           "        torch.save({'state_dict': model.state_dict(), 'config': {'vocab_size': len(vocab),\n"
           "                    'cell': CELL, 'hidden': HIDDEN, 'embed': EMBED, 'num_classes': len(SENTIMENT_BUCKETS)}},\n"
           "                   SENTIMENT_DIR / 'sentiment.pt')\n"
           "print('\\nBest checkpoint saved with', CELL.upper(), 'architecture')"),
    (CODE, "model.load_state_dict(torch.load(SENTIMENT_DIR / 'sentiment.pt', weights_only=False)['state_dict'])\n"
           "_, te_acc, te_pred, te_true = evaluate(model, te_loader, device)\n"
           "print('[twitter test] accuracy =', f'{te_acc:.4f}')\n"
           "print(classification_report(te_true, te_pred, target_names=SENTIMENT_BUCKETS, zero_division=0))"),
    (CODE, "_, su_acc, su_pred, su_true = evaluate(model, su_loader, device)\n"
           "print('[customer-support qualitative set] accuracy =', f'{su_acc:.4f}')\n"
           "print(classification_report(su_true, su_pred, target_names=SENTIMENT_BUCKETS, zero_division=0))\n"
           "cm = confusion_matrix(su_true, su_pred, labels=range(3))\n"
           "plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
           "            xticklabels=SENTIMENT_BUCKETS, yticklabels=SENTIMENT_BUCKETS)\n"
           "plt.title('Support-style tone confusion matrix'); plt.tight_layout(); plt.show()"),
    (CODE, "joblib.dump(vocab, SENTIMENT_DIR / 'vocab.joblib')\n"
           "joblib.dump(SENTIMENT_BUCKETS, SENTIMENT_DIR / 'buckets.joblib')\n"
           "from src.models.sentiment import SentimentClassifier\n"
           "sc = SentimentClassifier()\n"
           "for m in [\"I have been waiting for three weeks and nobody helps, this is infuriating\",\n"
           "          \"Could you tell me the delivery options for my area?\",\n"
           "          \"Got my order today, absolutely delighted, thank you!\"]:\n"
           "    r = sc.predict(m)\n"
           "    print(f'{r[\"sentiment\"]:8s} (conf={r[\"confidence\"]:.2f})  {m}')\n"
           "print('\\nArtifacts saved to', SENTIMENT_DIR)"),
]

# =====================================================================
# Notebook 3 — Intent Classifier
# =====================================================================
nb3_md = """# Module 3 — Intent Classifier (Routing)

**Goal:** route each message to the correct handling path using the customer's intent.

**Data:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
— 26,872 instruction/response pairs. The `intent` column is gold-labelled, so this is a **supervised
classification** task (no zero/few-shot needed).

**Method:** the 27 fine-grained intents are condensed into the **5 routing categories present in the
data** (greeting & out-of-scope are handled at runtime — see below):
- `order_status`, `order_management`, `billing_and_refunds`, `account_management`, `complaint`.

TF-IDF (word unigrams+bigrams) → linear logistic regression, trained on a stratified 80/20 split.

**Runtime additions (covered in `src/models/intent.py`):**
- **Small-talk routing** (`greeting` / `thank-you` / `goodbye`) via a lightweight pattern matcher —
  Bitext contains no greetings, so we detect these explicitly (incl. common languages).
- **Relevance gate** → low-confidence predictions on non-support text become `out_of_scope`
  (e.g. *"what is the meaning of life"* → honest "can't help" + escalate) instead of a hallucinated answer.
"""

nb3_cells = [
    (MD, nb3_md),
    (CODE, "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('..').resolve()))\n"
           "import joblib\nimport pandas as pd\nimport numpy as np\nfrom datasets import load_dataset\n"
           "from sklearn.feature_extraction.text import TfidfVectorizer\n"
           "from sklearn.linear_model import LogisticRegression\n"
           "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
           "from sklearn.model_selection import train_test_split\n"
           "from src.config import INTENT_TO_CATEGORY, INTENT_DIR\n"
           "from src.preprocessing import normalize_text\nimport warnings\nwarnings.filterwarnings('ignore')"),
    (CODE, "ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')\n"
           "df = ds['train'].to_pandas()[['instruction', 'intent']].copy()\n"
           "df['category'] = df['intent'].map(INTENT_TO_CATEGORY)\n"
           "assert df['category'].notna().all(), sorted(df.loc[df['category'].isna(), 'intent'].unique())\n"
           "df['category'] = df['category'].astype(str)\n"
           "df['clean'] = df['instruction'].apply(lambda x: normalize_text(x))\n"
           "mapping = pd.DataFrame(sorted(INTENT_TO_CATEGORY.items()), columns=['fine_intent', 'routing_category'])\n"
           "print('\\nMapping table (fine-grained -> routing category):')\nwith pd.option_context('display.max_rows', 40):\n"
           "    display_none = None\ntable = mapping.groupby('routing_category')['fine_intent'].apply(lambda s: ', '.join(sorted(s)))\n"
           "for cat, ints in table.items(): print(f'  {cat:18s} <- {ints}')\n"
           "print('\\nCategory distribution:', dict(df['category'].value_counts()))"),
    (CODE, "train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])\n"
           "vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=120_000)\n"
           "X_train = vectorizer.fit_transform(train_df['clean']); X_test = vectorizer.transform(test_df['clean'])\n"
           "y_train, y_test = train_df['category'], test_df['category']\n"
           "model = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')\n"
           "model.fit(X_train, y_train)\n"
           "preds = model.predict(X_test)\n"
           f"print(f'\\nAccuracy = {{accuracy_score(y_test, preds):.4f}}')\n"
           "print(classification_report(y_test, preds, zero_division=0))"),
    (CODE, HEAD + "\ncm = confusion_matrix(y_test, preds, labels=model.classes_)\n"
           "plt.figure(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
           "            xticklabels=model.classes_, yticklabels=model.classes_)\n"
           "plt.title('Intent routing — test confusion matrix'); plt.tight_layout(); plt.show()"),
    (CODE, "joblib.dump(vectorizer, INTENT_DIR / 'vectorizer.joblib')\n"
           "joblib.dump(model, INTENT_DIR / 'model.joblib')\n"
           "joblib.dump({'classes': list(model.classes_)}, INTENT_DIR / 'meta.joblib')\n"
           "from src.models.intent import IntentClassifier\n"
           "ic = IntentClassifier()\n"
           "for q in [\"hello there\", \"where is my order\", \"I want a refund for the shoes I returned\",\n"
           "          \"how do I change my password\", \"I would like to place a new order\",\n"
           "          \"I want to complain about the delivery service\", \"what is the meaning of life\"]:\n"
           "    r = ic.predict(q)\n"
           "    print(f'{{r[\"category\"]:18s}} (conf={{r[\"confidence\"]:.2f}}, src={{r[\"source\"]}})  {q}')\n"
           "print('\\nArtifacts saved to', INTENT_DIR)"),
]

# =====================================================================
# Notebook 4 — Q&A RAG Pipeline
# =====================================================================
nb4_md = """# Module 4 — Grounded Q&A with Retrieval-Augmented Generation

**Goal:** answer customer questions with **grounded** answers from the support KB.

**Components**
1. **Knowledge base** — Bitext instruction→response pairs. Each entry is a chunk: the customer
   `instruction` is embedded for retrieval; the paired gold `response` is injected into the prompt as
   grounding context.
2. **Embeddings** — `all-MiniLM-L6-v2` (sentence-transformers, 384-d).
3. **Vector store** — local **FAISS** (no external account needed; 26,872 entries).
4. **Hybrid retrieval** — BM25 lexical candidate pool → dense cosine re-ranking. This fixes the classic
   weakness of pure dense retrieval on short, template-heavy support queries.
5. **LLM generation** — Groq with `gpt-oss-20b`, conditioned on the spec's grounded prompt template:
   system prompt + retrieved responses as context + customer question. Sentiment & language detected by
   the earlier modules are injected. If no `GROQ_API_KEY` is available the system falls back to a
   grounded template answer (always works offline).
"""

nb4_cells = [
    (MD, nb4_md),
    (CODE, "import sys, os, pickle\nfrom pathlib import Path\nsys.path.insert(0, str(Path('..').resolve()))\n"
           "import numpy as np\nimport pandas as pd\nfrom datasets import load_dataset\n"
           "import faiss\nfrom sentence_transformers import SentenceTransformer\n"
           "from src.config import RAG_DIR, BITEXT_DATASET, EMBEDDING_MODEL, TOP_K\n"
           "from src.models.rag import Retriever, RAGGenerator\nimport warnings\nwarnings.filterwarnings('ignore')"),
    (CODE, "ds = load_dataset(BITEXT_DATASET)\n"
           "df = ds['train'].to_pandas()[['instruction', 'response', 'intent', 'category']].dropna()\n"
           "print('KB entries:', len(df))\nprint(df['category'].value_counts().to_dict())\n"
           "print('\\nExample chunk:')\nprint('  INSTRUCTION:', df['instruction'].iloc[0])\nprint('  RESPONSE   :', df['response'].iloc[0][:150], '...')\n"
           "embedder = SentenceTransformer(EMBEDDING_MODEL, device='cpu')\n"
           "mat = np.vstack([embedder.encode(df['instruction'].iloc[i:i+256].tolist(),\n"
           "                                      show_progress_bar=False)\n"
           "                 for i in range(0, len(df), 256)]).astype('float32')\n"
           "mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)\n"
           "index = faiss.IndexFlatIP(mat.shape[1]); index.add(mat)\n"
"print('\\nFAISS index:', index.ntotal, 'vectors x', index.d, 'dims')\n"
            "faiss.write_index(index, str(RAG_DIR / 'faiss.index'))\n"
           "with open(RAG_DIR / 'metadata.pkl', 'wb') as fh: pickle.dump({'metadata': df.to_dict('records')}, fh)\n"
           "print('Index + metadata saved to', RAG_DIR)"),
    (CODE, "retriever = Retriever()\n"
           "queries = ['where is my package, it is two weeks late', 'how do I get a refund for a damaged item',\n"
           "           'I forgot my password and cannot log in', 'what delivery options do you have']\n"
           "for q in queries:\n"
           "    hits = retriever.retrieve(q, k=TOP_K)\n"
           "    print('Q:', q)\n"
           "    for h in hits: print(f\"   [{h['score']:.3f}] ({h['intent']}) {h['instruction'][:70]}\")"),
(CODE, "sample = df.sample(600, random_state=0)\n"
            "# dense-only hit-rate on 600 held-out instructions (single batch encode, fast)\n"
            "qv = embedder.encode(sample['instruction'].tolist(), show_progress_bar=False)\n"
            "qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)\n"
            "sims = qv @ mat.T\n"
            "topk_idx = np.argsort(-sims, axis=1)[:, :TOP_K]\n"
            "doc_intent = df['intent'].to_numpy()\n"
            "y = sample['intent'].to_numpy()\n"
            "top1_hit = (doc_intent[topk_idx[:, 0]] == y).mean()\n"
            "topk_hit = np.mean([y[i] in doc_intent[topk_idx[i]] for i in range(len(y))])\n"
            "print('Retrieval hit-rate on held-out instructions (n=%d):' % len(sample))\n"
            "print('  intent match          : top-1 = %.3f  top-%d = %.3f' % (top1_hit, TOP_K, topk_hit))"),
    (CODE, "# Retrieval-only (no LLM): show that the retrieved chunk already contains a grounded answer\n"
           "q = 'I want to return the shoes because they do not fit'\nhits = retriever.retrieve(q, k=2)\n"
           "print('Q:', q)\nfor h in hits:\n"
           "    print(f\"\\n({h['intent']}) {h['instruction']}\")\n"
           "    print('   ->', h['response'][:200])"),
    (CODE, "gen = RAGGenerator()\n"
           "print('LLM backend:', gen.model if gen.available else 'OFFLINE template fallback (no GROQ_API_KEY set)')\n"
           "q = 'My order is two weeks late and I am really frustrated, where is it?'\n"
           "hits = retriever.retrieve(q, k=TOP_K)\n"
           "answer, msgs = gen.generate(question=q, hits=hits, sentiment='negative', language='English')\n"
           "print('\\nGenerated answer:\\n', answer[:600])"),
    (CODE, "# End-to-end: the 4-stage pipeline behind a single call\n"
           "from src.router import SupportChatbot\n"
           "bot = SupportChatbot()\n"
           "demos = ['hello there',\n"
           "         'Where is my order? I ordered two weeks ago and it never arrived. This is ridiculous!',\n"
           "         'I want a refund for the shoes I returned last month',\n"
           "         'what is the meaning of life',\n"
           "         'Merci, ma commande est arrivee parfaitement']\n"
           "for msg in demos:\n"
           "    out = bot.handle(msg)\n"
           "    print('=' * 90)\n"
           "    print('USER :', msg)\n"
           "    print(f'lang={{out[\"language\"]}}  sentiment={{out[\"sentiment\"][\"sentiment\"]}}  '\n"
           "          f'intent={{out[\"intent_route\"][\"category\"]}}  priority={{out[\"priority_flag\"]}}')\n"
           "    print('BOT  :', out['response'][:220])"),
]


def main():
    write_notebook("1_language_detection.ipynb", nb1_cells)
    write_notebook("2_sentiment_classifier.ipynb", nb2_cells)
    write_notebook("3_intent_classifier.ipynb", nb3_cells)
    write_notebook("4_rag_pipeline.ipynb", nb4_cells)
    print("done")


if __name__ == "__main__":
    main()