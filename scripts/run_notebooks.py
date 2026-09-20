"""Execute all project notebooks in-place so deliverables contain real outputs."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
ORDER = ["1_language_detection.ipynb", "3_intent_classifier.ipynb",
         "2_sentiment_classifier.ipynb", "4_rag_pipeline.ipynb"]


def run_one(path, timeout=3600):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="nlpchat",
                            resources={"metadata": {"path": str(path.parent)}})
    print(f"EXECUTING {path.name}", flush=True)
    client.execute()
    nbformat.write(nb, str(path))
    print(f"DONE     {path.name}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    files = ORDER if which == "all" else [f for f in ORDER if which in f]
    for f in files:
        run_one(NB_DIR / f)