import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("DOCU_DATA", str(ROOT / "data"))).resolve()
MODELS = Path(os.getenv("DOCU_MODELS", str(ROOT / "models"))).resolve()
OFFLINE = os.getenv("DOCU_OFFLINE", "true").lower() == "true"
DEVICE = os.getenv("DOCU_DEVICE", "cpu")
MAX_MB = int(os.getenv("DOCU_MAX_MB", "20"))
MAX_STATEMENTS = int(os.getenv("DOCU_MAX_STATEMENTS", "2000"))
MAX_PAIRS = int(os.getenv("DOCU_MAX_PAIRS", "10000"))
NLI_ID = os.getenv("DOCU_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
for folder in (DATA, DATA / "uploads", DATA / "results", DATA / "datasets", MODELS):
    folder.mkdir(parents=True, exist_ok=True)
