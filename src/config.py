"""Paths, constants and a tiny .env loader (no extra dependency)."""
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 20)  # the dataset's fixed reference date (KB-00.2)
DATASET_NAME = "Capstone_Project_Dataset_FTTT_2026"


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()


def dataset_dir() -> Path:
    """Locate the extracted dataset: $DATASET_DIR, ./<name>, or next to the repo."""
    candidates = [os.environ.get("DATASET_DIR"), ROOT / DATASET_NAME,
                  ROOT.parent / DATASET_NAME, Path.cwd() / DATASET_NAME]
    for c in candidates:
        if c and (Path(c) / "kb").is_dir() and (Path(c) / "data").is_dir():
            return Path(c).resolve()
    raise FileNotFoundError(
        f"Dataset not found. Extract {DATASET_NAME}.zip next to this repo or set DATASET_DIR.")


DATASET = dataset_dir()
KB_DIR = DATASET / "kb"
DATA_DIR = DATASET / "data"
PRACTICE_DIR = DATASET / "practice"

# The provided mock tools live inside the dataset; import them from there.
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))

TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.environ.get("GNANI_TTS_VOICE", "Kaveri")
TTS_VOICE_HINGLISH = os.environ.get("GNANI_TTS_VOICE_HINGLISH", "Poorvi")
STT_LANGUAGE = os.environ.get("GNANI_STT_LANGUAGE", "en-IN")
