"""All tunables live here (override via environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# SOURCE_DIR can hold several roots separated by ";" (Windows) or ":" (Mac/Linux), e.g. /nas/Sanjay Sir:/nas/Priya
SOURCE_DIRS = [Path(p).resolve() for p in os.getenv("SOURCE_DIR", str(ROOT / "sample_photos")).split(os.pathsep) if p.strip()]
SOURCE_DIR = SOURCE_DIRS[0]  # first root (kept for compatibility)
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
ORGANISED_DIR = Path(os.getenv("ORGANISED_DIR", DATA_DIR / "Organised")).resolve()
DB_PATH = DATA_DIR / "index.db"
UNDO_LOG = DATA_DIR / "undo.log"
ROLES_FILE = ROOT / "roles.json"
PEOPLE_FILE = ROOT / "people.json"  # {"Display Name": ["alias", "filename token"]}

# --- metadata reader: "exiftool" (full) or built-in pure-Python fallback (auto if exiftool missing)
EXIFTOOL_PATH = os.getenv("EXIFTOOL_PATH", "exiftool")      # e.g. C:\\tools\\exiftool.exe (portable unzip is fine)
FORCE_FALLBACK = os.getenv("FORCE_FALLBACK", "0") == "1"    # test the built-in reader even if exiftool exists

# --- vision: VISION_PROVIDER = anthropic | openai  ("openai" = any OpenAI-compatible server: Ollama/vLLM/DashScope/LM Studio)
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "anthropic").lower()
VISION_MODEL = os.getenv("VISION_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "http://localhost:11434/v1")  # Ollama default
VISION_API_KEY = os.getenv("VISION_API_KEY", "ollama")
_ready = bool(ANTHROPIC_API_KEY) if VISION_PROVIDER == "anthropic" else bool(VISION_BASE_URL)
USE_VISION = _ready and os.getenv("USE_VISION", "1") == "1"
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")

# Folder decision thresholds (spec section 5)
THRESH_HIGH = float(os.getenv("THRESH_HIGH", 0.80))
THRESH_LOW = float(os.getenv("THRESH_LOW", 0.60))
REPEAT_FACE_ASK = 3  # DEFAULT N for unnamed face clusters (not used in MVP)

# One folder per person? PERSON_FOLDERS=child -> first-level folder under each root is a person ("Sanjay Sir/", "Yogesh Jain/")
#                         PERSON_FOLDERS=root  -> each SOURCE_DIR root itself is a person's folder
#                         PERSON_FOLDERS=off   -> only people.json aliases (default)
PERSON_FOLDERS = os.getenv("PERSON_FOLDERS", "off").lower()

EXCLUDE = [p for p in os.getenv("EXCLUDE", "").split(",") if p]  # folders to skip
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
DOC_EXT = {".pdf"}
SUPPORTED = IMAGE_EXT | DOC_EXT

DATA_DIR.mkdir(parents=True, exist_ok=True)
