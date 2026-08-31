import json

from pathlib import Path
from decouple import config


ROOT_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = ROOT_DIR / ".cache"

INDEXES_DIR = CACHE_DIR / "indexes"

RESOURCES_DIR = ROOT_DIR / "resources"

with open(RESOURCES_DIR / "book_aliases.json", encoding="utf-8") as f:
    BOOK_ALIASES = json.load(f)

with open(RESOURCES_DIR / "book_names.json", encoding="utf-8") as f:
    BOOK_NAMES = json.load(f)

with open(RESOURCES_DIR / "bible_verses.json", encoding="utf-8") as f:
    VERSE_COUNTS = json.load(f)

GEMINI_API_KEY = config("GEMINI_API_KEY", default=None)
