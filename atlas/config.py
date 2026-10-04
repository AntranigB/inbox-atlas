"""Paths and env config shared by every module."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("ATLAS_DATA", ROOT / "data"))
DB_PATH = DATA / "mail.sqlite"
INDEX_DIR = DATA / "index"
MODELS = ROOT / "models"

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.20-0309-non-reasoning")
XAI_BASE = "https://api.x.ai/v1"

ENCODER = os.getenv("ATLAS_ENCODER", "base")
DIM = int(os.getenv("ATLAS_DIM") or 0) or None
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
SIDECAR_URL = os.getenv("SIDECAR_URL", "http://localhost:8766")


def env(name, default=""):
    return os.getenv(name, default)
