import os
from pathlib import Path

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
STATIC_DIR = Path(__file__).parent / "static"
