import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DA3_CACHE_DIR = PROJECT_ROOT / "models" / "depth-anything-3"


def get_da3_cache_dir():
    cache_dir = os.environ.get("DA3_CACHE_DIR", str(DEFAULT_DA3_CACHE_DIR))
    return str(Path(cache_dir).expanduser().resolve())
