"""Paths and settings shared by every pipeline stage.

The Cohere key comes from the CO_API_KEY environment variable or, failing that, the
repo's .env file (copy .env.example to start). Run `python -m labelrag.config` to check
the setup without making an API call.
"""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SECTIONS_FILE = PROCESSED_DIR / "sections.jsonl"
INDEX_DIR = DATA_DIR / "index"
SAMPLE_DIR = DATA_DIR / "sample"
MANIFEST = DATA_DIR / "manifest.csv"
CACHE_DIR = ROOT / ".cache"
CACHE_DB = CACHE_DIR / "cohere.sqlite"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"

KEYS_PAGE = "https://dashboard.cohere.com/api-keys"

# Longest chunk of label text, in characters: about 500 tokens, far under the embed
# models' 8,192-token input limit.
CHUNK_CHARS = 2000

# Trial-key limits from docs.cohere.com/docs/rate-limits (checked 2026-11-08).
MONTHLY_CALL_BUDGET = 1000
EMBED_INPUTS_PER_MINUTE = 2000
RERANK_CALLS_PER_MINUTE = 10
CHAT_CALLS_PER_MINUTE = 20  # per model

# Cohere models, checked against the trial key by `python -m labelrag.smoke`.
# Embed Pro and Fast share one embedding space, so label sections are indexed once
# with Pro and questions are embedded with the faster model.
EMBED_DOC_MODEL = "embed-v5.0-pro"
EMBED_QUERY_MODEL = "embed-v5.0-fast"
EMBED_DIMENSION = 2048  # the models' default; they also offer 256 to 1536
RERANK_MODEL = "rerank-v5.0-pro"
RERANK_MODEL_FAST = "rerank-v5.0-fast"
CHAT_MODEL = "command-a-plus-05-2026"


class MissingAPIKeyError(RuntimeError):
    """CO_API_KEY isn't set in the environment or in .env."""


def api_key(env_file: Path | None = None) -> str:
    """Return the Cohere key. The environment variable wins over the .env file."""
    env_file = env_file or ENV_FILE
    key = os.environ.get("CO_API_KEY") or dotenv_values(env_file).get("CO_API_KEY") or ""
    key = key.strip()
    if not key:
        raise MissingAPIKeyError(
            "CO_API_KEY isn't set. Copy .env.example to .env and put your Cohere key after "
            f"CO_API_KEY= (keys are at {KEYS_PAGE}), or set the environment variable."
        )
    return key


def main() -> int:
    """Report whether setup is complete. Never prints the key itself."""
    print(f"Repo root: {ROOT}")
    print(f".env file: {'found' if ENV_FILE.exists() else 'not found'}", flush=True)
    try:
        api_key()
    except MissingAPIKeyError as err:
        print(f"CO_API_KEY: missing\n\n{err}", file=sys.stderr)
        return 1
    print("CO_API_KEY: set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
