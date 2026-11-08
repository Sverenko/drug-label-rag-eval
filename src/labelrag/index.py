"""Embed every chunk with Embed Pro and save a local vector index.

    python -m labelrag.index               # writes data/index/
    python -m labelrag.index --no-header   # writes data/index-no-header/, for comparison

Reads data/processed/sections.jsonl and embeds each chunk's header and text
(labelrag.chunk.embed_text) through the cached client, so a rebuild makes no API calls.
Writes vectors.npy (float32, one row per chunk), chunk_ids.json and meta.json.

5,730 vectors fit easily in memory, and exact cosine similarity over all of them takes
milliseconds, so the project doesn't need a vector database.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from labelrag import chunk, config
from labelrag.client import CachedCohere, stamp, utc_now

GROUP = 960  # texts per progress line: ten embed calls


@dataclass
class Index:
    vectors: np.ndarray  # (n, dimension) float32, each row unit length
    chunk_ids: list[str]
    meta: dict[str, Any]


def index_dir(with_header: bool = True) -> Path:
    return config.INDEX_DIR if with_header else config.DATA_DIR / "index-no-header"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    client: CachedCohere, chunks: list[dict[str, Any]], with_header: bool = True, progress=None
) -> np.ndarray:
    """Embed every chunk as a search document. Returns the vectors in chunk order."""
    texts = [chunk.embed_text(c, with_header) for c in chunks]
    parts = []
    for start in range(0, len(texts), GROUP):
        result = client.embed(
            texts[start : start + GROUP],
            model=config.EMBED_DOC_MODEL,
            input_type="search_document",
            dimension=config.EMBED_DIMENSION,
        )
        parts.append(result.vectors)
        if progress:
            progress(min(start + GROUP, len(texts)), len(texts))
    if not parts:
        return np.empty((0, config.EMBED_DIMENSION), dtype=np.float32)
    return np.vstack(parts)


def save(path: Path, vectors: np.ndarray, chunk_ids: list[str], meta: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", vectors.astype(np.float32))
    (path / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load(path: Path | None = None, sections_file: Path | None = None) -> Index:
    """Load an index, checking it matches the configured model and the current chunks."""
    path = path or index_dir()
    sections_file = sections_file or config.SECTIONS_FILE
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"No index in {path}. Run `python -m labelrag.index` first.")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if (meta["model"], meta["dimension"]) != (config.EMBED_DOC_MODEL, config.EMBED_DIMENSION):
        raise ValueError(
            f"The index in {path} was built with {meta['model']} at {meta['dimension']} "
            f"dimensions, but the config says {config.EMBED_DOC_MODEL} at "
            f"{config.EMBED_DIMENSION}. Rebuild it with `python -m labelrag.index`."
        )
    if sections_file.exists() and digest(sections_file) != meta["sections_sha256"]:
        raise ValueError(
            "The chunks have changed since the index was built. Rebuild it with "
            "`python -m labelrag.index`; unchanged chunks come from the cache."
        )
    vectors = np.load(path / "vectors.npy")
    chunk_ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
    if vectors.shape != (len(chunk_ids), meta["dimension"]):
        raise ValueError(f"{path} is inconsistent: {vectors.shape} vectors, {len(chunk_ids)} IDs")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return Index(vectors.astype(np.float32), chunk_ids, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed the chunks into a local vector index.")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="embed the chunk text without its drug-and-section header",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=100,
        help="stop before making more than this many new API calls (default 100)",
    )
    args = parser.parse_args(argv)
    if not config.SECTIONS_FILE.exists():
        print("No sections yet: run `python -m labelrag.chunk` first.")
        return 1
    with_header = not args.no_header
    chunks = chunk.load_chunks()
    client = CachedCohere(max_new_calls=args.max_calls)

    def progress(done: int, total: int) -> None:
        print(f"  embedded {done:,} of {total:,} chunks", flush=True)

    vectors = build(client, chunks, with_header, progress)
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "input_type": "search_document",
        "with_header": with_header,
        "chunks": len(chunks),
        "sections_sha256": digest(config.SECTIONS_FILE),
        "built_at": stamp(utc_now()),
    }
    out = index_dir(with_header)
    save(out, vectors, [c["chunk_id"] for c in chunks], meta)
    print(
        f"Indexed {len(chunks):,} chunks with {config.EMBED_DOC_MODEL} "
        f"({config.EMBED_DIMENSION} dimensions) into {out.relative_to(config.ROOT)}"
    )
    print(client.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
