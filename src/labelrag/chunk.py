"""Split the labels into section chunks for embedding.

    python -m labelrag.chunk

Reads data/manifest.csv and data/raw/ and writes data/processed/sections.jsonl, one record
per chunk with its set_id, section, drug names and text.

Only the numbered sections of the prescribing information are used. openFDA also stores
some subsections as fields of their own (pregnancy, pharmacokinetics and so on) and every
table twice, as HTML and inside the plain text, so those would only repeat text. The
patient guides, carton text and distributor lines are left out too.

A section longer than CHUNK_CHARS is split at its numbered subsection headings (5.1, 5.2,
...), then at sentence ends, then at spaces, and small neighbouring pieces are packed back
together up to the limit. Every chunk carries a header naming the drug and the section;
the index stage decides whether to embed the header with the text.
"""

import json
import re
import statistics
import sys
from collections import Counter
from typing import Any

from labelrag import config, fetch

# The numbered sections of a label in the current format, in label order.
SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "dosage_forms_and_strengths": "Dosage Forms and Strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "drug_abuse_and_dependence": "Drug Abuse and Dependence",
    "overdosage": "Overdosage",
    "description": "Description",
    "clinical_pharmacology": "Clinical Pharmacology",
    "nonclinical_toxicology": "Nonclinical Toxicology",
    "clinical_studies": "Clinical Studies",
    "references": "References",
    "how_supplied": "How Supplied/Storage and Handling",
    "information_for_patients": "Patient Counseling Information",
}
LABEL_METADATA = {"set_id", "id", "version", "effective_time", "openfda"}


def clean(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def unique(names: list[str]) -> list[str]:
    """Drop repeats that differ only in case, keeping the first spelling."""
    seen: dict[str, str] = {}
    for name in names:
        if name:
            seen.setdefault(name.lower(), name)
    return list(seen.values())


def section_number(text: str) -> str | None:
    """The number a section's text opens with: "5" for "5 WARNINGS AND PRECAUTIONS ..."."""
    match = re.match(r"(\d{1,2})\.?\s+[A-Z]", text)
    return match.group(1) if match else None


def pack(pieces: list[str], limit: int) -> list[str]:
    """Join neighbouring pieces with spaces while they fit in `limit` characters."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def hard_wrap(text: str, limit: int) -> list[str]:
    """Split at spaces into pieces of at most `limit` characters (a word longer than that
    is cut, which real label text never needs)."""
    words: list[str] = []
    for word in text.split(" "):
        words.extend(word[i : i + limit] for i in range(0, len(word), limit))
    return pack(words, limit)


def split_unit(text: str, limit: int) -> list[str]:
    """Split at sentence ends and before bullets, then at spaces if a sentence is too long."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\s+(?=•)", text):
        pieces.extend(hard_wrap(sentence, limit) if len(sentence) > limit else [sentence])
    return pack(pieces, limit)


def split_section(text: str, limit: int = config.CHUNK_CHARS) -> list[str]:
    """Split one section into chunks of at most `limit` characters, breaking at numbered
    subsection headings first. Joining the chunks with spaces gives back the text."""
    if len(text) <= limit:
        return [text]
    number = section_number(text)
    if number:
        units = re.split(rf"(?<![\w.(])(?={number}\.\d{{1,2}}\s+[A-Z])", text)
    else:
        units = [text]
    pieces: list[str] = []
    for unit in units:
        if unit.strip():
            pieces.extend(split_unit(unit.strip(), limit))
    return pack(pieces, limit)


def chunk_label(
    label: dict[str, Any], row: dict[str, str], limit: int = config.CHUNK_CHARS
) -> list[dict[str, Any]]:
    """Chunk records for one label. `row` is the label's manifest row."""
    if label["set_id"] != row["set_id"]:
        raise ValueError(f"label {label['set_id']} doesn't match manifest row {row['set_id']}")
    brands = " / ".join(unique(row["brand_name"].split("; ")))
    generics = " / ".join(unique(row["generic_name"].split("; ")))
    chunks = []
    for section, title in SECTIONS.items():
        text = clean(label.get(section, []))
        if not text:
            continue
        parts = split_section(text, limit)
        for i, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": f"{label['set_id']}:{section}:{i}",
                    "set_id": label["set_id"],
                    "version": str(label.get("version", "")),
                    "section": section,
                    "section_title": title,
                    "part": i,
                    "parts": len(parts),
                    "brand_names": brands,
                    "generic_name": generics,
                    "theme": row["theme"],
                    "header": f"{brands} ({generics}): {title}",
                    "text": part,
                }
            )
    return chunks


def embed_text(chunk: dict[str, Any], with_header: bool = True) -> str:
    """What gets embedded for a chunk: the drug-and-section header, then the text."""
    return f"{chunk['header']}\n\n{chunk['text']}" if with_header else chunk["text"]


def load_chunks(path=None) -> list[dict[str, Any]]:
    path = path or config.SECTIONS_FILE
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    rows = fetch.read_manifest(config.MANIFEST)
    chunks: list[dict[str, Any]] = []
    left_out: Counter[str] = Counter()
    for row in rows:
        path = fetch.raw_path(config.RAW_DIR, row["set_id"])
        if not path.exists():
            print(f"Missing {path}. Run `python -m labelrag.fetch` first.")
            return 1
        label = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(chunk_label(label, row))
        left_out.update(k for k in label if k not in SECTIONS and k not in LABEL_METADATA)

    config.SECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.SECTIONS_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = sorted(len(c["text"]) for c in chunks)
    with_header = max(len(embed_text(c)) for c in chunks)
    per_label = Counter(c["set_id"] for c in chunks)
    print(f"{len(rows)} labels -> {len(chunks):,} chunks in data/processed/sections.jsonl")
    print(
        f"Chunks per label: median {statistics.median(per_label.values()):.0f}, "
        f"max {max(per_label.values())}"
    )
    print(
        f"Chunk text: median {statistics.median(lengths):,.0f} characters, "
        f"p95 {lengths[int(0.95 * (len(lengths) - 1))]:,}, max {lengths[-1]:,} "
        f"(limit {config.CHUNK_CHARS:,}); with the header, max {with_header:,}"
    )
    print(f"Embed calls to index them all: {-(-len(chunks) // 96)} (96 texts per call)")
    print("Chunks per section:")
    by_section = Counter(c["section"] for c in chunks)
    for section in SECTIONS:
        if by_section[section]:
            print(f"  {by_section[section]:>5}  {SECTIONS[section]}")
    print(
        f"Left out: {len(left_out)} other fields (subsection copies, tables, patient guides,"
        " carton text). Most common: " + ", ".join(f"{k} ({n})" for k, n in left_out.most_common(6))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
