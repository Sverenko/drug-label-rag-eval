"""The eval questions: their format, a validator, and helpers for writing them.

    python -m labelrag.questions                     # check eval/questions.jsonl
    python -m labelrag.questions find irbesartan     # matching labels, set_ids, DailyMed links
    python -m labelrag.questions show Avapro dosage_and_administration   # text as indexed

Each line of eval/questions.jsonl is one JSON object:

    {"id": "q01", "question": "...", "answerable": true,
     "gold": [{"set_id": "...", "section": "dosage_and_administration"}],
     "answer": "...", "key_facts": ["..."], "tags": ["near_duplicate"]}

Answerable questions need at least one gold location (a label's set_id plus a section key
from labelrag.chunk.SECTIONS), a reference answer and at least one key fact. Unanswerable
questions set "answerable": false and have no gold locations. "tags" and "notes" are
optional.

The checks that need the label text (does each gold section exist, do the key facts appear
in it word for word, do they also appear somewhere else) run only once
data/processed/sections.jsonl has been built, so CI skips them.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labelrag import chunk, config, fetch

KNOWN_TAGS = {"brand_vs_generic", "near_duplicate", "multi_section"}
FIELDS = {"id", "question", "answerable", "gold", "answer", "key_facts", "tags", "notes"}
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={}"
SHOWN_ELSEWHERE = 4
# Words in a generic name that don't identify the active ingredient.
NOT_INGREDIENTS = {
    "and", "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "citrate",
    "besylate", "maleate", "phosphate", "tartrate", "succinate", "mesylate", "medoxomil",
    "cilexetil", "fumarate", "propanediol", "dihydrate", "monohydrate", "anhydrous", "benzoate",
    "pidolate", "tromethamine", "etexilate", "acid", "ethyl", "extended", "release", "oral",
    "tablets", "injection", "kit",
}  # fmt: skip

Location = tuple[str, str]  # (set_id, section)


@dataclass
class Report:
    questions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corpus_checked: bool = False


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def section_texts(chunks: list[dict[str, Any]]) -> dict[Location, str]:
    """Each label section's full text, rebuilt from its chunks in order."""
    parts: dict[Location, list[tuple[int, str]]] = defaultdict(list)
    for c in chunks:
        parts[(c["set_id"], c["section"])].append((c["part"], c["text"]))
    return {loc: " ".join(text for _, text in sorted(pieces)) for loc, pieces in parts.items()}


def is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def gold_locations(q: dict[str, Any]) -> list[Location]:
    gold = q.get("gold") if isinstance(q.get("gold"), list) else []
    return [
        (g["set_id"], g["section"])
        for g in gold
        if isinstance(g, dict) and is_text(g.get("set_id")) and is_text(g.get("section"))
    ]


def field_errors(q: dict[str, Any], where: str) -> list[str]:
    errors = []
    for name in ("id", "question"):
        if not is_text(q.get(name)):
            errors.append(f"{where}: missing '{name}'")
    if not isinstance(q.get("answerable"), bool):
        errors.append(f"{where}: 'answerable' should be true or false")
    tags = q.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        errors.append(f"{where}: 'tags' should be a list of strings")
    if q.get("answerable") is False:
        if q.get("gold"):
            errors.append(f"{where}: an unanswerable question can't have gold locations")
        return errors
    if q.get("answerable") is not True:
        return errors
    gold = q.get("gold")
    if not isinstance(gold, list) or not gold:
        errors.append(f"{where}: an answerable question needs at least one gold location")
    elif len(gold_locations(q)) != len(gold):
        errors.append(f"{where}: each gold location needs a 'set_id' and a 'section'")
    if not is_text(q.get("answer")):
        errors.append(f"{where}: missing 'answer'")
    facts = q.get("key_facts")
    if not isinstance(facts, list) or not facts or not all(is_text(f) for f in facts):
        errors.append(f"{where}: 'key_facts' should be a list of one or more phrases")
    return errors


def field_warnings(q: dict[str, Any], where: str) -> list[str]:
    warnings = [f"{where}: unknown field '{name}'" for name in sorted(set(q) - FIELDS)]
    tags = q.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag not in KNOWN_TAGS:
                warnings.append(
                    f"{where}: unfamiliar tag '{tag}' (known: {', '.join(sorted(KNOWN_TAGS))})"
                )
    return warnings


def gold_errors(
    q: dict[str, Any], where: str, names: dict[str, str], texts: dict[Location, str] | None
) -> list[str]:
    errors = []
    for set_id, section in gold_locations(q):
        if set_id not in names:
            errors.append(f"{where}: set_id {set_id} isn't in the manifest")
        elif section not in chunk.SECTIONS:
            keys = ", ".join(chunk.SECTIONS)
            errors.append(f"{where}: '{section}' isn't a section key. Keys: {keys}")
        elif texts is not None and (set_id, section) not in texts:
            errors.append(f"{where}: {names[set_id]} has no {chunk.SECTIONS[section]} section")
    return errors


def ingredient_words(generic_name: str) -> set[str]:
    """The words that name active ingredients: IRBESARTAN AND HYDROCHLOROTHIAZIDE gives
    {"irbesartan", "hydrochlorothiazide"}."""
    return set(re.findall(r"[a-z]+", generic_name.lower())) - NOT_INGREDIENTS


def fact_warnings(
    q: dict[str, Any],
    where: str,
    names: dict[str, str],
    ingredients: dict[str, set[str]],
    texts: dict[Location, str],
) -> list[str]:
    """Key facts that aren't in the gold text word for word, and sections that hold every
    key fact but aren't listed as gold: other sections of a gold label, or sections of a
    label that shares an active ingredient with one (Avapro and Avalide both carry
    irbesartan, so much of their text is the same)."""
    gold = gold_locations(q)
    facts = [f for f in q.get("key_facts") or [] if is_text(f)]
    if not facts or not gold or any(loc not in texts for loc in gold):
        return []
    warnings = []
    for fact in facts:
        if not any(normalize(fact) in texts[loc] for loc in gold):
            warnings.append(
                f"{where}: key fact {fact!r} doesn't appear word for word in the gold sections"
                " (`python -m labelrag.questions show` prints them as indexed)"
            )
    gold_labels = {set_id for set_id, _ in gold}
    gold_ingredients = set().union(*(ingredients.get(set_id, set()) for set_id in gold_labels))

    def related(set_id: str) -> bool:
        return set_id in gold_labels or bool(ingredients.get(set_id, set()) & gold_ingredients)

    wanted = [normalize(f) for f in facts]
    elsewhere = [
        loc
        for loc, text in texts.items()
        if loc not in gold and related(loc[0]) and all(fact in text for fact in wanted)
    ]
    if elsewhere:
        shown = ", ".join(
            f"{names.get(set_id, set_id)} {chunk.SECTIONS[section]}"
            for set_id, section in elsewhere[:SHOWN_ELSEWHERE]
        )
        more = (
            f" and {len(elsewhere) - SHOWN_ELSEWHERE} more"
            if len(elsewhere) > SHOWN_ELSEWHERE
            else ""
        )
        warnings.append(
            f"{where}: every key fact also appears in {shown}{more}. If those sections answer"
            " the question too, add them as gold locations."
        )
    return warnings


def validate(
    path: Path | None = None,
    manifest_path: Path | None = None,
    sections_path: Path | None = None,
) -> Report:
    path = path or config.QUESTIONS_FILE
    manifest_path = manifest_path or config.MANIFEST
    sections_path = sections_path or config.SECTIONS_FILE
    rows = fetch.read_manifest(manifest_path)
    names = {row["set_id"]: row["brand_name"] for row in rows}
    ingredients = {row["set_id"]: ingredient_words(row["generic_name"]) for row in rows}
    texts = None
    if sections_path.exists():
        raw = section_texts(chunk.load_chunks(sections_path))
        texts = {loc: normalize(text) for loc, text in raw.items()}
    report = Report(corpus_checked=texts is not None)
    ids: Counter[str] = Counter()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as err:
            report.errors.append(f"line {n}: not valid JSON ({err.msg})")
            continue
        if not isinstance(q, dict):
            report.errors.append(f"line {n}: expected a JSON object")
            continue
        where = f"line {n} ({q.get('id') or 'no id'})"
        report.questions.append(q)
        if is_text(q.get("id")):
            ids[q["id"]] += 1
        report.errors.extend(field_errors(q, where))
        report.warnings.extend(field_warnings(q, where))
        if q.get("answerable") is True:
            report.errors.extend(gold_errors(q, where, names, texts))
            if texts is not None:
                report.warnings.extend(fact_warnings(q, where, names, ingredients, texts))
    for qid, count in ids.items():
        if count > 1:
            report.errors.append(f"id {qid!r} is used {count} times")
    return report


def find(term: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Manifest rows whose set_id, brand or generic name matches; exact brand matches win."""
    term = term.strip().lower()
    matches = [
        row
        for row in rows
        if term == row["set_id"]
        or term in row["brand_name"].lower()
        or term in row["generic_name"].lower()
    ]
    exact = [row for row in matches if term in (b.lower() for b in row["brand_name"].split("; "))]
    return exact or matches


def cmd_check() -> int:
    if not config.QUESTIONS_FILE.exists():
        print("No eval/questions.jsonl yet. The README's Eval set section describes the format.")
        return 1
    report = validate()
    answerable = sum(q.get("answerable") is True for q in report.questions)
    print(
        f"{len(report.questions)} questions: {answerable} answerable, "
        f"{len(report.questions) - answerable} not"
    )
    tags = Counter(t for q in report.questions for t in q.get("tags") or [] if isinstance(t, str))
    if tags:
        print("Tags: " + ", ".join(f"{tag} {n}" for tag, n in tags.most_common()))
    if not report.corpus_checked:
        print("Skipped the corpus checks: run `python -m labelrag.chunk` to build the sections.")
    for error in report.errors:
        print(f"  error: {error}")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    if not report.errors and not report.warnings:
        print("No problems found.")
    return 1 if report.errors else 0


def cmd_find(term: str) -> int:
    matches = find(term, fetch.read_manifest(config.MANIFEST))
    if not matches:
        print(f"No label matches {term!r}.")
        return 1
    texts = section_texts(chunk.load_chunks()) if config.SECTIONS_FILE.exists() else {}
    for row in matches:
        print(f"{row['brand_name']}  ({row['generic_name']}; {row['theme']})")
        print(f"  set_id    {row['set_id']}  (version {row['version']}, {row['effective_time']})")
        print(f"  DailyMed  {DAILYMED.format(row['set_id'])}")
        present = [s for s in chunk.SECTIONS if (row["set_id"], s) in texts]
        if present:
            print(f"  sections  {', '.join(present)}")
    return 0


def cmd_show(term: str, section: str | None) -> int:
    matches = find(term, fetch.read_manifest(config.MANIFEST))
    if len(matches) != 1:
        found = "; ".join(f"{r['brand_name']} {r['set_id']}" for r in matches) or "nothing"
        print(f"{term!r} needs to match exactly one label (found: {found}). Try its set_id.")
        return 1
    if not config.SECTIONS_FILE.exists():
        print("No sections yet: run `python -m labelrag.chunk` first.")
        return 1
    row = matches[0]
    texts = section_texts(chunk.load_chunks())
    if section is None:
        print(f"{row['brand_name']} ({row['set_id']}) sections:")
        for key, title in chunk.SECTIONS.items():
            if (row["set_id"], key) in texts:
                print(f"  {key:<30} {title} ({len(texts[(row['set_id'], key)]):,} characters)")
        return 0
    if (row["set_id"], section) not in texts:
        print(f"{row['brand_name']} has no section {section!r}. Leave it out to list them.")
        return 1
    print(texts[(row["set_id"], section)])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the eval questions, or look up labels while writing them."
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("check", help="check eval/questions.jsonl (the default)")
    finder = commands.add_parser("find", help="labels whose brand or generic name matches")
    finder.add_argument("term")
    shower = commands.add_parser("show", help="a label's section text, exactly as indexed")
    shower.add_argument("label", help="brand name, generic name or set_id")
    shower.add_argument("section", nargs="?", help="section key; leave it out to list them")
    args = parser.parse_args(argv)
    if args.command == "find":
        return cmd_find(args.term)
    if args.command == "show":
        return cmd_show(args.label, args.section)
    return cmd_check()


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
