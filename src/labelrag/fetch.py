"""Choose the corpus on openFDA, pin it in data/manifest.csv, and download the labels.

    python -m labelrag.fetch --select   # run the selection and write a new manifest
    python -m labelrag.fetch            # download the manifest's labels, warning on new versions

The selection keeps prescription labels that have an openfda block, come from the original
packager, sit under an NDA or BLA, have an indications section in the current labeling
format, and contain an active ingredient from one of THEMES. Labels that share an
application number count as one product, and only one of them is kept: a brand-named
label first, then the most recently updated.

Raw label JSON goes to data/raw/, which stays out of git. openFDA only serves the current
version of each label, so a re-fetch warns when a label has changed since it was pinned.
"""

import argparse
import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from labelrag import config
from labelrag.openfda import OpenFDA

# Active ingredients by theme, as words matched against openFDA's substance and generic
# names, so every salt form counts. Some have no current innovator label; they stay so the
# rule reads as written.
THEMES = {
    "Diabetes": [
        "metformin", "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
        "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "bexagliflozin",
        "sotagliflozin", "semaglutide", "liraglutide", "dulaglutide", "exenatide",
        "lixisenatide", "tirzepatide", "pioglitazone", "rosiglitazone", "glipizide",
        "glimepiride", "glyburide", "repaglinide", "nateglinide", "acarbose", "miglitol",
        "pramlintide",
    ],
    "Blood thinners": [
        "apixaban", "rivaroxaban", "edoxaban", "dabigatran", "warfarin", "clopidogrel",
        "prasugrel", "ticagrelor", "vorapaxar",
    ],
    "Cholesterol": [
        "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
        "fluvastatin", "pitavastatin", "ezetimibe", "bempedoic", "icosapent", "evolocumab",
        "alirocumab", "inclisiran", "fenofibrate", "fenofibric",
    ],
    "Urology": [
        "sildenafil", "tadalafil", "vardenafil", "avanafil", "finasteride", "dutasteride",
        "tamsulosin", "alfuzosin", "silodosin", "doxazosin", "terazosin", "mirabegron",
        "vibegron", "solifenacin", "darifenacin", "fesoterodine", "tolterodine", "oxybutynin",
        "trospium",
    ],
    "Blood pressure (ARBs)": [
        "losartan", "valsartan", "irbesartan", "candesartan", "olmesartan", "telmisartan",
        "azilsartan", "eprosartan",
    ],
    "Migraine": [
        "sumatriptan", "rizatriptan", "zolmitriptan", "eletriptan", "naratriptan",
        "almotriptan", "frovatriptan", "ubrogepant", "rimegepant", "atogepant", "zavegepant",
        "lasmiditan", "erenumab", "fremanezumab", "galcanezumab", "eptinezumab",
    ],
}  # fmt: skip

RX = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
ORIGINAL = "openfda.is_original_packager:true"
INNOVATOR = "(openfda.application_number:NDA* openfda.application_number:BLA*)"

# Words that don't make a name a brand: "Fluvastatin Sodium ER" is still a generic name.
NAME_STOP = {
    "and", "er", "xr", "sr", "cr", "la", "xl", "odt", "dr", "extended", "release", "delayed",
    "tablets", "tablet", "capsules", "capsule", "injection", "oral", "solution", "suspension",
    "for", "film", "coated", "hcl", "kit", "usp",
}  # fmt: skip

MANIFEST_FIELDS = [
    "theme",
    "brand_name",
    "generic_name",
    "manufacturer",
    "application_numbers",
    "set_id",
    "version",
    "effective_time",
]
SET_IDS_PER_REQUEST = 25


def theme_query(words: list[str]) -> str:
    names = " ".join(f"openfda.substance_name:{w} openfda.generic_name:{w}" for w in words)
    return f"{RX} AND {ORIGINAL} AND {INNOVATOR} AND ({names})"


def fda(label: dict[str, Any]) -> dict[str, Any]:
    return label.get("openfda") or {}


def name_words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower())) - NAME_STOP


def is_brand_named(label: dict[str, Any]) -> bool:
    """True if a brand name is more than the generic name's own words."""
    generic: set[str] = set()
    for name in fda(label).get("generic_name", []) + fda(label).get("substance_name", []):
        generic |= name_words(name)
    return any(
        name_words(b) and not name_words(b) <= generic for b in fda(label).get("brand_name", [])
    )


def brand_names(label: dict[str, Any]) -> str:
    """Every brand name on the label, joined with "; ". One label can cover several brands:
    the oral semaglutide label covers both RYBELSUS and OZEMPIC tablets."""
    names: dict[str, str] = {}
    for name in fda(label).get("brand_name", []):
        names.setdefault(name.lower(), name)
    return "; ".join(names.values())


def current_format(label: dict[str, Any]) -> bool:
    """Has an indications section and the current labeling format's Warnings and Precautions."""
    indications = "".join(label.get("indications_and_usage", [])).strip()
    return bool(indications) and "warnings_and_cautions" in label


def one_per_approval(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group labels that share any application number and keep one from each group."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def apps(label: dict[str, Any]) -> list[str]:
        return fda(label).get("application_number") or [label["set_id"]]

    for label in labels:
        first, *rest = apps(label)
        for other in rest:
            parent[find(other)] = find(first)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        groups[find(apps(label)[0])].append(label)
    return [
        min(
            group,
            key=lambda lb: (
                not is_brand_named(lb),
                -int(lb.get("effective_time") or 0),
                lb["set_id"],
            ),
        )
        for group in groups.values()
    ]


def select(api: OpenFDA) -> tuple[list[dict[str, Any]], dict[str, set[str]], list[tuple[str, int]]]:
    """Run the selection. Returns the kept labels, each label's themes, and the funnel."""
    funnel = [
        ("labels on openFDA", api.total()),
        ("with an openfda block", api.total("_exists_:openfda")),
        ("prescription", api.total(RX)),
        ("from the original packager", api.total(f"{RX} AND {ORIGINAL}")),
        ("under an NDA or BLA", api.total(f"{RX} AND {ORIGINAL} AND {INNOVATOR}")),
    ]
    found: dict[str, dict[str, Any]] = {}
    themes: dict[str, set[str]] = defaultdict(set)
    for theme, words in THEMES.items():
        for label in api.search_all(theme_query(words)):
            found[label["set_id"]] = label
            themes[label["set_id"]].add(theme)
    funnel.append(("with an ingredient from the six themes", len(found)))
    current = [label for label in found.values() if current_format(label)]
    funnel.append(("with indications, in the current format", len(current)))
    kept = one_per_approval(current)
    funnel.append(("one label per approval", len(kept)))
    return kept, themes, funnel


def manifest_row(label: dict[str, Any], themes: set[str]) -> dict[str, str]:
    return {
        "theme": "; ".join(t for t in THEMES if t in themes),
        "brand_name": brand_names(label),
        "generic_name": "; ".join(fda(label).get("generic_name", [])),
        "manufacturer": (fda(label).get("manufacturer_name") or [""])[0],
        "application_numbers": "; ".join(fda(label).get("application_number", [])),
        "set_id": label["set_id"],
        "version": str(label.get("version", "")),
        "effective_time": str(label.get("effective_time", "")),
    }


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    order = {theme: i for i, theme in enumerate(THEMES)}

    def sort_key(row: dict[str, str]) -> tuple:
        first_theme = row["theme"].split("; ")[0]
        return order.get(first_theme, len(order)), row["brand_name"].lower(), row["set_id"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=sort_key))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def raw_path(raw_dir: Path, set_id: str) -> Path:
    return raw_dir / f"{set_id}.json"


def save_raw(label: dict[str, Any], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path(raw_dir, label["set_id"]).write_text(
        json.dumps(label, ensure_ascii=False), encoding="utf-8"
    )


def refetch(api: OpenFDA, rows: list[dict[str, str]], raw_dir: Path) -> tuple[int, list[str]]:
    """Download the manifest's labels. Keeps a local copy that matches the pinned version,
    so the corpus doesn't drift. Returns how many were downloaded, and warnings."""
    current: dict[str, dict[str, Any]] = {}
    for batch in itertools.batched([row["set_id"] for row in rows], SET_IDS_PER_REQUEST):
        query = "(" + " ".join(f'set_id:"{set_id}"' for set_id in batch) + ")"
        for label in api.search_all(query):
            current[label["set_id"]] = label
    downloaded, warnings = 0, []
    for row in rows:
        name, set_id, pinned = row["brand_name"], row["set_id"], row["version"]
        local = raw_path(raw_dir, set_id)
        local_version = (
            json.loads(local.read_text(encoding="utf-8")).get("version") if local.exists() else None
        )
        label = current.get(set_id)
        if label is None:
            kept = "kept the local copy" if local_version else "no copy to fall back on"
            warnings.append(f"{name} ({set_id}) is no longer on openFDA; {kept}")
            continue
        served = str(label.get("version", ""))
        if served != pinned:
            change = f"the manifest pins version {pinned}, openFDA now serves {served}"
            if str(local_version) == pinned:
                warnings.append(f"{name}: {change}; kept your pinned copy")
                continue
            warnings.append(f"{name}: {change}; downloaded the new version, so answers may differ")
        elif str(local_version) == pinned:
            continue
        save_raw(label, raw_dir)
        downloaded += 1
    return downloaded, warnings


def print_funnel(funnel: list[tuple[str, int]]) -> None:
    previous = None
    for description, count in funnel:
        removed = f"  ({previous - count:,} removed)" if previous is not None else ""
        print(f"  {count:>9,}  {description}{removed}")
        previous = count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the drug label corpus from openFDA.")
    parser.add_argument(
        "--select",
        action="store_true",
        help="run the selection queries and write a new data/manifest.csv",
    )
    args = parser.parse_args(argv)
    api = OpenFDA()

    if args.select:
        kept, themes, funnel = select(api)
        rows = [manifest_row(label, themes[label["set_id"]]) for label in kept]
        for label in kept:
            save_raw(label, config.RAW_DIR)
        write_manifest(rows, config.MANIFEST)
        print("Selection funnel:")
        print_funnel(funnel)
        print("Labels per theme:")
        for theme in THEMES:
            print(f"  {sum(theme in row['theme'].split('; ') for row in rows):>4}  {theme}")
        print(f"Wrote {len(rows)} labels to data/manifest.csv and their JSON to data/raw/.")
    else:
        if not config.MANIFEST.exists():
            print("No manifest yet. Run `python -m labelrag.fetch --select` to choose the corpus.")
            return 1
        rows = read_manifest(config.MANIFEST)
        downloaded, warnings = refetch(api, rows, config.RAW_DIR)
        print(f"{len(rows)} labels in the manifest; downloaded {downloaded}.")
        for warning in warnings:
            print(f"  warning: {warning}")
        if not warnings:
            print("Every label matches its pinned version.")
    print(f"openFDA requests: {api.requests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
