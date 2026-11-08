import json
import re

import httpx
import pytest

from labelrag import fetch
from labelrag.openfda import OpenFDA


def make_label(
    set_id,
    apps,
    *,
    brand=("Brand",),
    generic=("GENERIC",),
    substances=("GENERIC",),
    effective="20250101",
    version="1",
    product=None,
    indications=True,
    current=True,
):
    label = {
        "set_id": set_id,
        "version": version,
        "effective_time": effective,
        "openfda": {
            "application_number": list(apps),
            "brand_name": list(brand),
            "generic_name": list(generic),
            "substance_name": list(substances),
            "manufacturer_name": ["Maker"],
        },
        "spl_product_data_elements": [product or " ".join(brand + generic)],
    }
    if indications:
        label["indications_and_usage"] = ["1 INDICATIONS AND USAGE ..."]
    if current:
        label["warnings_and_cautions"] = ["5 WARNINGS AND PRECAUTIONS ..."]
    return label


class FakeAPI:
    """Answers the theme queries and set_id lookups from canned labels."""

    def __init__(self, labels_by_theme=None, by_set_id=None):
        self.labels_by_theme = labels_by_theme or {}
        self.by_set_id = by_set_id or {}
        self.searches: list[str] = []
        self.requests = 0

    def total(self, search=None):
        return 1000 if search is None else 500

    def search_all(self, search):
        self.searches.append(search)
        for theme, words in fetch.THEMES.items():
            if search == fetch.theme_query(words):
                return self.labels_by_theme.get(theme, [])
        ids = re.findall(r'set_id:"([^"]+)"', search)
        return [self.by_set_id[i] for i in ids if i in self.by_set_id]


def test_theme_query_matches_each_ingredient_on_both_name_fields():
    query = fetch.theme_query(["metformin", "sitagliptin"])
    assert query.startswith(f"{fetch.RX} AND {fetch.ORIGINAL} AND {fetch.INNOVATOR} AND (")
    assert query.endswith(
        "(openfda.substance_name:metformin openfda.generic_name:metformin"
        " openfda.substance_name:sitagliptin openfda.generic_name:sitagliptin)"
    )


@pytest.mark.parametrize(
    ("brand", "generic", "expected"),
    [
        ("Vybrique", "SILDENAFIL", True),
        ("Soliqua 100/33", "INSULIN GLARGINE AND LIXISENATIDE", True),
        ("Sildenafil", "SILDENAFIL CITRATE", False),
        ("FLUVASTATIN SODIUM ER", "FLUVASTATIN SODIUM", False),
        (
            "Sitagliptin and Metformin Hydrochloride",
            "SITAGLIPTIN AND METFORMIN HYDROCHLORIDE",
            False,
        ),
    ],
)
def test_a_brand_name_is_more_than_the_generic_words(brand, generic, expected):
    label = make_label("s", ["NDA1"], brand=(brand,), generic=(generic,), substances=())
    assert fetch.is_brand_named(label) is expected


def test_every_brand_name_on_a_label_is_kept_once():
    semaglutide = make_label("s", ["NDA213051"], brand=("OZEMPIC", "RYBELSUS"))
    xarelto = make_label("x", ["NDA022406"], brand=("Xarelto", "XARELTO"))
    assert fetch.brand_names(semaglutide) == "OZEMPIC; RYBELSUS"
    assert fetch.brand_names(xarelto) == "Xarelto"


def test_stubs_and_older_format_labels_are_dropped():
    assert fetch.current_format(make_label("a", ["NDA1"]))
    assert not fetch.current_format(make_label("b", ["NDA1"], indications=False))
    assert not fetch.current_format(make_label("c", ["NDA1"], current=False))


def test_one_label_per_approval_prefers_the_brand_then_the_newest():
    sildenafil = {"generic": ("SILDENAFIL CITRATE",), "substances": ("SILDENAFIL CITRATE",)}
    labels = [
        make_label(
            "viagra-old", ["NDA020895"], brand=("Viagra",), effective="20171215", **sildenafil
        ),
        make_label(
            "viagra-new", ["NDA020895"], brand=("Viagra",), effective="20231117", **sildenafil
        ),
        make_label(
            "authorized-generic",
            ["NDA020895"],
            brand=("Sildenafil Citrate",),
            effective="20240829",
            **sildenafil,
        ),
        make_label("revatio", ["NDA021845", "NDA022473"], brand=("Revatio",), **sildenafil),
        make_label(
            "revatio-iv", ["NDA022473"], brand=("Revatio",), effective="20200101", **sildenafil
        ),
    ]
    kept = fetch.one_per_approval(labels)
    assert sorted(label["set_id"] for label in kept) == ["revatio", "viagra-new"]


def test_select_merges_themes_then_filters_and_dedupes():
    januvia = make_label("januvia", ["NDA021995"], brand=("JANUVIA",))
    stub = make_label("stub", ["NDA216778"], brand=("ZITUVIMET XR",), indications=False)
    caduet = make_label("caduet", ["NDA021540"], brand=("Caduet",))
    api = FakeAPI(
        {"Diabetes": [januvia, stub], "Cholesterol": [caduet], "Blood pressure (ARBs)": [caduet]}
    )
    kept, themes, funnel = fetch.select(api)
    assert sorted(label["set_id"] for label in kept) == ["caduet", "januvia"]
    assert themes["caduet"] == {"Cholesterol", "Blood pressure (ARBs)"}
    assert [count for _, count in funnel][-3:] == [3, 2, 2]


def test_manifest_round_trip_sorts_by_theme_then_brand(tmp_path):
    rows = [
        fetch.manifest_row(make_label("b", ["NDA2"], brand=("Zocor",)), {"Cholesterol"}),
        fetch.manifest_row(make_label("a", ["NDA1", "NDA9"], brand=("Januvia",)), {"Diabetes"}),
    ]
    path = tmp_path / "manifest.csv"
    fetch.write_manifest(rows, path)
    back = fetch.read_manifest(path)
    assert [row["brand_name"] for row in back] == ["Januvia", "Zocor"]
    assert back[0]["application_numbers"] == "NDA1; NDA9"
    assert back[0]["theme"] == "Diabetes"


def test_refetch_keeps_pinned_copies_and_warns_on_changes(tmp_path):
    pinned = make_label("a", ["NDA1"], brand=("Alpha",), version="3")
    fetch.save_raw(pinned, tmp_path)
    rows = [
        fetch.manifest_row(pinned, {"Diabetes"}),
        fetch.manifest_row(make_label("b", ["NDA2"], brand=("Beta",), version="7"), {"Diabetes"}),
        fetch.manifest_row(make_label("c", ["NDA3"], brand=("Gamma",), version="1"), {"Diabetes"}),
    ]
    api = FakeAPI(
        by_set_id={
            "a": make_label("a", ["NDA1"], brand=("Alpha",), version="4"),
            "b": make_label("b", ["NDA2"], brand=("Beta",), version="7"),
        }
    )
    downloaded, warnings = fetch.refetch(api, rows, tmp_path)
    assert downloaded == 1
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["version"] == "3"
    assert json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["version"] == "7"
    assert warnings == [
        "Alpha: the manifest pins version 3, openFDA now serves 4; kept your pinned copy",
        "Gamma (c) is no longer on openFDA; no copy to fall back on",
    ]


def test_refetch_takes_the_new_version_when_there_is_no_local_copy(tmp_path):
    rows = [
        fetch.manifest_row(make_label("a", ["NDA1"], brand=("Alpha",), version="3"), {"Diabetes"})
    ]
    api = FakeAPI(by_set_id={"a": make_label("a", ["NDA1"], brand=("Alpha",), version="4")})
    downloaded, warnings = fetch.refetch(api, rows, tmp_path)
    assert downloaded == 1
    assert "downloaded the new version" in warnings[0]


def test_refetch_skips_pinned_local_copies_and_batches_lookups(tmp_path):
    labels = [make_label(f"s{i}", [f"NDA{i}"], version="2") for i in range(60)]
    for label in labels:
        fetch.save_raw(label, tmp_path)
    api = FakeAPI(by_set_id={label["set_id"]: label for label in labels})
    rows = [fetch.manifest_row(label, {"Diabetes"}) for label in labels]
    assert fetch.refetch(api, rows, tmp_path) == (0, [])
    assert len(api.searches) == 3  # 25 + 25 + 10 set_ids


def openfda_with(handler, slept=None):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenFDA(
        http=http, pause=0, sleep=(slept.append if slept is not None else lambda s: None)
    )


def test_openfda_pages_through_results():
    def handler(request):
        skip = int(request.url.params["skip"])
        count = 100 if skip < 200 else 30
        return httpx.Response(
            200, json={"results": [{"set_id": str(skip + i)} for i in range(count)]}
        )

    api = openfda_with(handler)
    assert len(api.search_all("anything")) == 230
    assert api.requests == 3


def test_openfda_treats_404_as_no_matches():
    api = openfda_with(lambda request: httpx.Response(404, json={"error": {"code": "NOT_FOUND"}}))
    assert api.search_all("nothing") == []
    assert api.total("nothing") == 0


def test_openfda_counts_everything_without_a_search():
    def handler(request):
        assert "search" not in request.url.params
        return httpx.Response(200, json={"meta": {"results": {"total": 262909}}, "results": []})

    assert openfda_with(handler).total() == 262909


def test_openfda_retries_rate_limits():
    responses = iter(
        [httpx.Response(429), httpx.Response(200, json={"meta": {"results": {"total": 5}}})]
    )
    slept: list[float] = []
    api = openfda_with(lambda request: next(responses), slept)
    assert api.total("x") == 5
    assert api.requests == 2
    assert 2.0 in slept
