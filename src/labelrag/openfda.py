"""A small client for openFDA's drug label endpoint. No API key is needed at this size:
without one, openFDA allows 240 requests a minute and 1,000 a day."""

import time
from collections.abc import Callable
from typing import Any

import httpx

BASE_URL = "https://api.fda.gov/drug/label.json"
PAGE_SIZE = 100
MAX_ATTEMPTS = 4


class OpenFDA:
    def __init__(
        self,
        http: httpx.Client | None = None,
        pause: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.http = http or httpx.Client(
            timeout=120, headers={"User-Agent": "labelrag (drug label RAG eval)"}
        )
        self.pause, self.sleep = pause, sleep
        self.requests = 0

    def get(self, params: dict[str, Any]) -> dict[str, Any]:
        """One request, retried on 429s, server errors and dropped connections."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.sleep(self.pause)  # stay well under 240 requests a minute
            self.requests += 1
            try:
                resp = self.http.get(BASE_URL, params=params)
            except httpx.TransportError:
                if attempt == MAX_ATTEMPTS:
                    raise
                self.sleep(2.0**attempt)
                continue
            if resp.status_code == 404:  # openFDA's way of saying "no matches"
                return {"meta": {"results": {"total": 0}}, "results": []}
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS:
                self.sleep(2.0**attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise AssertionError("unreachable")

    def total(self, search: str | None = None) -> int:
        """How many labels match `search` (every label when it's None)."""
        params: dict[str, Any] = {"limit": 1}
        if search:
            params["search"] = search
        return self.get(params)["meta"]["results"]["total"]

    def search_all(self, search: str) -> list[dict[str, Any]]:
        """Every label matching `search`, one page of 100 at a time."""
        labels: list[dict[str, Any]] = []
        while True:
            params = {"search": search, "limit": PAGE_SIZE, "skip": len(labels)}
            page = self.get(params).get("results", [])
            labels.extend(page)
            if len(page) < PAGE_SIZE:
                return labels
