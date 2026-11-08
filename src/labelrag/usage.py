"""Show this month's Cohere calls from the call ledger, or record calls made outside it.

    python -m labelrag.usage
    python -m labelrag.usage --add 3 --note "tried something in the Cohere playground"

The ledger is our own count. Cohere's dashboard is the final word on the trial key's usage.
"""

import argparse
import sys

from labelrag import config
from labelrag.client import Store, stamp, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cohere calls recorded in the call ledger.")
    parser.add_argument("--add", type=int, default=0, help="record this many calls made elsewhere")
    parser.add_argument("--note", default="recorded by hand", help="note stored with --add")
    args = parser.parse_args(argv)

    store = Store(config.CACHE_DB)
    now = utc_now()
    for _ in range(args.add):
        store.log_call(stamp(now), "other", "-", 0, "ok", note=args.note)

    used = store.month_calls(now)
    print(f"{now:%B %Y} (UTC): {used:,} of {config.MONTHLY_CALL_BUDGET:,} calls")
    for endpoint, model, outcome, count in store.month_breakdown(now):
        print(f"  {count:>5}  {endpoint:<7} {model:<28} {outcome}")
    embeddings, responses = store.cache_sizes()
    print(f"Cache: {embeddings:,} embedded texts, {responses:,} rerank and chat responses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
