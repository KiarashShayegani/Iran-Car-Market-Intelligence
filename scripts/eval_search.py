#!/usr/bin/env python3
"""Run the evaluation set: semantic-only vs hybrid side by side.

Mechanical checks (asserted): hybrid results satisfy 100% of their filters.
Reported per query: candidate counts, term-hit ratio, top results for eyeball
review. Output saved to docs/internal/benchmark_runs/search_eval_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from icmi.core.paths import repo_root
from icmi.retrieval.encoder import DEFAULT_MODEL_ID, get_encoder
from icmi.retrieval.index import load_index
from icmi.retrieval.search import SearchFilters, SearchService


def _term_hit(results, terms) -> float:
    if not terms:
        return -1.0  # no term check
    if not results:
        return 0.0
    hits = 0
    for result in results[:5]:
        text = result.text
        if any(term in text for term in terms):
            hits += 1
    return hits / min(len(results), 5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", default=None, help="alternative queries JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)

    eval_path = (
        Path(args.eval_file)
        if args.eval_file
        else repo_root() / "tests" / "data" / "eval_queries.json"
    )
    eval_set = json.loads(eval_path.read_text(encoding="utf-8"))

    vectors, metadata, info = load_index()
    service = SearchService(vectors, metadata, get_encoder(args.model))

    rows = []
    filter_violations = 0
    for item in eval_set["queries"]:
        filters = SearchFilters(**item.get("filters", {}))
        hybrid = service.search(item["query"], filters, top_k=args.top_k)
        semantic = service.search(item["query"], SearchFilters(), top_k=args.top_k)

        # mechanical requirement: hybrid results satisfy filters 100%
        for result in hybrid.results:
            if not _result_satisfies(result, filters, metadata):
                filter_violations += 1

        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "intent": item.get("intent"),
                "hybrid_candidates": hybrid.candidates_after_filters,
                "hybrid_results": len(hybrid.results),
                "semantic_results": len(semantic.results),
                "term_hit_ratio_hybrid": _term_hit(
                    hybrid.results, item.get("require_terms", [])
                ),
                "term_hit_ratio_semantic": _term_hit(
                    semantic.results, item.get("require_terms", [])
                ),
                "hybrid_top": [r.summary_line() for r in hybrid.results[:3]],
                "semantic_top": [r.summary_line() for r in semantic.results[:3]],
            }
        )

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "model_id": args.model,
        "index_info": info,
        "filter_violations": filter_violations,
        "queries": rows,
    }
    out_dir = repo_root() / "docs" / "internal" / "benchmark_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"search_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"queries: {len(rows)} | filter violations (MUST be 0): {filter_violations}")
    for row in rows:
        th = row["term_hit_ratio_hybrid"]
        term_str = f"{th:.2f}" if th >= 0 else "n/a"
        print(
            f"  {row['id']}: candidates={row['hybrid_candidates']:>3} "
            f"results={row['hybrid_results']}/{row['semantic_results']} "
            f"term-hit(h/s)={term_str}/"
            f"{row['term_hit_ratio_semantic'] if row['term_hit_ratio_semantic'] >= 0 else 'n/a'}"
            f"  {row['query'][:50]}"
        )
    print(f"report: {out_path}")
    return 0 if filter_violations == 0 else 1


def _result_satisfies(result, filters: SearchFilters, metadata) -> bool:
    """Verify one result against the filters via its listing metadata."""
    row = metadata[metadata["listing_key"] == result.listing_key]
    if row.empty:
        return False
    row = row.iloc[0]
    if filters.platforms and row["source_platform"] not in filters.platforms:
        return False
    def ok(field, min_v=None, max_v=None):
        if pd_isna(row[field]):
            return False
        if min_v is not None and row[field] < min_v:
            return False
        if max_v is not None and row[field] > max_v:
            return False
        return True

    if filters.price_min is not None and not ok("price_amount", min_v=filters.price_min):
        return False
    if filters.price_max is not None and not ok("price_amount", max_v=filters.price_max):
        return False
    if filters.year_min is not None and not ok("year_num", min_v=filters.year_min):
        return False
    if filters.year_max is not None and not ok("year_num", max_v=filters.year_max):
        return False
    if filters.mileage_max_km is not None and not ok("mileage_km", max_v=filters.mileage_max_km):
        return False
    return True


def pd_isna(value) -> bool:
    import math

    return value is None or (isinstance(value, float) and math.isnan(value))


if __name__ == "__main__":
    sys.exit(main())
