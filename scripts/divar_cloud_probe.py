#!/usr/bin/env python3
"""Divar cloud-accessibility probe (bounded, diagnostics-only).

Runs the exact POST pattern verified locally against the Divar API FROM the
GitHub runner environment, and reports:

1. cold-start POST (category light + sort_date, Tehran) — status, bytes,
   post count, first-post title.
2. one cursor-continuation POST (pagination.data echo).
3. public SSR listing page GET (divar.ir/v/<token>) — status + bytes.

Writes data/divar-probe/diagnostics.json (uploaded as an artifact; NOT
committed). No corpus collection — this is an accessibility experiment only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://api.divar.ir/v8/postlist/w/search"
SSR = "https://divar.ir/v/{token}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://divar.ir",
    "Referer": "https://divar.ir/s/tehran/car",
}


def cold_start() -> dict:
    body = {
        "city_ids": ["1"],
        "search_data": {"form_data": {"data": {
            "category": {"str": {"value": "light"}},
            "sort": {"str": {"value": "sort_date"}},
        }}},
    }
    r = requests.post(BASE, json=body, timeout=30, headers=HEADERS)
    out = {"step": "cold_start", "http": r.status_code, "bytes": len(r.content)}
    if r.status_code == 200:
        d = r.json()
        rows = [w.get("data", {}) for w in d.get("list_widgets", [])
                if w.get("widget_type") == "POST_ROW"]
        out.update({
            "post_rows": len(rows),
            "first_title": (rows[0] or {}).get("title", "")[:50] if rows else None,
            "tokens": [rw.get("token") for rw in rows if rw.get("token")][:5],
        })
        out["_cursor"] = (d.get("pagination") or {}).get("data")
        out["_first_token"] = (rows[0] or {}).get("token") if rows else None
    else:
        out["body_head"] = r.text[:200]
    return out


def continuation(cursor: dict) -> dict:
    body = {"request_trigger": "REQUEST_TRIGGER_NORMAL", "pagination_data": cursor}
    r = requests.post(BASE, json=body, timeout=30, headers=HEADERS)
    out = {"step": "cursor_continuation", "http": r.status_code, "bytes": len(r.content)}
    if r.status_code == 200:
        d = r.json()
        rows = [w.get("data", {}) for w in d.get("list_widgets", [])
                if w.get("widget_type") == "POST_ROW"]
        tokens = [rw.get("token") for rw in rows if rw.get("token")]
        out.update({
            "post_rows": len(rows),
            "tokens": tokens[:5],
            "has_next": (d.get("pagination") or {}).get("has_next_page"),
        })
    else:
        out["body_head"] = r.text[:200]
    return out


def ssr_page(token: str) -> dict:
    r = requests.get(f"https://divar.ir/v/{token}", timeout=30,
                     headers={"User-Agent": HEADERS["User-Agent"]})
    return {"step": "ssr_page", "http": r.status_code, "bytes": len(r.content)}


def main() -> int:
    from icmi.ops.journal import journal_event

    out = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "environment": "github-actions-runner"}
    print("== divar cloud probe ==", flush=True)

    cold = cold_start()
    out["cold_start"] = {k: v for k, v in cold.items() if not k.startswith("_")}
    print(" cold_start:", json.dumps(out["cold_start"], ensure_ascii=False)[:200], flush=True)

    time.sleep(2.5)
    cont = continuation(cold["_cursor"]) if "_cursor" in cold else {
        "step": "cursor_continuation", "skipped": "no cursor from cold start"}
    out["continuation"] = {k: v for k, v in cont.items() if not k.startswith("_")}
    print(" continuation:", json.dumps(out["continuation"], ensure_ascii=False)[:200], flush=True)

    token = cold.get("_first_token")
    time.sleep(2.5)
    if token:
        ssr = ssr_page(token)
        out["ssr_page"] = ssr
        print(" ssr_page:", json.dumps(ssr), flush=True)

    out["verdict"] = (
        "cloud-viable"
        if cold.get("http") == 200 and cont.get("http") == 200
        else "blocked"
    )
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    dest = Path("data/divar-probe")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("diagnostics written:", dest / "diagnostics.json", flush=True)
    journal_event("divar_probe", "completed", verdict=out["verdict"])
    return 0 if out["verdict"] == "cloud-viable" else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
