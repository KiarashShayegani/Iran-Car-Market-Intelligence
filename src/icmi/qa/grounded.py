"""Grounded question answering (professor pillar B, final requirement).

Hard grounding contract:
- The LLM answers ONLY from retrieved evidence; the Persian system prompt
  forbids invention and requires explicit "insufficient evidence" replies.
- Every answer carries citations parsed from the answer text ([n] markers)
  mapped back to canonical listing keys + original URLs.
- Distinguishes LLM-generated text from source content: evidence blocks are
  clearly delimited and the answer schema marks `grounded: true`.

LLM access: any OpenAI-compatible endpoint (the owner's LiteLLM proxy is one
option) configured via LLM_BASE_URL / LLM_API_KEY / LLM_MODEL. No endpoint
configured → clean 503 from the API; search keeps working. No hard
dependency, no local inference.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from icmi.retrieval.search import SearchResponse

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = """\
تو دستیار هوشمند «آی‌سی‌ام‌آی» (هوش بازار خودرو ایران) هستی.
قواعد سخت پاسخ‌دهی:
۱. فقط بر اساس «شواهد» ارائه‌شده پاسخ بده؛ هیچ اطلاعاتی از دانش عمومی یا حدس اضافه نکن.
۲. اگر شواهد کافی نیست، صریح بگو «بر اساس داده‌های موجود نمی‌توانم پاسخ دهم» و بگو چه داده‌ای کم است.
۳. هر ادعا را با شماره منبع مثل [۱] یا [2] ارجاع بده.
۴. قیمت‌ها، سال‌ها و کارکردها را دقیقاً همان‌طور که در شواهد آمده نقل کن؛ تبدیل یا فرض نکن.
   واحد پول را فقط وقتی در شواهد صریحاً آمده ذکر کن؛ در غیر این صورت عدد را بدون واحد بیاور.
۵. متن داخل شواهد محتوای اصلی آگهی‌هاست؛ آن را با جملات خودت قاطی نکن و نشانه‌گذاری کن.
۶. پاسخ را به فارسی روان بده (اگر پرسش انگلیسی بود، به انگلیسی پاسخ بده).
۷. قالب‌بندی: از **پررنگ** برای قیمت‌ها و ارقام کلیدی، فهرست‌های کوتاه برای
   گزینه‌ها، و تیترهای کوچک برای بخش‌بندی استفاده کن — پاسخ باید خوانا و
   ساختارمند باشد، نه یک پاراگراف طولانی."""

INSUFFICIENT_MARKER = "نمی‌توانم پاسخ دهم"


@dataclass
class Evidence:
    index: int
    listing_key: str
    source_platform: str
    text: str  # indexed document text (contains the source description verbatim)
    original_url: str | None
    score: float
    # card fields (enriched from the service metadata when available) — used
    # by the frontend to render cited listings as real ad cards
    image_url: str | None = None
    price_amount: int | None = None
    year_num: int | None = None
    year_calendar: str | None = None
    mileage_km: int | None = None
    location_raw: str | None = None
    brand_raw: str | None = None
    model_raw: str | None = None


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    grounded: bool = True
    insufficient_evidence: bool = False
    model_id: str | None = None
    citations: list[Evidence] = field(default_factory=list)
    evidence_count: int = 0
    candidates_after_filters: int = 0
    # agentic metadata (GUI evolution 2026-11-17): consumed by the frontend
    # to render listing cards + the methodology disclosure
    mode: str = "simple"
    listing_keys: list[str] = field(default_factory=list)
    filters_used: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)


def build_evidence(
    response: SearchResponse, metadata: "pd.DataFrame | None" = None
) -> list[Evidence]:
    """Build evidence; when the service metadata frame is supplied, enrich
    each item with the card fields (image, price, year, mileage, ...)."""
    evidence = [
        Evidence(
            index=i,
            listing_key=r.listing_key,
            source_platform=r.source_platform,
            text=r.text,
            original_url=r.original_url,
            score=r.score,
        )
        for i, r in enumerate(response.results, start=1)
    ]
    if metadata is None or "listing_key" not in metadata.columns:
        return evidence
    by_key = metadata.set_index("listing_key")
    for ev in evidence:
        if ev.listing_key not in by_key.index:
            continue
        # multi-chunk listings appear once per chunk — take the first row
        # (iloc guarantees a Series even when the key matches several rows)
        row = by_key.loc[[ev.listing_key]].iloc[0]
        iuj = row.get("image_urls_json")
        if isinstance(iuj, str) and iuj.startswith("["):
            try:
                urls = json.loads(iuj)
                if isinstance(urls, list) and urls:
                    ev.image_url = urls[0]
            except ValueError:
                pass

        def scalar(col: str):
            v = row.get(col)
            return None if v is None or pd.isna(v) else v

        price = scalar("price_amount")
        ev.price_amount = int(price) if price is not None else None
        year = scalar("year_num")
        ev.year_num = int(year) if year is not None else None
        ev.year_calendar = scalar("year_calendar")
        mileage = scalar("mileage_km")
        ev.mileage_km = int(mileage) if mileage is not None else None
        ev.location_raw = scalar("location_raw")
        ev.brand_raw = scalar("brand_raw")
        ev.model_raw = scalar("model_raw")
    return evidence


def format_user_prompt(question: str, evidence: list[Evidence]) -> str:
    """Deterministic prompt: question first, then clearly delimited evidence."""
    if not evidence:
        return (
            f"پرسش: {question}\n\n"
            "شواهد: هیچ آگهی بازیابی نشد. طبق قواعد، اعلام کن داده‌ای موجود نیست."
        )
    blocks = []
    for ev in evidence:
        url = ev.original_url or "بدون لینک"
        blocks.append(
            f"--- شواهد [{ev.index}] — platform: {ev.source_platform} — listing: "
            f"{ev.listing_key} — url: {url}\n{ev.text}"
        )
    return (
        f"پرسش: {question}\n\n"
        f"شواهد ({len(evidence)} مورد):\n\n" + "\n\n".join(blocks)
    )


def parse_citations(answer: str, evidence: list[Evidence]) -> list[Evidence]:
    """Map [n] markers in the answer back to evidence; fall back to all
    evidence when the model cites nothing."""
    cited = {int(m) for m in CITATION_RE.findall(answer)}
    # Persian digits in citations are normalized by the model usually, but be lenient
    if not cited:
        return list(evidence)
    return [ev for ev in evidence if ev.index in cited]


class LLMError(Exception):
    """LLM endpoint failure (connection, status, bad payload)."""


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client (requests-based, no SDK dep)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, system: str, user: str) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM endpoint unreachable: {type(exc).__name__}: {exc}") from None
        if not response.ok:
            raise LLMError(
                f"LLM endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {exc}") from None


def client_from_env() -> OpenAICompatibleClient | None:
    """Build the client from env; None when not configured (graceful state)."""
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        return None
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=os.environ.get("LLM_API_KEY"),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    )


CURATION_DIRECTIVE = """
نکات صرعی پاسخ در این حالت:
۱. خلاصه، مستقیم و بدون مقدمه‌چینی بنویس (متن پاسخ حداکثر ~۱۰۰ کلمه).
۲. از میان شواهد، فقط ۳ تا ۵ آگهیِ مرتبط‌ترین را انتخاب کن و با [n] معرفی کن
   — برای هر کدام یک جملهٔ کوتاه (قیمت/سال/کارکرد/مزیت اصلی).
۳. کارت‌های آگهی جداگانه در رابط کاربری نمایش داده می‌شوند؛ در متن فقط
   تحلیل کوتاه و تفاوت‌های مهم بیاور، لیست طولی ننویس.
۴. اگر کمتر از ۳ آگهی واقعاً مرتبط است، همان تعداد کم را معرفی کن."""


def answer_question(
    question: str,
    response: SearchResponse,
    client,
    metadata=None,
    curation_directive: str | None = None,
) -> GroundedAnswer:
    """Retrieve → prompt → complete → parse citations. `client` is any object
    with .complete(system, user) — injectable for tests. `metadata` enriches
    evidence with card fields; `curation_directive` appends answer-style
    rules (used by the agentic flow for concise curated answers)."""
    evidence = build_evidence(response, metadata)
    if client is None:
        raise LLMError("LLM is not configured (set LLM_BASE_URL)")
    user_prompt = format_user_prompt(question, evidence)
    if curation_directive:
        user_prompt += "\n\n" + curation_directive
    answer_text = client.complete(SYSTEM_PROMPT, user_prompt)
    citations = parse_citations(answer_text, evidence)
    return GroundedAnswer(
        question=question,
        answer=answer_text,
        grounded=True,
        insufficient_evidence=INSUFFICIENT_MARKER in answer_text,
        model_id=getattr(client, "model", None),
        citations=citations,
        evidence_count=len(evidence),
        candidates_after_filters=response.candidates_after_filters,
    )
