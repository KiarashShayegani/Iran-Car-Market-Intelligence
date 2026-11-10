"use client";

import { useState } from "react";
import {
  AskApiResponse,
  Citation,
  EMPTY_FILTERS,
  FiltersState,
  askQuestion,
} from "@/lib/api";

export default function AskPanel({ filters = EMPTY_FILTERS }: { filters?: FiltersState }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AskApiResponse | null>(null);

  const submit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await askQuestion(question.trim(), filters));
    } catch (e) {
      setError(e instanceof Error ? e.message : "خطای ناشناخته");
    } finally {
      setLoading(false);
    }
  };

  const hasActiveFilters =
    filters.platforms.length > 0 ||
    Boolean(filters.price_max) ||
    Boolean(filters.price_min) ||
    Boolean(filters.mileage_max_km);

  return (
    <div className="card p-4">
      <label className="block text-sm font-semibold mb-2">
        پرسش خود را به فارسی (یا انگلیسی) بپرسید — پاسخ فقط بر اساس آگهی‌های
        بازیابی‌شده ساخته می‌شود:
      </label>
      <textarea
        rows={2}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="مثلاً: ارزان‌ترین ماشین اتوماتیک تو این داده‌ها چی هست و چقدره؟"
      />
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {hasActiveFilters &&
            "پرسش با فیلترهای فعال جستجو محدود می‌شود (فیلترها را در تب جستجو تنظیم کنید)"}
        </span>
        <button className="btn" onClick={submit} disabled={loading || !question.trim()}>
          {loading ? "در حال پردازش…" : "بپرس"}
        </button>
      </div>

      {error && (
        <div
          className="mt-4 p-3 rounded-lg text-sm"
          style={{ background: "#fef9c3", color: "#713f12" }}
        >
          ⚠️ {error}
        </div>
      )}

      {answer && (
        <div className="mt-4">
          {answer.insufficient_evidence && (
            <div
              className="mb-3 p-2 rounded-lg text-xs"
              style={{ background: "#fee2e2", color: "#991b1b" }}
            >
              مدل اعلام کرده که شواهد کافی برای پاسخ نیست — این یک پاسخ صادقانه
              است، نه حدس.
            </div>
          )}
          <div className="card p-4 answer-text text-sm">{answer.answer}</div>
          <div className="mt-3">
            <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>
              منابع استنادشده ({answer.citations.length} از {answer.evidence_count} شاهد):
            </p>
            <div className="flex flex-wrap gap-2">
              {answer.citations.map((c) => (
                <CitationChip key={`${c.index}-${c.listing_key}`} citation={c} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CitationChip({ citation }: { citation: Citation }) {
  const label = citation.original_url ? "↗" : "";
  return (
    <a
      href={citation.original_url ?? undefined}
      target="_blank"
      rel="noopener noreferrer"
      className={`chip platform-${citation.source_platform}`}
      style={!citation.original_url ? { opacity: 0.6, pointerEvents: "none" } : {}}
      title={`${citation.listing_key} — ${citation.original_url ?? "بدون لینک"}`}
    >
      [{citation.index}] {citation.listing_key} {label}
    </a>
  );
}
