"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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

      {!question && !answer && !error && (
        <div className="mt-3">
          <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>نمونه‌پرسش‌ها:</p>
          <div className="flex flex-wrap gap-2">
            {[
              "ارزان‌ترین ماشین اتوماتیک چیه؟",
              "کدوم ماشین‌ها گارانتی فعال دارن؟",
              "بین پژوها کدوم مدل گرون‌تره؟",
              "ماشین‌های کم‌کارکرد زیر ده میلیارد چی داریم؟",
            ].map((q) => (
              <button key={q} className="chip btn-secondary" onClick={() => setQuestion(q)}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div
          className="mt-4 p-3 rounded-lg text-sm"
          style={{ background: "#fef9c3", color: "#713f12" }}
        >
          ⚠️ {error}
        </div>
      )}

      {loading && (
        <div className="mt-4 space-y-3">
          <div className="card p-4 space-y-2">
            <div className="skeleton h-4 w-1/3" />
            <div className="skeleton h-3 w-full" />
            <div className="skeleton h-3 w-5/6" />
            <div className="skeleton h-3 w-4/6" />
          </div>
          <p className="text-xs text-center" style={{ color: "var(--muted)" }}>
            در حال تحلیل پرسش، جستجوی معنایی و ساخت پاسخ مبتنی بر شواهد…
          </p>
        </div>
      )}

      {answer && (
        <div className="mt-4 space-y-3">
          {answer.insufficient_evidence && (
            <div
              className="p-3 rounded-lg text-xs"
              style={{ background: "#fee2e2", color: "#991b1b" }}
            >
              مدل اعلام کرده که شواهد کافی برای پاسخ نیست — این یک پاسخ صادقانه
              است، نه حدس. پرسش را دقیق‌تر یا با فیلترهای متفاوت امتحان کنید.
            </div>
          )}

          <div className="card p-4 answer-text text-sm prose-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.answer}</ReactMarkdown>
          </div>

          {answer.citations.length > 0 && (
            <div>
              <p className="text-xs font-semibold mb-2">
                آگهی‌های مرجع پاسخ ({answer.citations.length} از {answer.evidence_count} شاهد):
              </p>
              <div className="space-y-2">
                {answer.citations.map((c) => (
                  <CitationCard key={`${c.index}-${c.listing_key}`} citation={c} />
                ))}
              </div>
            </div>
          )}

          <details className="card p-3 text-xs" style={{ color: "var(--muted)" }}>
            <summary className="cursor-pointer font-semibold" style={{ color: "var(--secondary)" }}>
              روش پاسخ و شفافیت
            </summary>
            <p className="mt-2">
              حالت: {answer.mode === "agentic" ? "دو مرحله‌ای (تحلیل پرسش → جستجو → پاسخ)" : "ساده"} ·
              مدل: {answer.model_id ?? "—"} · شواهد بازیابی‌شده: {answer.evidence_count}
            </p>
            {answer.mode === "agentic" && answer.steps.length > 0 && (
              <ol className="mt-1 space-y-0.5 list-decimal list-inside">
                {answer.steps.map((s, i) => (
                  <li key={i}>
                    {s.step}
                    {typeof s.results === "number" ? ` — ${s.results} نتیجه` : ""}
                    {s.note ? ` — ${s.note}` : ""}
                  </li>
                ))}
              </ol>
            )}
            <p className="mt-1">
              پاسخ فقط از آگهی‌های بازیابی‌شده ساخته می‌شود؛ مدل به داده‌های دیگر
              دسترسی ندارد.
            </p>
          </details>
        </div>
      )}
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  const [platform, id] = citation.listing_key.split(":");
  return (
    <div className="card p-3 flex items-center gap-3">
      <span
        className="flex items-center justify-center rounded-lg font-bold shrink-0"
        style={{
          width: 34,
          height: 34,
          background: "var(--teal-soft)",
          color: "var(--secondary)",
        }}
      >
        [{citation.index}]
      </span>
      <div className="min-w-0 flex-1">
        {citation.excerpt && (
          <p className="text-xs truncate mb-0.5" style={{ color: "var(--muted)" }}>
            {citation.excerpt.slice(0, 110)}
          </p>
        )}
        <a
          href={`/listings/${platform}/${id}`}
          className="text-xs font-semibold"
          style={{ color: "var(--secondary)" }}
        >
          مشاهدهٔ آگهی در خودروبان →
        </a>
      </div>
      {citation.original_url && (
        <a
          href={citation.original_url}
          target="_blank"
          rel="noopener noreferrer"
          className="chip btn-secondary shrink-0"
          title="مشاهده در منبع اصلی"
        >
          منبع اصلی ↗
        </a>
      )}
    </div>
  );
}
