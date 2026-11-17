"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import VehicleImage from "@/components/VehicleImage";
import PlatformMark from "@/components/PlatformMark";
import {
  AskApiResponse,
  Citation,
  EMPTY_FILTERS,
  FiltersState,
  askQuestion,
  formatNumber,
  formatToman,
  formatYearText,
  getCompareKeys,
  toggleCompareKey,
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
              <p className="text-sm font-semibold mb-2">
                آگهی‌های پیشنهادی برای پرسش شما:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {answer.citations.map((c) => (
                  <CitationAdCard key={`${c.index}-${c.listing_key}`} citation={c} />
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

function CitationAdCard({ citation }: { citation: Citation }) {
  const [platform, id] = citation.listing_key.split(":");
  const [compareKeys, setCompareKeys] = useState<string[]>(getCompareKeys());
  const inCompare = compareKeys.includes(citation.listing_key);
  const title =
    [citation.brand_raw, citation.model_raw]
      .filter(Boolean)
      .join(" ") || "—";

  const toggleCompare = () => {
    toggleCompareKey(citation.listing_key);
    setCompareKeys(getCompareKeys());
    window.dispatchEvent(new Event("icmi-compare-change"));
  };

  return (
    <div className="card card-hover overflow-hidden relative">
      {/* citation index badge */}
      <span
        className="absolute top-2 right-2 z-10 flex items-center justify-center rounded-lg font-bold num"
        style={{
          width: 30,
          height: 30,
          background: "var(--brand-700, #14514F)",
          color: "#fff",
          fontSize: 13,
        }}
        title="شمارهٔ ارجاع در پاسخ"
      >
        [{citation.index}]
      </span>
      <a href={`/listings/${platform}/${id}`} className="block">
        <VehicleImage
          src={citation.image_url}
          alt={title}
          className="aspect-[4/3] w-full"
        />
      </a>
      <div className="p-3">
        <div className="flex items-center justify-between gap-2">
          <PlatformMark platform={citation.source_platform} />
          {citation.price_amount !== null && citation.price_amount !== undefined && (
            <span className="num text-sm font-bold" style={{ color: "var(--accent)" }}>
              {formatToman(citation.price_amount)}
            </span>
          )}
        </div>
        <p className="mt-2 text-sm font-bold truncate">{title}</p>
        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
          {citation.year_num !== null && citation.year_num !== undefined && (
            <span className="chip btn-secondary num">
              سال {formatYearText(citation.year_num ?? null, citation.year_calendar ?? null, null)}
            </span>
          )}
          {citation.mileage_km !== null && citation.mileage_km !== undefined && (
            <span className="chip btn-secondary num">
              کارکرد {formatNumber(citation.mileage_km)} کیلومتر
            </span>
          )}
          {citation.location_raw && (
            <span className="chip btn-secondary truncate max-w-[9rem]">
              {citation.location_raw}
            </span>
          )}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <a
            href={`/listings/${platform}/${id}`}
            className="text-xs font-semibold"
            style={{ color: "var(--secondary)" }}
          >
            جزئیات در خودروبان →
          </a>
          {citation.original_url && (
            <a
              href={citation.original_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs inline-flex items-center gap-1"
              style={{ color: "var(--accent)" }}
            >
              آگهی اصلی ↗
            </a>
          )}
          <button
            onClick={toggleCompare}
            className={`chip shrink-0 mr-auto ${inCompare ? "score-high" : "btn-secondary"}`}
            title="افزودن/حذف از مقایسه"
          >
            ⚖️
          </button>
        </div>
      </div>
    </div>
  );
}
