"use client";

import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2, ExternalLink, Loader2, MessageSquareText, PenLine,
  Search as SearchIcon, Sparkles, User,
} from "lucide-react";
import AiMarkdown from "@/components/AiMarkdown";
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

/** The real agentic pipeline order — mirrored live while loading, then
    replaced by the actual steps trace from the response. */
const STEP_META = [
  { icon: Sparkles, label: "تحلیل پرسش" },
  { icon: SearchIcon, label: "جستجو در آگهی‌ها" },
  { icon: PenLine, label: "نگارش پاسخ مبتنی بر شواهد" },
];

export default function AskPanel({ filters = EMPTY_FILTERS }: { filters?: FiltersState }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState(0); // animated step progression
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AskApiResponse | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const submit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setPhase(0);
    setError(null);
    setAnswer(null);
    // live step progression while the backend runs intent → retrieval → answer
    timerRef.current = setInterval(() => {
      setPhase((p) => Math.min(p + 1, STEP_META.length - 1));
    }, 2200);
    try {
      setAnswer(await askQuestion(question.trim(), filters));
    } catch (e) {
      setError(e instanceof Error ? e.message : "خطای ناشناخته");
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setLoading(false);
    }
  };

  const hasActiveFilters =
    filters.platforms.length > 0 ||
    Boolean(filters.price_max) ||
    Boolean(filters.price_min) ||
    Boolean(filters.mileage_max_km);

  const showConversation = loading || Boolean(answer);

  return (
    <div className="space-y-4">
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
              if (!loading) submit();
            }
          }}
          placeholder="مثلاً: بهترین پژو ۲۰۶ با قیمت زیر یک میلیارد تومان کدام است؟"
          disabled={loading}
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {hasActiveFilters &&
              "پرسش با فیلترهای فعال جستجو محدود می‌شود (فیلترها را در تب جستجو تنظیم کنید)"}
          </span>
          <button className="btn" onClick={submit} disabled={loading || !question.trim()}>
            {loading ? "…" : "بپرس"}
          </button>
        </div>

        {!question && !answer && !error && !loading && (
          <div className="mt-3">
            <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>نمونه‌پرسش‌ها:</p>
            <div className="flex flex-wrap gap-2">
              {[
                "بهترین پژو ۲۰۶ با قیمت زیر یک میلیارد تومان کدام است؟",
                "ارزان‌ترین ماشین اتوماتیک چیه؟",
                "ماشین‌های کم‌کارکرد زیر ده میلیارد چی داریم؟",
                "بین پژوها کدوم مدل گرون‌تره؟",
              ].map((q) => (
                <button key={q} className="chip btn-secondary" onClick={() => setQuestion(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {error && (
        <div
          className="p-3 rounded-lg text-sm"
          style={{ background: "#fef9c3", color: "#713f12" }}
        >
          ⚠️ {error}
        </div>
      )}

      {showConversation && (
        <div className="space-y-3">
          {/* user message */}
          <div className="flex justify-start">
            <div
              className="max-w-[85%] rounded-2xl rounded-tl-sm px-4 py-2.5"
              style={{ background: "var(--teal-soft)" }}
            >
              <p
                className="text-[11px] font-bold mb-1 flex items-center gap-1"
                style={{ color: "var(--secondary)" }}
              >
                <User size={12} /> شما
              </p>
              <p className="text-sm">
                {answer?.question ?? question}
              </p>
            </div>
          </div>

          {/* AI answer */}
          <div className="flex gap-3">
            <div
              className="shrink-0 flex items-center justify-center rounded-full"
              style={{
                width: 38,
                height: 38,
                background: "var(--gradient-brand)",
                color: "#fff",
              }}
              title="دستیار هوشمند خودروبان"
            >
              <Sparkles size={18} />
            </div>
            <div className="flex-1 min-w-0 space-y-3">
              <div className="flex items-center gap-2">
                <p className="text-sm font-bold">دستیار هوشمند خودروبان</p>
                {answer && !answer.insufficient_evidence && (
                  <span className="chip score-high !text-[10px]">پاسخ مبتنی بر شواهد</span>
                )}
              </div>

              {loading ? (
                /* live agent steps */
                <div className="card p-4 space-y-2.5">
                  {STEP_META.map((s, i) => {
                    const Icon = s.icon;
                    const done = i < phase;
                    const active = i === phase;
                    return (
                      <div key={s.label} className="flex items-center gap-2.5 text-sm">
                        {done ? (
                          <CheckCircle2 size={17} style={{ color: "var(--success)" }} />
                        ) : active ? (
                          <Loader2 size={17} className="animate-spin" style={{ color: "var(--secondary)" }} />
                        ) : (
                          <Icon size={17} style={{ color: "var(--muted)" }} />
                        )}
                        <span style={{ color: active ? "var(--foreground)" : "var(--muted)" }}>
                          {s.label}
                          {active && "…"}
                        </span>
                      </div>
                    );
                  })}
                  <div className="space-y-1.5 pt-1">
                    <div className="skeleton h-3 w-full" />
                    <div className="skeleton h-3 w-5/6" />
                    <div className="skeleton h-3 w-4/6" />
                  </div>
                </div>
              ) : (
                answer && (
                  <>
                    {answer.insufficient_evidence && (
                      <div
                        className="p-3 rounded-lg text-xs"
                        style={{ background: "#fee2e2", color: "#991b1b" }}
                      >
                        مدل اعلام کرده که شواهد کافی برای پاسخ نیست — این یک پاسخ
                        صادقانه است، نه حدس. پرسش را دقیق‌تر یا با فیلترهای متفاوت
                        امتحان کنید.
                      </div>
                    )}

                    {/* step trace: compact transparency chips */}
                    {answer.steps.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
                        {answer.steps.map((s, i) => (
                          <span
                            key={i}
                            className="chip btn-secondary inline-flex items-center gap-1"
                            title={`مرحلهٔ ${i + 1} از پردازش`}
                          >
                            <CheckCircle2 size={11} style={{ color: "var(--success)" }} />
                            {s.step === "intent"
                              ? "تحلیل پرسش"
                              : s.step === "search"
                                ? `جستجو (${s.results ?? "—"} نتیجه)`
                                : s.step === "search_refine"
                                  ? `جستجوی گسترده‌تر (${s.results ?? "—"} نتیجه)`
                                  : s.step === "answer"
                                    ? "نگارش پاسخ"
                                    : s.step}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="card p-4 answer-text text-sm">
                      <AiMarkdown content={answer.answer} />
                    </div>

                    {answer.citations.length > 0 && (
                      <div>
                        <p
                          className="text-xs font-semibold mb-2 flex items-center gap-1.5"
                          style={{ color: "var(--muted)" }}
                        >
                          <MessageSquareText size={13} />
                          آگهی‌های پیشنهادی برای پرسش شما:
                        </p>
                        <div className="space-y-2">
                          {answer.citations.map((c) => (
                            <MiniResultRow key={`${c.index}-${c.listing_key}`} citation={c} />
                          ))}
                        </div>
                      </div>
                    )}

                    <details className="card p-3 text-xs" style={{ color: "var(--muted)" }}>
                      <summary
                        className="cursor-pointer font-semibold"
                        style={{ color: "var(--secondary)" }}
                      >
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
                        پاسخ فقط از آگهی‌های بازیابی‌شده ساخته می‌شود؛ مدل به داده‌های
                        دیگر دسترسی ندارد.
                      </p>
                    </details>
                  </>
                )
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Compact in-chat result row — minimal, integrated, not a full ad card. */
function MiniResultRow({ citation }: { citation: Citation }) {
  const [platform, id] = citation.listing_key.split(":");
  const [compareKeys, setCompareKeys] = useState<string[]>(getCompareKeys());
  const inCompare = compareKeys.includes(citation.listing_key);
  const title = [citation.brand_raw, citation.model_raw].filter(Boolean).join(" ") || "—";

  const toggleCompare = () => {
    toggleCompareKey(citation.listing_key);
    setCompareKeys(getCompareKeys());
    window.dispatchEvent(new Event("icmi-compare-change"));
  };

  return (
    <div className="card p-2.5 flex items-center gap-3 hover:shadow-md transition-shadow">
      <span
        className="shrink-0 flex items-center justify-center rounded-lg font-bold num"
        style={{
          width: 26,
          height: 26,
          fontSize: 11,
          background: "var(--brand-700, #14514F)",
          color: "#fff",
        }}
        title="شمارهٔ ارجاع در پاسخ"
      >
        {citation.index}
      </span>
      <VehicleImage
        src={citation.image_url}
        alt={title}
        className="w-16 h-12 rounded-lg object-cover shrink-0"
      />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-bold truncate">{title}</p>
        <p className="text-[11px] mt-0.5" style={{ color: "var(--muted)" }}>
          {[
            citation.year_num !== null && citation.year_num !== undefined
              ? `سال ${formatYearText(citation.year_num ?? null, citation.year_calendar ?? null, null)}`
              : null,
            citation.mileage_km !== null && citation.mileage_km !== undefined
              ? `کارکرد ${formatNumber(citation.mileage_km)}`
              : null,
            citation.location_raw,
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      <div className="shrink-0 text-left">
        <p className="num text-sm font-bold" style={{ color: "var(--accent)" }}>
          {citation.price_amount !== null && citation.price_amount !== undefined
            ? formatToman(citation.price_amount)
            : "توافقی"}
        </p>
        <div className="flex items-center gap-1.5 mt-0.5 justify-end">
          <a
            href={`/listings/${platform}/${id}`}
            className="text-[11px] font-semibold"
            style={{ color: "var(--secondary)" }}
            title="مشاهدهٔ آگهی در خودروبان"
          >
            جزئیات
          </a>
          {citation.original_url && (
            <a
              href={citation.original_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] inline-flex items-center gap-0.5"
              style={{ color: "var(--accent)" }}
              title="مشاهده در منبع اصلی"
            >
              منبع <ExternalLink size={10} />
            </a>
          )}
          <button
            onClick={toggleCompare}
            className={`chip !px-1.5 !py-0.5 ${inCompare ? "score-high" : "btn-secondary"}`}
            title="افزودن/حذف از مقایسه"
          >
            ⚖️
          </button>
        </div>
      </div>
    </div>
  );
}
