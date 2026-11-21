"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import VehicleImage from "@/components/VehicleImage";
import AiMarkdown from "@/components/AiMarkdown";
import { Bot, Loader2, Sparkles } from "lucide-react";
import {
  CALENDAR_LABELS,
  CompareApiResponse,
  CompareVehicle,
  EMPTY_FILTERS,
  askQuestion,
  fetchCompare,
  formatNumber,
  formatToman,
} from "@/lib/api";

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="card p-6 text-sm">در حال بارگذاری…</div>}>
      <CompareContent />
    </Suspense>
  );
}

function CompareContent() {
  const params = useSearchParams();
  const keysParam = params.get("keys") ?? "";
  const keys = keysParam.split(",").filter(Boolean);

  const [data, setData] = useState<CompareApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [aiError, setAiError] = useState<string | null>(null);

  const runAiAnalysis = async () => {
    if (!data || data.vehicles.length < 2) return;
    setAiLoading(true);
    setAiError(null);
    setAiAnalysis(null);
    try {
      const lines = data.vehicles.map((v, i) => {
        const price = v.listing.price_amount !== null ? formatToman(v.listing.price_amount) : "توافقی";
        const pos = pricePosition(v);
        return `${i + 1}. ${v.listing.title ?? v.listing.listing_key} — قیمت: ${price}` +
          `، سال: ${v.listing.year_num ?? "نامشخص"}` +
          `، کارکرد: ${v.listing.mileage_km ?? "نامشخص"} کیلومتر` +
          (pos ? `، وضعیت قیمت: ${pos.label}` : "");
      });
      const question =
        "این آگهی‌های خودرو را از نظر ارزش خرید، تفاوت‌های مهم و وضعیت قیمت نسبت به بازار مقایسه کن و " +
        "پیشنهاد مشخص بده:\n" + lines.join("\n");
      const res = await askQuestion(question, EMPTY_FILTERS, 5, "simple");
      setAiAnalysis(res.answer);
    } catch (e) {
      setAiError(e instanceof Error ? e.message : "خطای ناشناخته");
    } finally {
      setAiLoading(false);
    }
  };

  useEffect(() => {
    if (keys.length < 2) {
      setError("برای مقایسه، حداقل دو خودرو انتخاب کنید — از نتایج جستجو با دکمهٔ مقایسه اضافه کنید.");
      setData(null);
      return;
    }
    fetchCompare(keys)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
  }, [keysParam]);

  if (error)
    return (
      <div className="card p-8 text-center">
        <p className="text-sm font-bold mb-2">مقایسه خالی است</p>
        <p className="text-sm mb-4" style={{ color: "var(--muted)" }}>
          {error} — آگهی‌ها را از کارت‌های نتایج جستجو با دکمهٔ «مقایسه» انتخاب کنید.
        </p>
        <a href="/search" className="btn !px-5 !py-2 !text-sm inline-flex items-center gap-1.5">
          🔍 رفتن به جستجوی خودروها
        </a>
      </div>
    );
  if (!data) return <div className="card p-6 text-sm">در حال دریافت…</div>;
  const vehicles = data.vehicles;

  // price-position intelligence: advertised price vs the vehicle's own
  // comparable-market band (honest, sample-size aware)
  function pricePosition(v: CompareVehicle): {
    label: string;
    color: string;
    detail: string;
  } | null {
    const price = v.listing.price_amount;
    const { median, p25, p75, count } = v.market_range;
    if (price === null || median === null || count < 3) {
      return count > 0
        ? { label: "نمونهٔ کم", color: "var(--warning-text)", detail: `${count} آگهی مشابه — قضاوت معتبر نیست` }
        : null;
    }
    const diffPct = Math.round(((price - median) / median) * 100);
    const below = diffPct < -8;
    const above = diffPct > 8;
    const inBand = p25 !== null && p75 !== null && price >= p25 && price <= p75;
    if (inBand)
      return {
        label: "داخل بازهٔ بازار",
        color: "var(--success)",
        detail: `${diffPct >= 0 ? "+" : ""}${diffPct}٪ نسبت به میانهٔ ${count} آگهی مشابه`,
      };
    if (below)
      return {
        label: "زیر قیمت بازار",
        color: "var(--success)",
        detail: `${Math.abs(diffPct)}٪ ارزان‌تر از میانهٔ ${count} آگهی مشابه — بررسی وضعیت آگهی توصیه می‌شود`,
      };
    if (above)
      return {
        label: "بالاتر از قیمت بازار",
        color: "var(--danger)",
        detail: `${diffPct}٪ گران‌تر از میانهٔ ${count} آگهی مشابه`,
      };
    return {
      label: "نزدیک میانهٔ بازار",
      color: "var(--info-text)",
      detail: `${diffPct >= 0 ? "+" : ""}${diffPct}٪ نسبت به ${count} آگهی مشابه`,
    };
  }

  const specRows: { label: string; pick: (v: CompareVehicle) => string | null }[] = [
    { label: "برند", pick: (v) => v.listing.brand_raw },
    { label: "مدل", pick: (v) => v.listing.model_raw },
    { label: "تیپ", pick: (v) => v.listing.trim_raw },
    { label: "سال (خام)", pick: (v) => formatNumber(v.listing.year_raw) },
    { label: "تقویم", pick: (v) => (v.listing.year_calendar ? CALENDAR_LABELS[v.listing.year_calendar] : null) },
    { label: "قیمت آگهی", pick: (v) => formatToman(v.listing.price_amount) },
    { label: "کارکرد", pick: (v) => (v.listing.mileage_raw ? formatNumber(v.listing.mileage_raw) : null) },
    { label: "سوخت", pick: (v) => v.listing.fuel_raw },
    { label: "گیربکس", pick: (v) => v.listing.transmission_raw },
    { label: "نوع بدنه", pick: (v) => v.listing.body_type_raw },
    { label: "وضعیت بدنه", pick: (v) => v.listing.body_condition_raw },
    { label: "رنگ", pick: (v) => v.listing.color_raw },
    { label: "موقعیت", pick: (v) => v.listing.location_raw },
    {
      label: "میانه بازار مشابه‌ها",
      pick: (v) => formatToman(v.market_range.median),
    },
    {
      label: "بازهٔ بازار (P25–P75)",
      pick: (v) =>
        v.market_range.p25 !== null
          ? `${formatToman(v.market_range.p25)} تا ${formatToman(v.market_range.p75)}`
          : null,
    },
    {
      label: "نمونه‌های مشابه",
      pick: (v) => formatNumber(v.market_range.count),
    },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-bold">مقایسهٔ خودروها</h1>
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: `repeat(${vehicles.length}, minmax(0, 1fr))` }}
      >
        {vehicles.map((v) => {
          const [platform, id] = v.listing.listing_key.split(":");
          return (
            <div key={v.listing.listing_key} className="card overflow-hidden">
              <VehicleImage
                src={v.image_url}
                alt={v.listing.title ?? v.listing.listing_key}
                className="aspect-[4/3] w-full"
              />
              <div className="p-3">
                <p className="text-sm font-bold">{v.listing.title ?? "—"}</p>
                <a
                  href={v.listing.original_url ?? undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-flex items-center gap-1 text-xs underline"
                  style={{ color: "var(--accent)" }}
                >
                  مشاهدهٔ آگهی اصلی ↗
                </a>
                {(() => {
                  const pos = pricePosition(v);
                  if (!pos) return null;
                  return (
                    <div
                      className="mt-2 rounded-lg p-2 text-xs"
                      style={{ background: "var(--bg-muted)" }}
                    >
                      <p className="font-bold" style={{ color: pos.color }}>
                        {pos.label}
                      </p>
                      <p style={{ color: "var(--muted)" }}>{pos.detail}</p>
                    </div>
                  );
                })()}
              </div>
            </div>
          );
        })}
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <tbody>
            {specRows.map(({ label, pick }) => {
              const values = vehicles.map((v) => pick(v));
              const allSame = values.every((v) => v === values[0]);
              return (
                <tr key={label} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="p-2.5 text-xs font-semibold" style={{ color: "var(--muted)" }}>
                    {label}
                  </td>
                  {values.map((value, i) => (
                    <td key={i} className={`p-2.5 num ${allSame ? "" : "font-medium"}`}>
                      {value ?? (
                        <span style={{ color: "var(--muted)" }}>— نامشخص</span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs" style={{ color: "var(--muted)" }}>
        بازهٔ بازار هر خودرو از آگهی‌های مشابهِ همان برند/مدل/سال (تقویم‌آگاه)
        محاسبه می‌شود و کاملاً قابل‌بررسی است — این یک برآورد آماری است، نه قیمت
        قطعی. مقادیر نامشخص صادقانه «—» نمایش داده می‌شوند.
      </p>

      {/* AI-assisted analysis */}
      <div className="card p-4">
        <div className="flex items-center justify-between gap-2 mb-3">
          <p className="text-sm font-bold flex items-center gap-1.5">
            <Sparkles size={15} style={{ color: "var(--accent)" }} />
            تحلیل هوشمند مقایسه
          </p>
          <button
            className="btn !px-4 !py-2 !text-sm flex items-center gap-1.5"
            onClick={runAiAnalysis}
            disabled={aiLoading || data.vehicles.length < 2}
          >
            {aiLoading ? <Loader2 size={14} className="animate-spin" /> : <Bot size={14} />}
            {aiLoading ? "در حال تحلیل…" : "تحلیل با هوش مصنوعی"}
          </button>
        </div>
        {aiError && (
          <div className="p-3 rounded-lg text-xs" style={{ background: "#fef9c3", color: "#713f12" }}>
            ⚠️ {aiError}
          </div>
        )}
        {aiLoading && (
          <div className="flex items-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
            <Loader2 size={15} className="animate-spin" />
            در حال بررسی آگهی‌های انتخابی و نگارش تحلیل…
          </div>
        )}
        {aiAnalysis && (
          <div className="text-sm">
            <AiMarkdown content={aiAnalysis} />
          </div>
        )}
        {!aiAnalysis && !aiLoading && !aiError && (
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            دستیار هوشمند، آگهی‌های انتخابی را از نظر ارزش خرید، تفاوت‌ها و وضعیت
            قیمت تحلیل می‌کند — پاسخ فقط بر اساس همین آگهی‌ها ساخته می‌شود.
          </p>
        )}
      </div>
    </div>
  );
}
