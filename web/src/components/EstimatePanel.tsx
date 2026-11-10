"use client";

import { useState } from "react";
import {
  API_BASE,
  CALENDAR_LABELS,
  ListingDetail,
  PLATFORM_LABELS,
  formatNumber,
  formatToman,
} from "@/lib/api";

interface EstimateResponse {
  brand_key: string;
  model_key: string | null;
  year_num: number | null;
  year_calendar: string | null;
  mileage_km: number | null;
  count: number;
  median: number | null;
  p25: number | null;
  p75: number | null;
  min: number | null;
  max: number | null;
  quality: "ok" | "low_sample" | "no_data";
  quality_note: string;
  comparables: {
    listing_key: string;
    source_platform: string;
    title: string | null;
    model_raw: string | null;
    year_num: number | null;
    year_calendar: string | null;
    mileage_km: number | null;
    price_amount: number;
    original_url: string | null;
  }[];
}

export default function EstimatePanel({ onOpenDetail }: { onOpenDetail?: (k: string) => void }) {
  const [brand, setBrand] = useState("");
  const [model, setModel] = useState("");
  const [year, setYear] = useState("");
  const [mileage, setMileage] = useState("");
  const [trimOutliers, setTrimOutliers] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EstimateResponse | null>(null);
  const [detail, setDetail] = useState<ListingDetail | null>(null);

  const submit = async () => {
    if (!brand.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setDetail(null);
    try {
      const p = new URLSearchParams({ brand: brand.trim() });
      if (model.trim()) p.set("model", model.trim());
      if (year.trim()) p.set("year", year.trim());
      if (mileage.trim()) p.set("mileage_km", mileage.trim());
      if (trimOutliers) p.set("trim_outliers", "true");
      const res = await fetch(`${API_BASE}/estimate?${p.toString()}`);
      if (!res.ok) throw new Error(`خطای تخمین (${res.status})`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "خطای ناشناخته");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <p className="text-xs mb-3" style={{ color: "var(--muted)" }}>
          تخمین بر پایه قیمت‌های واقعی آگهی‌های مشابه در داده‌های فعلی — بدون
          مدل یادگیری، بدون حدس؛ هر تخمین همراه با نمونه‌های قابل‌بررسی ارائه
          می‌شود.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <input type="text" value={brand} onChange={(e) => setBrand(e.target.value)}
            placeholder="برند * (پژو)" />
          <input type="text" value={model} onChange={(e) => setModel(e.target.value)}
            placeholder="مدل (206)" />
          <input type="text" value={year} onChange={(e) => setYear(e.target.value)}
            placeholder="سال (1398)" />
          <input type="number" value={mileage} onChange={(e) => setMileage(e.target.value)}
            placeholder="کارکرد (45000)" />
        </div>
        <div className="mt-3 flex items-center justify-between gap-2">
          <label className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
            <input
              type="checkbox"
              checked={trimOutliers}
              onChange={(e) => setTrimOutliers(e.target.checked)}
              className="!w-auto"
            />
            حذف قیمت‌های پرت (IQR — فقط با ۶ نمونه یا بیشتر)
          </label>
          <button className="btn" onClick={submit} disabled={loading || !brand.trim()}>
            {loading ? "…" : "تخمین بازار"}
          </button>
        </div>
        <p className="mt-2 text-xs" style={{ color: "var(--muted)" }}>
          معیار انتخاب مشابه‌ها: همان برند (+مدل اگر وارد شود)، سال در بازهٔ ±۲
          از تقویم واحد، کارکرد در بازهٔ ±۳۰هزار کیلومتر — همه از دادهٔ واقعی
          آگهی‌ها.
        </p>
      </div>

      {error && (
        <div className="card p-4 text-sm" style={{ color: "#991b1b" }}>⚠️ {error}</div>
      )}

      {result && (
        <div className="card p-5">
          {result.quality === "no_data" ? (
            <p className="text-sm text-center" style={{ color: "var(--muted)" }}>
              {result.quality_note}
            </p>
          ) : (
            <>
              <div className="flex items-center gap-2 flex-wrap mb-3">
                <span className="chip score-chip">
                  {result.quality === "ok" ? "نمونهٔ کافی" : "نمونهٔ کم"}
                </span>
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                  {result.count} آگهی مشابه
                  {result.year_num !== null &&
                    ` · سال ${formatNumber(result.year_num)} (${
                      CALENDAR_LABELS[result.year_calendar ?? ""] ?? result.year_calendar
                    })`}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="p-3 rounded" style={{ background: "var(--background)" }}>
                  <p className="text-xs" style={{ color: "var(--muted)" }}>کم‌ترین (P25)</p>
                  <p className="num font-semibold">{formatToman(result.p25)}</p>
                </div>
                <div
                  className="p-3 rounded"
                  style={{ background: "var(--accent)", color: "#fff" }}
                >
                  <p className="text-xs" style={{ opacity: 0.85 }}>میانه بازار</p>
                  <p className="num font-bold">{formatToman(result.median)}</p>
                </div>
                <div className="p-3 rounded" style={{ background: "var(--background)" }}>
                  <p className="text-xs" style={{ color: "var(--muted)" }}>بیش‌ترین (P75)</p>
                  <p className="num font-semibold">{formatToman(result.p75)}</p>
                </div>
              </div>

              <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
                {result.quality_note}
              </p>

              {result.comparables.length > 0 && (
                <div className="mt-4">
                  <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>
                    نمونه‌های مبنای تخمین (نزدیک‌ترین به مشخصات شما):
                  </p>
                  <div className="space-y-2">
                    {result.comparables.map((c) => (
                      <div
                        key={c.listing_key}
                        className="flex items-center justify-between gap-2 p-2 rounded text-xs flex-wrap"
                        style={{ background: "var(--background)" }}
                      >
                        <span className={`chip platform-${c.source_platform}`}>
                          {PLATFORM_LABELS[c.source_platform] ?? c.source_platform}
                        </span>
                        <span className="flex-1 min-w-40">
                          {c.model_raw ?? c.title ?? "—"} · سال{" "}
                          <span className="num">{formatNumber(c.year_num)}</span> · کارکرد{" "}
                          <span className="num">{formatNumber(c.mileage_km)}</span>
                        </span>
                        <span className="num font-semibold">{formatToman(c.price_amount)}</span>
                        <button
                          className="underline"
                          style={{ color: "var(--accent)" }}
                          onClick={async () => {
                            const res = await fetch(
                              `${API_BASE}/listings/${c.listing_key.replace(":", "/")}`
                            );
                            if (res.ok) setDetail(await res.json());
                            onOpenDetail?.(c.listing_key);
                          }}
                        >
                          جزئیات
                        </button>
                        {c.original_url && (
                          <a href={c.original_url} target="_blank" rel="noopener noreferrer"
                            className="underline" style={{ color: "var(--muted)" }}>
                            منبع ↗
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {detail && <ListingDetailModal detail={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}

function ListingDetailModal({ detail, onClose }: { detail: ListingDetail; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{ background: "rgba(15,23,42,0.45)" }}
      onClick={onClose}
    >
      <div className="card p-5 max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-bold text-sm">{detail.title ?? "—"}</h3>
          <button onClick={onClose} className="btn-secondary text-xs">بستن ✕</button>
        </div>
        <p className="text-sm excerpt mb-3">{detail.description ?? "بدون توضیحات"}</p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm">
          {[
            ["سال", formatNumber(detail.year_raw)],
            ["کارکرد", formatNumber(detail.mileage_raw)],
            ["قیمت", formatToman(detail.price_amount)],
            ["رنگ", detail.color_raw],
            ["موقعیت", detail.location_raw],
            ["گیربکس", detail.transmission_raw],
          ].map(([label, value]) => (
            <div key={label} className="p-2 rounded" style={{ background: "var(--background)" }}>
              <p className="text-xs" style={{ color: "var(--muted)" }}>{label}</p>
              <p className="num">{value ?? "—"}</p>
            </div>
          ))}
        </div>
        {detail.original_url && (
          <a href={detail.original_url} target="_blank" rel="noopener noreferrer"
            className="underline text-sm mt-3 inline-block" style={{ color: "var(--accent)" }}>
            مشاهده آگهی اصلی ↗
          </a>
        )}
      </div>
    </div>
  );
}
