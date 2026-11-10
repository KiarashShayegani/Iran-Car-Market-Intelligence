"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import VehicleImage from "@/components/VehicleImage";
import {
  CALENDAR_LABELS,
  CompareApiResponse,
  CompareVehicle,
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
      <div className="card p-6 text-sm" style={{ color: "#991b1b" }}>
        ⚠️ {error}
      </div>
    );
  if (!data) return <div className="card p-6 text-sm">در حال دریافت…</div>;
  const vehicles = data.vehicles;

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
                <p className="mt-1 text-xs num" style={{ color: "var(--muted)" }}>
                  {v.listing.listing_key}
                </p>
                <a
                  href={v.listing.original_url ?? undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-block text-xs underline"
                  style={{ color: "var(--accent)" }}
                >
                  آگهی اصلی ↗
                </a>
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
    </div>
  );
}
