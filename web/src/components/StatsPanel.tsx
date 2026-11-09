"use client";

import { useEffect, useState } from "react";
import {
  PLATFORM_LABELS,
  StatsApiResponse,
  fetchStats,
  formatNumber,
} from "@/lib/api";

export default function StatsPanel() {
  const [stats, setStats] = useState<StatsApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
  }, []);

  if (error)
    return (
      <div className="card p-6 text-sm" style={{ color: "#991b1b" }}>
        ⚠️ {error} — مطمئن شوید سرور API روی پورت ۸۰۰۰ اجرا شده است.
      </div>
    );
  if (!stats) return <div className="card p-6 text-sm">در حال دریافت آمار…</div>;

  const maxYearMedian = Math.max(
    ...stats.by_year.map((r) => Number(r.median ?? 0)),
    1
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {stats.overview.map((row) => (
          <div key={String(row.source_platform)} className="card p-4">
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              {PLATFORM_LABELS[String(row.source_platform)] ?? row.source_platform}
            </p>
            <p className="text-2xl font-bold mt-1 num">
              {formatNumber(Number(row.count))}
            </p>
            <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
              آگهی دارای قیمت
            </p>
            <p className="text-sm mt-2 num">
              میانه: {formatTomanShort(Number(row.median))}
            </p>
          </div>
        ))}
      </div>

      <div className="card p-4">
        <h3 className="font-semibold text-sm mb-3">میانه قیمت برند (پرآگهی‌ترین‌ها)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs" style={{ color: "var(--muted)" }}>
                <th className="text-right p-2">برند</th>
                <th className="text-right p-2">منبع</th>
                <th className="text-right p-2">تعداد</th>
                <th className="text-right p-2">میانه قیمت</th>
                <th className="text-right p-2">میانه سال</th>
                <th className="text-right p-2">تقویم‌ها</th>
              </tr>
            </thead>
            <tbody>
              {stats.by_brand.slice(0, 12).map((row, i) => (
                <tr key={i} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="p-2 font-medium">{String(row.brand_raw ?? "—")}</td>
                  <td className="p-2 text-xs">
                    {PLATFORM_LABELS[String(row.source_platform)] ?? row.source_platform}
                  </td>
                  <td className="p-2 num">{formatNumber(Number(row.count))}</td>
                  <td className="p-2 num">{formatTomanShort(Number(row.median))}</td>
                  <td className="p-2 num">{formatNumber(Number(row.median_year ?? 0))}</td>
                  <td className="p-2 text-xs">{String(row.calendars ?? "—")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-4">
        <h3 className="font-semibold text-sm mb-3">
          میانه قیمت بر اساس سال تولید (تقویم‌ها جداگانه — هرگز ترکیب نمی‌شوند)
        </h3>
        <div className="space-y-2">
          {stats.by_year.slice(0, 14).map((row, i) => {
            const median = Number(row.median ?? 0);
            const width = Math.max((median / maxYearMedian) * 100, 4);
            return (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-14 num">{formatNumber(Number(row.year_num))}</span>
                <span
                  className="chip"
                  style={{
                    background:
                      row.year_calendar === "shamsi" ? "#ccfbf1" : "#e0e7ff",
                    color: row.year_calendar === "shamsi" ? "#0f766e" : "#4338ca",
                  }}
                >
                  {row.year_calendar === "shamsi" ? "شمسی" : "میلادی"}
                </span>
                <div
                  className="h-4 rounded"
                  style={{ width: `${width}%`, background: "var(--accent)", opacity: 0.7 }}
                />
                <span className="num w-28">{formatTomanShort(median)}</span>
                <span className="num" style={{ color: "var(--muted)" }}>
                  ({formatNumber(Number(row.count))} آگهی)
                </span>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
          نمودارها فقط روی آگهی‌های دارای قیمتِ اعلام‌شده محاسبه شده‌اند؛ آگهی‌های
          توافقی (قیمت صفر) هیچ‌گاه وارد آمار نمی‌شوند.
        </p>
      </div>
    </div>
  );
}

function formatTomanShort(amount: number): string {
  const b = amount / 1_000_000_000;
  return `${new Intl.NumberFormat("fa-IR", { maximumFractionDigits: 2 }).format(b)} میلیارد`;
}
