"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  PLATFORM_LABELS,
  fetchFacets,
  formatNumber,
  formatToman,
} from "@/lib/api";
import PlatformMark from "@/components/PlatformMark";

interface AnalyticsResponse {
  total: number;
  insufficient: string | null;
  kpis: Record<string, number | null>;
  by_brand: { label: string; count: number }[];
  by_model: { brand: string; label: string; count: number }[];
  by_platform: { platform: string; count: number }[];
  by_year: { year: number; calendar: string; count: number; median: number | null }[];
  by_mileage: { bucket: string; count: number; median: number | null }[];
  price_trend: { date: string; count: number; median: number | null; p25: number | null; p75: number | null }[];
}

const fmtB = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `${new Intl.NumberFormat("fa-IR", { maximumFractionDigits: 1 }).format(v / 1e9)} میلیارد`;

const fmtAxisB = (v: number) => `${(v / 1e9).toFixed(1)}`;

export default function AnalyticsPanel() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/analytics`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`خطا ${r.status}`))))
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
  }, []);

  if (error) return <div className="card p-6 text-sm" style={{ color: "var(--danger)" }}>⚠️ {error}</div>;
  if (!data) return <div className="card p-6 text-sm">در حال محاسبهٔ آمار بازار…</div>;

  const kpi = data.kpis ?? {};
  const trend = data.price_trend.map((r) => ({ ...r, medianB: (r.median ?? 0) / 1e9, p25B: (r.p25 ?? 0) / 1e9, p75B: (r.p75 ?? 0) / 1e9 }));
  const years = data.by_year.map((r) => ({ ...r, medianB: (r.median ?? 0) / 1e9 }));

  return (
    <div className="space-y-5">
      {data.insufficient && (
        <div className="rounded-xl p-3 text-xs" style={{ background: "var(--warning-bg)", color: "var(--warning-text)" }}>
          ⓘ {data.insufficient}
        </div>
      )}

      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi label="کل آگهی‌های دارای قیمت" value={formatNumber(kpi.total)} />
        <Kpi label="میانهٔ قیمت" value={fmtB(kpi.median)} accent />
        <Kpi label="برندها / مدل‌ها" value={`${formatNumber(kpi.brands)} / ${formatNumber(kpi.models)}`} />
        <Kpi label="میانهٔ کارکرد" value={kpi.median_mileage ? `${formatNumber(kpi.median_mileage)} km` : "—"} />
      </div>

      {/* price trend */}
      <div className="card p-4">
        <h3 className="mb-3 text-sm font-semibold">روند میانهٔ قیمت بر اساس اجرای جمع‌آوری</h3>
        {trend.length < 2 ? (
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            هنوز دادهٔ تاریخی کافی برای روند وجود ندارد — با اجراهای بعدی جمع‌آوری، این نمودار کامل می‌شود.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={trend} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fontFamily: "Vazirmatn" }} />
              <YAxis tickFormatter={fmtAxisB} tick={{ fontSize: 11 }} width={70} />
              <Tooltip
                formatter={(v: unknown) => fmtB(Number(v) * 1e9)}
                contentStyle={{ fontFamily: "Vazirmatn", direction: "rtl", borderRadius: 8, fontSize: 12 }}
              />
              <Line type="monotone" dataKey="medianB" name="میانه" stroke="#176b70" strokeWidth={2.5} dot />
              <Line type="monotone" dataKey="p75B" name="P75" stroke="#e6a23c" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
              <Line type="monotone" dataKey="p25B" name="P25" stroke="#5a7da0" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
        <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
          سری‌زمانی بر اساس اجراهای جمع‌آوری است — آمار قیمت «آگهی» است، نه معاملهٔ واقعی.
        </p>
      </div>

      {/* brand distribution */}
      <div className="card p-4">
        <h3 className="mb-3 text-sm font-semibold">پرتکرارترین برندها (بر اساس تعداد آگهی)</h3>
        <ResponsiveContainer width="100%" height={Math.max(data.by_brand.length * 34, 160)}>
          <BarChart data={data.by_brand} layout="vertical" margin={{ top: 4, right: 20, left: 20, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="label" tick={{ fontSize: 12, fontFamily: "Vazirmatn" }} width={90} />
            <Tooltip formatter={(v: unknown) => formatNumber(Number(v))} contentStyle={{ fontFamily: "Vazirmatn", direction: "rtl", borderRadius: 8 }} />
            <Bar dataKey="count" fill="#176b70" radius={[0, 6, 6, 0]} barSize={16} />
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
          تعداد آگهی است، نه تعداد خودروی یکتا — یک خودرو ممکن است در چند پلتفرم آگهی شده باشد.
        </p>
      </div>

      {/* year distribution */}
      <div className="card p-4">
        <h3 className="mb-3 text-sm font-semibold">میانهٔ قیمت بر اساس سال تولید (تقویم‌ها جداگانه)</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={years} margin={{ top: 4, right: 8, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="year" tick={{ fontSize: 11 }} />
            <YAxis tickFormatter={fmtAxisB} tick={{ fontSize: 11 }} width={70} />
            <Tooltip
              formatter={(v: unknown, name: unknown) =>
                name === "medianB" ? fmtB(Number(v) * 1e9) : formatNumber(Number(v))
              }
              contentStyle={{ fontFamily: "Vazirmatn", direction: "rtl", borderRadius: 8 }}
            />
            <Bar dataKey="count" name="تعداد" fill="#5a7da0" radius={[6, 6, 0, 0]} barSize={18} />
            <Bar dataKey="medianB" name="میانه (میلیارد)" fill="#e6a23c" radius={[6, 6, 0, 0]} barSize={18} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* platform distribution */}
      <div className="card p-4">
        <h3 className="mb-3 text-sm font-semibold">پوشش منابع</h3>
        <div className="space-y-2">
          {data.by_platform.map((row) => (
            <div key={row.platform} className="flex items-center gap-2 text-xs">
              <PlatformMark platform={row.platform} />
              <div
                className="h-4 rounded"
                style={{
                  width: `${Math.max((row.count / Math.max(...data.by_platform.map((r) => r.count))) * 100, 4)}%`,
                  background: "var(--gradient-brand)",
                }}
              />
              <span className="num" style={{ color: "var(--muted)" }}>{formatNumber(row.count)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Kpi({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="card p-4 text-center">
      <p className={`num text-xl font-extrabold ${accent ? "" : ""}`}
        style={accent ? { background: "var(--gradient-brand)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" } : {}}>
        {value}
      </p>
      <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>{label}</p>
    </div>
  );
}
