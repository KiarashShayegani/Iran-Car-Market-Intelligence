"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  PLATFORM_LABELS,
  fetchFacets,
  formatDateJalaliShort,
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

interface DashboardFilters {
  brand: string;
  model: string;
  year_min: string;
  year_max: string;
  fuel: string;
  transmission: string;
  platform: string;
}

const EMPTY_FILTERS: DashboardFilters = {
  brand: "", model: "", year_min: "", year_max: "", fuel: "", transmission: "", platform: "",
};

export default function AnalyticsPanel() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<DashboardFilters>(EMPTY_FILTERS);
  const [facets, setFacets] = useState<Awaited<ReturnType<typeof fetchFacets>> | null>(null);

  useEffect(() => {
    fetchFacets().then(setFacets).catch(() => {});
  }, []);

  // refetch whenever filters change (params mirror the /analytics contract)
  useEffect(() => {
    const p = new URLSearchParams();
    if (filters.brand) p.set("brands", filters.brand);
    if (filters.model) p.set("models", filters.model);
    if (filters.year_min) p.set("year_min", filters.year_min);
    if (filters.year_max) p.set("year_max", filters.year_max);
    if (filters.fuel) p.set("fuel", filters.fuel);
    if (filters.transmission) p.set("transmission", filters.transmission);
    if (filters.platform) p.set("platforms", filters.platform);
    fetch(`${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/analytics?${p}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`خطا ${r.status}`))))
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
  }, [filters]);

  const activeCount = Object.values(filters).filter(Boolean).length;
  const selectedBrandFacet = facets?.brands.find((b) => b.label === filters.brand) ?? null;
  const yearLo = facets?.bounds.min_year ? Math.max(1360, facets.bounds.min_year) : 1360;
  const yearHi = facets?.bounds.max_year ? Math.min(1406, facets.bounds.max_year) : 1406;
  const yearOptions: number[] = [];
  for (let y = yearHi; y >= yearLo; y--) yearOptions.push(y);

  const kpi = data?.kpis ?? {};
  const trend = (data?.price_trend ?? []).map((r) => ({
    ...r,
    dateJalali: formatDateJalaliShort(r.date),
    medianB: (r.median ?? 0) / 1e9,
    p25B: (r.p25 ?? 0) / 1e9,
    p75B: (r.p75 ?? 0) / 1e9,
  }));
  const mileage = (data?.by_mileage ?? []).map((r) => ({
    ...r,
    medianB: (r.median ?? 0) / 1e9,
  }));
  const years = (data?.by_year ?? [])
    .filter((r) => (r.calendar === "shamsi" ? r.year >= 1330 && r.year <= 1406 : r.year >= 1940 && r.year <= 2030))
    .map((r) => ({
      yearJalali: r.calendar === "gregorian" ? r.year - 621 : r.year,
      ...r,
      medianB: (r.median ?? 0) / 1e9,
    }))
    .sort((a, b) => a.yearJalali - b.yearJalali);

  return (
    <div className="space-y-5">
      {error && (
        <div className="card p-6 text-sm" style={{ color: "var(--danger)" }}>⚠️ {error}</div>
      )}

      {/* interactive filter bar */}
      <div className="card p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold">فیلترهای بازار</h3>
          {activeCount > 0 && (
            <button
              className="text-xs underline"
              style={{ color: "var(--accent)" }}
              onClick={() => setFilters(EMPTY_FILTERS)}
            >
              پاک‌کردن ({activeCount.toLocaleString("fa-IR")})
            </button>
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
          <select className="!py-1.5 !text-xs" value={filters.brand}
            onChange={(e) => setFilters((f) => ({ ...f, brand: e.target.value, model: "" }))}>
            <option value="">همهٔ برندها</option>
            {(facets?.brands ?? []).map((b) => (
              <option key={b.key} value={b.label}>{b.label}</option>
            ))}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.model} disabled={!selectedBrandFacet}
            onChange={(e) => setFilters((f) => ({ ...f, model: e.target.value }))}>
            <option value="">همهٔ مدل‌ها</option>
            {(selectedBrandFacet?.models ?? []).slice(0, 80).map((m) => (
              <option key={m.label} value={m.label}>{m.label}</option>
            ))}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.year_min}
            onChange={(e) => setFilters((f) => ({ ...f, year_min: e.target.value }))}>
            <option value="">از سال…</option>
            {yearOptions.map((y) => <option key={y} value={String(y)}>{y.toLocaleString("fa-IR")}</option>)}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.year_max}
            onChange={(e) => setFilters((f) => ({ ...f, year_max: e.target.value }))}>
            <option value="">تا سال…</option>
            {yearOptions.map((y) => <option key={y} value={String(y)}>{y.toLocaleString("fa-IR")}</option>)}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.fuel}
            onChange={(e) => setFilters((f) => ({ ...f, fuel: e.target.value }))}>
            <option value="">همهٔ سوخت‌ها</option>
            {(facets?.fuel ?? []).map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.transmission}
            onChange={(e) => setFilters((f) => ({ ...f, transmission: e.target.value }))}>
            <option value="">همهٔ گیربکس‌ها</option>
            {(facets?.transmission ?? []).map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
          <select className="!py-1.5 !text-xs" value={filters.platform}
            onChange={(e) => setFilters((f) => ({ ...f, platform: e.target.value }))}>
            <option value="">همهٔ منابع</option>
            {Object.entries(PLATFORM_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      {data?.insufficient && (
        <div className="rounded-xl p-3 text-xs" style={{ background: "var(--warning-bg)", color: "var(--warning-text)" }}>
          ⓘ {data.insufficient}
        </div>
      )}

      {!data && !error && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="card p-4">
              <div className="skeleton h-6 w-2/3 mx-auto" />
              <div className="skeleton h-3 w-1/2 mx-auto mt-2" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <>
          <section>
            <h2 className="mb-2 text-sm font-semibold">نمای کلی بازار (با اعمال فیلترها)</h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <Kpi label="آگهی‌های دارای قیمت" value={formatNumber(kpi.total)} />
              <Kpi label="میانهٔ قیمت" value={fmtB(kpi.median)} accent />
              <Kpi label="ربع اول (P25)" value={fmtB(kpi.p25)} />
              <Kpi label="ربع سوم (P75)" value={fmtB(kpi.p75)} />
              <Kpi label="برندها / مدل‌ها" value={`${formatNumber(kpi.brands)} / ${formatNumber(kpi.models)}`} />
              <Kpi label="میانهٔ کارکرد" value={kpi.median_mileage ? `${formatNumber(kpi.median_mileage)} km` : "—"} />
            </div>
          </section>

          <section className="card p-4">
            <h3 className="mb-3 text-sm font-semibold">روند میانهٔ قیمت بر اساس اجرای جمع‌آوری</h3>
            {trend.length < 2 ? (
              <p className="text-xs" style={{ color: "var(--muted)" }}>
                هنوز دادهٔ تاریخی کافی برای روند وجود ندارد — با اجراهای بعدی جمع‌آوری، این نمودار کامل می‌شود.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={trend} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="dateJalali" tick={{ fontSize: 11, fontFamily: "Vazirmatn" }} />
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
              هر نقطه = میانهٔ قیمت آگهی‌های دارای قیمت در «یک اجرای جمع‌آوری»
              (دو بار در روز به‌صورت زمان‌بندی‌شده). آمار قیمت «آگهی» است، نه
              معاملهٔ واقعی؛ با بیشتر شدن تاریخ داده، نقاط کامل‌تر می‌شوند.
            </p>
          </section>

          <section className="card p-4">
            <h3 className="mb-3 text-sm font-semibold">میانهٔ قیمت بر اساس کارکرد</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={mileage} margin={{ top: 4, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="bucket" tick={{ fontSize: 10, fontFamily: "Vazirmatn" }} />
                <YAxis tickFormatter={fmtAxisB} tick={{ fontSize: 11 }} width={70} />
                <Tooltip
                  formatter={(v: unknown, name: unknown) =>
                    name === "medianB" ? fmtB(Number(v) * 1e9) : formatNumber(Number(v))
                  }
                  contentStyle={{ fontFamily: "Vazirmatn", direction: "rtl", borderRadius: 8 }}
                />
                <Bar dataKey="medianB" name="میانهٔ قیمت (میلیارد)" fill="#176b70" radius={[6, 6, 0, 0]} barSize={26} />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
              هر ستون = میانهٔ قیمت آگهی‌های همان بازهٔ کارکرد — الگوی افت قیمت با کارکرد را نشان می‌دهد.
            </p>
          </section>

          <section className="card p-4">
            <h3 className="mb-3 text-sm font-semibold">پرتکرارترین برندها (بر اساس تعداد آگهی)</h3>
            <ResponsiveContainer width="100%" height={Math.max((data.by_brand ?? []).length * 34, 160)}>
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
          </section>

          <section className="card p-4">
            <h3 className="mb-3 text-sm font-semibold">میانهٔ قیمت بر اساس سال تولید (سال شمسی)</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={years} margin={{ top: 4, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="yearJalali" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={fmtAxisB} tick={{ fontSize: 11 }} width={70} />
                <Tooltip
                  formatter={(v: unknown, name: unknown) =>
                    name === "medianB" ? fmtB(Number(v) * 1e9) : formatNumber(Number(v))
                  }
                  labelFormatter={(y) => `سال ${Number(y).toLocaleString("fa-IR")}`}
                  contentStyle={{ fontFamily: "Vazirmatn", direction: "rtl", borderRadius: 8 }}
                />
                <Bar dataKey="count" name="تعداد" fill="#5a7da0" radius={[6, 6, 0, 0]} barSize={18} />
                <Bar dataKey="medianB" name="میانه (میلیارد)" fill="#e6a23c" radius={[6, 6, 0, 0]} barSize={18} />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-[11px]" style={{ color: "var(--muted)" }}>
              سال‌های میلادی منبع به تقویم شمسی نرمال‌سازی شده‌اند (تقریب مدل‌سال)؛ سال‌های غیرواقعی از نمودار حذف شده‌اند.
            </p>
          </section>

          <section className="card p-4">
            <h3 className="mb-3 text-sm font-semibold">پوشش منابع</h3>
            <div className="space-y-2">
              {(data.by_platform ?? []).map((row) => (
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
          </section>
        </>
      )}
    </div>
  );
}

function Kpi({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="card p-4 text-center">
      <p className="num text-xl font-extrabold"
        style={accent ? { background: "var(--gradient-brand)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" } : {}}>
        {value}
      </p>
      <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>{label}</p>
    </div>
  );
}
