"use client";

import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";
import {
  CALENDAR_LABELS,
  EMPTY_FILTERS,
  FacetsApiResponse,
  FiltersState,
  fetchFacets,
  formatNumber,
} from "@/lib/api";

/** Database-driven filter panel: facets from /facets, dependent brand→model
    selectors, searchable dropdowns. No invented options. */

function SearchSelect({
  options,
  value,
  onChange,
  placeholder,
}: {
  options: { value: string; label: string; hint?: string }[];
  value: string[];
  onChange: (values: string[]) => void;
  placeholder: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () =>
      options.filter(
        (o) =>
          !query.trim() ||
          o.label.includes(query.trim()) ||
          (o.hint ?? "").includes(query.trim())
      ),
    [options, query]
  );

  const selectedLabels = value
    .map((v) => options.find((o) => o.value === v)?.label ?? v)
    .join(", ");

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full text-right"
        style={{
          border: "1px solid var(--border)",
          borderRadius: "0.6rem",
          padding: "0.45rem 0.75rem",
          background: "#fff",
          fontSize: "0.8125rem",
        }}
      >
        {selectedLabels || <span style={{ color: "var(--muted)" }}>{placeholder}</span>}
      </button>
      {open && (
        <div
          className="absolute z-30 mt-1 w-full rounded-lg border p-2 shadow-lg"
          style={{ background: "#fff", borderColor: "var(--border)" }}
        >
          <input
            type="text"
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="جستجو…"
            className="!mb-2"
          />
          <div className="max-h-48 overflow-y-auto">
            {filtered.length === 0 && (
              <p className="p-2 text-xs" style={{ color: "var(--muted)" }}>
                گزینه‌ای یافت نشد
              </p>
            )}
            {filtered.map((o) => {
              const checked = value.includes(o.value);
              return (
                <button
                  key={o.value}
                  type="button"
                  onClick={() =>
                    onChange(
                      checked
                        ? value.filter((v) => v !== o.value)
                        : [...value, o.value]
                    )
                  }
                  className="flex w-full items-center justify-between rounded px-2 py-1.5 text-right text-[13px] hover:bg-slate-50"
                >
                  <span className={checked ? "font-semibold" : ""}>{o.label}</span>
                  {o.hint && (
                    <span className="num text-[11px]" style={{ color: "var(--muted)" }}>
                      {o.hint}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          {value.length > 0 && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="mt-1 w-full rounded px-2 py-1 text-xs hover:bg-slate-50"
              style={{ color: "var(--danger)" }}
            >
              پاک‌کردن انتخاب
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function FilterPanel({
  filters,
  onChange,
  onReset,
}: {
  filters: FiltersState;
  onChange: (f: FiltersState) => void;
  onReset: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [facets, setFacets] = useState<FacetsApiResponse | null>(null);

  useEffect(() => {
    fetchFacets()
      .then(setFacets)
      .catch(() => setFacets(null));
  }, []);

  const set = (patch: Partial<FiltersState>) => onChange({ ...filters, ...patch });

  const toggle = (field: "platforms", v: string) => {
    const current = filters.platforms;
    set({ platforms: current.includes(v) ? current.filter((x) => x !== v) : [...current, v] });
  };
  const toggleCsv = (field: "fuel" | "transmission" | "body_type", v: string) => {
    const current = (filters[field] as string).split(",").filter(Boolean);
    const next = current.includes(v) ? current.filter((x) => x !== v) : [...current, v];
    set({ [field]: next.join(",") } as Partial<FiltersState>);
  };

  const fuelList = filters.fuel.split(",").filter(Boolean);
  const transmissionList = filters.transmission.split(",").filter(Boolean);
  const bodyTypeList = filters.body_type.split(",").filter(Boolean);
  const activeModels = filters.models.split(",").filter(Boolean);
  const selectedBrandObjs = facets?.brands.filter((b) =>
    filters.brands.split(",").filter(Boolean).includes(b.key)
  ) ?? [];
  const modelOptions = (selectedBrandObjs.length > 0
    ? selectedBrandObjs.flatMap((b) => b.models.map((m) => ({ value: m.label, label: m.label, hint: String(m.count) })))
    : facets?.brands.flatMap((b) => b.models.map((m) => ({ value: m.label, label: m.label, hint: String(m.count) }))) ?? []
  ).filter((o, i, arr) => arr.findIndex((x) => x.value === o.value) === i);

  const activeChips: { label: string; clear: () => void }[] = [];
  for (const p of filters.platforms)
    activeChips.push({ label: `منبع: ${p}`, clear: () => set({ platforms: filters.platforms.filter((x) => x !== p) }) });
  for (const b of filters.brands.split(",").filter(Boolean))
    activeChips.push({ label: `برند: ${facets?.brands.find((x) => x.key === b)?.label ?? b}`, clear: () => set({ brands: filters.brands.split(",").filter((x) => x !== b).join(","), models: "" }) });
  for (const m of activeModels)
    activeChips.push({ label: `مدل: ${m}`, clear: () => set({ models: activeModels.filter((x) => x !== m).join(",") }) });
  if (filters.price_min) activeChips.push({ label: `از ${formatNumber(filters.price_min)} تومان`, clear: () => set({ price_min: "" }) });
  if (filters.price_max) activeChips.push({ label: `تا ${formatNumber(filters.price_max)} تومان`, clear: () => set({ price_max: "" }) });
  if (filters.year_min) activeChips.push({ label: `از سال ${formatNumber(filters.year_min)}`, clear: () => set({ year_min: "" }) });
  if (filters.year_max) activeChips.push({ label: `تا سال ${formatNumber(filters.year_max)}`, clear: () => set({ year_max: "" }) });
  if (filters.year_calendar) activeChips.push({ label: `تقویم: ${CALENDAR_LABELS[filters.year_calendar]}`, clear: () => set({ year_calendar: "" }) });
  if (filters.mileage_max_km) activeChips.push({ label: `کارکرد تا ${formatNumber(filters.mileage_max_km)}`, clear: () => set({ mileage_max_km: "" }) });
  if (filters.location_contains.trim()) activeChips.push({ label: `موقعیت: ${filters.location_contains}`, clear: () => set({ location_contains: "" }) });
  for (const f of fuelList) activeChips.push({ label: `سوخت: ${f}`, clear: () => set({ fuel: fuelList.filter((x) => x !== f).join(",") }) });
  for (const t of transmissionList) activeChips.push({ label: `گیربکس: ${t}`, clear: () => set({ transmission: transmissionList.filter((x) => x !== t).join(",") }) });
  for (const b of bodyTypeList) activeChips.push({ label: `بدنه: ${b}`, clear: () => set({ body_type: bodyTypeList.filter((x) => x !== b).join(",") }) });

  return (
    <div className="card mb-4 p-4">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 text-sm font-semibold"
          aria-expanded={open}
        >
          🔎 فیلترها
          {activeChips.length > 0 && (
            <span className="chip score-mid num">فعال: {formatNumber(activeChips.length)}</span>
          )}
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {open ? "▲" : "▼"}
          </span>
        </button>
        {activeChips.length > 0 && (
          <button onClick={onReset} className="text-xs underline" style={{ color: "var(--danger)" }}>
            پاک‌کردن همه
          </button>
        )}
      </div>

      {open && (
        <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>برند</label>
            <SearchSelect
              placeholder="انتخاب برند"
              value={filters.brands.split(",").filter(Boolean)}
              onChange={(keys) => set({ brands: keys.join(","), models: "" })}
              options={(facets?.brands ?? []).map((b) => ({ value: b.key, label: b.label, hint: String(b.count) }))}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>
              مدل {selectedBrandObjs.length > 0 && "(بر اساس برند انتخابی)"}
            </label>
            <SearchSelect
              placeholder="انتخاب مدل"
              value={activeModels}
              onChange={(labels) => set({ models: labels.join(",") })}
              options={modelOptions}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>سوخت</label>
            <SearchSelect
              placeholder="همه"
              value={fuelList}
              onChange={(v) => set({ fuel: v.join(",") })}
              options={(facets?.fuel ?? []).map((f) => ({ value: f, label: f }))}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>گیربکس</label>
            <SearchSelect
              placeholder="همه"
              value={transmissionList}
              onChange={(v) => set({ transmission: v.join(",") })}
              options={(facets?.transmission ?? []).map((t) => ({ value: t, label: t }))}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>نوع بدنه</label>
            <SearchSelect
              placeholder="همه"
              value={bodyTypeList}
              onChange={(v) => set({ body_type: v.join(",") })}
              options={(facets?.body_type ?? []).map((b) => ({ value: b, label: b }))}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>موقعیت</label>
            <SearchSelect
              placeholder="همه"
              value={filters.location_contains.trim() ? [filters.location_contains.trim()] : []}
              onChange={(v) => set({ location_contains: v[0] ?? "" })}
              options={(facets?.locations ?? []).map((l) => ({ value: l.label, label: l.label, hint: String(l.count) }))}
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>حداقل قیمت (تومان)</label>
              <input type="number" value={filters.price_min}
                onChange={(e) => set({ price_min: e.target.value })} placeholder="۰" />
            </div>
            <div>
              <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>حداکثر قیمت (تومان)</label>
              <input type="number" value={filters.price_max}
                onChange={(e) => set({ price_max: e.target.value })} placeholder="۵۰۰۰۰۰۰۰۰۰" />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>سال از</label>
              <input type="number" value={filters.year_min}
                onChange={(e) => set({ year_min: e.target.value })} placeholder="1398" />
            </div>
            <div>
              <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>سال تا</label>
              <input type="number" value={filters.year_max}
                onChange={(e) => set({ year_max: e.target.value })} placeholder="1404" />
            </div>
            <div>
              <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>تقویم</label>
              <select value={filters.year_calendar}
                onChange={(e) => set({ year_calendar: e.target.value })}>
                <option value="">هر دو</option>
                {Object.entries(CALENDAR_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <label className="mb-1 block text-xs" style={{ color: "var(--muted)" }}>حداکثر کارکرد (کیلومتر)</label>
            <input type="number" value={filters.mileage_max_km}
              onChange={(e) => set({ mileage_max_km: e.target.value })} placeholder="۶۰۰۰۰" />
          </div>
        </div>
      )}

      {activeChips.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {activeChips.map((chip, i) => (
            <button
              key={i}
              onClick={chip.clear}
              className="chip score-mid flex items-center gap-1"
              title="حذف این فیلتر"
            >
              {chip.label} <X size={11} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
