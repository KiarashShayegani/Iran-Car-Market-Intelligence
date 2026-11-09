"use client";

import { useState } from "react";
import {
  CALENDAR_LABELS,
  FiltersState,
  PLATFORM_LABELS,
} from "@/lib/api";

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
  const set = (patch: Partial<FiltersState>) => onChange({ ...filters, ...patch });

  const togglePlatform = (key: string) => {
    const next = filters.platforms.includes(key)
      ? filters.platforms.filter((p) => p !== key)
      : [...filters.platforms, key];
    set({ platforms: next });
  };

  return (
    <div className="card p-4 mb-4">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setOpen(!open)}
          className="font-semibold text-sm"
          aria-expanded={open}
        >
          🔎 فیلترها {filtersActive(filters) && <span className="chip score-chip">فعال</span>}
          <span className="mr-1 text-xs" style={{ color: "var(--muted)" }}>
            {open ? "▲" : "▼"}
          </span>
        </button>
        {filtersActive(filters) && (
          <button onClick={onReset} className="text-xs underline" style={{ color: "var(--muted)" }}>
            پاک‌کردن همه
          </button>
        )}
      </div>

      {open && (
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
          <div>
            <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
              منبع
            </label>
            <div className="flex gap-2">
              {Object.entries(PLATFORM_LABELS).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => togglePlatform(key)}
                  className={`chip ${filters.platforms.includes(key) ? `platform-${key}` : "btn-secondary"}`}
                  style={filters.platforms.includes(key) ? {} : { border: "1px solid var(--border)" }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
              برند (مثلاً: پژو، تویوتا)
            </label>
            <input
              type="text"
              value={filters.brands}
              onChange={(e) => set({ brands: e.target.value })}
              placeholder="پژو"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
                حداقل قیمت (تومان)
              </label>
              <input
                type="number"
                value={filters.price_min}
                onChange={(e) => set({ price_min: e.target.value })}
                placeholder="۰"
              />
            </div>
            <div>
              <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
                حداکثر قیمت (تومان)
              </label>
              <input
                type="number"
                value={filters.price_max}
                onChange={(e) => set({ price_max: e.target.value })}
                placeholder="۵۰۰۰۰۰۰۰۰۰"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
                سال از
              </label>
              <input
                type="number"
                value={filters.year_min}
                onChange={(e) => set({ year_min: e.target.value })}
                placeholder="1398"
              />
            </div>
            <div>
              <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
                سال تا
              </label>
              <input
                type="number"
                value={filters.year_max}
                onChange={(e) => set({ year_max: e.target.value })}
                placeholder="1404"
              />
            </div>
            <div>
              <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
                تقویم
              </label>
              <select
                value={filters.year_calendar}
                onChange={(e) => set({ year_calendar: e.target.value })}
              >
                <option value="">هر دو</option>
                {Object.entries(CALENDAR_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
              حداکثر کارکرد (کیلومتر)
            </label>
            <input
              type="number"
              value={filters.mileage_max_km}
              onChange={(e) => set({ mileage_max_km: e.target.value })}
              placeholder="۶۰۰۰۰"
            />
          </div>

          <div>
            <label className="block mb-1 text-xs" style={{ color: "var(--muted)" }}>
              شهر / موقعیت
            </label>
            <input
              type="text"
              value={filters.location_contains}
              onChange={(e) => set({ location_contains: e.target.value })}
              placeholder="تهران"
            />
          </div>

          <p className="col-span-full text-xs" style={{ color: "var(--muted)" }}>
            فیلترها پیش از رتبه‌بندی اعمال می‌شوند؛ نتیجهٔ خالی یعنی واقعاً هیچ
            آگهی مطابقت ندارد. آگهی‌های بدون قیمت/کارکردِ مشخص با فعال‌شدن
            فیلتر مربوطه حذف می‌شوند. سال‌ها به دو تقویم شمسی و میلادی
            تفکیک می‌شوند — برای مقایسه درست، تقویم را مشخص کنید.
          </p>
        </div>
      )}
    </div>
  );
}

function filtersActive(f: FiltersState): boolean {
  return Boolean(
    f.platforms.length ||
      f.brands.trim() ||
      f.price_min ||
      f.price_max ||
      f.year_min ||
      f.year_max ||
      f.year_calendar ||
      f.mileage_max_km ||
      f.location_contains.trim()
  );
}
