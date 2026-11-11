"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Search as SearchIcon } from "lucide-react";
import VehicleImage from "@/components/VehicleImage";
import {
  DEMO_QUERIES,
  PLATFORM_LABELS,
  fetchFacets,
  fetchLatest,
  formatNumber,
  formatToman,
} from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [latest, setLatest] = useState<Awaited<ReturnType<typeof fetchLatest>> | null>(null);
  const [facets, setFacets] = useState<Awaited<ReturnType<typeof fetchFacets>> | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchLatest(8), fetchFacets()])
      .then(([l, f]) => {
        setLatest(l);
        setFacets(f);
      })
      .catch(() =>
        setApiError("سرور API در دسترس نیست — برای راه‌اندازی: scripts/run_api.py")
      );
  }, []);

  const submit = (q: string) => {
    if (!q.trim()) return;
    router.push(`/search?q=${encodeURIComponent(q.trim())}`);
  };

  return (
    <div className="space-y-8">
      {/* hero */}
      <section
        className="p-8 text-center rounded-[1.1rem]"
        style={{ background: "var(--gradient-hero)", color: "#fff" }}
      >
        <h1 className="text-2xl font-extrabold tracking-tight">بازار هوشمند خودرو ایران</h1>
        <p className="mt-2 text-sm" style={{ color: "rgba(255,255,255,0.82)" }}>
          جستجوی معنایی آگهی‌ها از باما، شیپور و همراه‌مکانیک — با فیلترهای
          دقیق و ارجاع به آگهی اصلی
        </p>
        <div className="mx-auto mt-4 flex max-w-xl gap-2">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit(query)}
            placeholder="مثلاً: ماشین اتوماتیک کم کارکرد خانوادگی"
          />
          <button className="btn shrink-0 flex items-center gap-1.5" onClick={() => submit(query)}>
            <SearchIcon size={16} />
            جستجو
          </button>
        </div>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {DEMO_QUERIES.slice(0, 3).map((dq) => (
            <button
              key={dq}
              onClick={() => submit(dq)}
              className="chip"
              style={{ background: "rgba(255,255,255,0.14)", color: "#fff" }}
            >
              {dq}
            </button>
          ))}
        </div>
        {apiError && (
          <p className="mt-3 text-xs" style={{ color: "#fecaca" }}>
            ⚠️ {apiError}
          </p>
        )}
      </section>

      {/* freshness + coverage strip */}
      {facets && (
        <section className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
          <FreshnessCard label="کل آگهی‌ها" value={formatNumber(facets.freshness.total_listings)} />
          <FreshnessCard label="دارای قیمت" value={formatNumber(facets.freshness.priced_listings)} />
          <FreshnessCard label="برندها" value={formatNumber(facets.brands.length)} />
          <FreshnessCard
            label="آخرین جمع‌آوری"
            value={(facets.freshness.newest_collection ?? "—").slice(0, 10)}
            small
          />
        </section>
      )}

      {/* brand discovery */}
      {facets && (
        <section>
          <h2 className="mb-2 text-sm font-semibold">کاوش بر اساس برند</h2>
          <div className="flex flex-wrap gap-2">
            {facets.brands.slice(0, 14).map((b) => (
              <button
                key={b.key}
                className="btn-secondary num"
                onClick={() => router.push(`/search?q=${encodeURIComponent(b.label ?? "")}`)}
              >
                {b.label} <span style={{ color: "var(--muted)" }}>({formatNumber(b.count)})</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* latest listings */}
      <section>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold">تازه‌ترین آگهی‌های جمع‌آوری‌شده</h2>
          <button
            className="text-xs underline"
            style={{ color: "var(--accent)" }}
            onClick={() => router.push("/search")}
          >
            همهٔ نتایج →
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {latest?.listings.map((card) => (
            <button
              key={card.listing_key}
              onClick={() => {
                const [platform, id] = card.listing_key.split(":");
                router.push(`/listings/${platform}/${id}`);
              }}
              className="card card-hover overflow-hidden text-right"
            >
              <VehicleImage
                src={card.image_url}
                alt={card.title ?? card.listing_key}
                className="aspect-[4/3] w-full"
              />
              <div className="p-3">
                <p className="text-xs font-semibold truncate">
                  {[card.brand_raw, card.model_raw, card.title]
                    .filter(Boolean)
                    .slice(0, 2)
                    .join(" ") || "—"}
                </p>
                <p className="mt-1 text-xs num font-bold" style={{ color: "var(--accent)" }}>
                  {card.price_amount !== null ? formatToman(card.price_amount) : "توافقی"}
                </p>
                <p className="mt-1 text-[11px] num" style={{ color: "var(--muted)" }}>
                  سال {formatNumber(card.year_num)} · {formatNumber(card.mileage_km)} km
                </p>
                <div className="mt-2 flex items-center justify-between">
                  <span className={`chip platform-${card.source_platform}`}>
                    {PLATFORM_LABELS[card.source_platform] ?? card.source_platform}
                  </span>
                  <span className="text-[10px]" style={{ color: "var(--muted)" }}>
                    {card.first_seen_at?.slice(0, 10)}
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

function FreshnessCard({
  label,
  value,
  small = false,
}: {
  label: string;
  value: string;
  small?: boolean;
}) {
  return (
    <div className="card p-3">
      <p className={small ? "text-xs num" : "text-lg font-bold num"}>{value}</p>
      <p className="text-[11px]" style={{ color: "var(--muted)" }}>
        {label}
      </p>
    </div>
  );
}
