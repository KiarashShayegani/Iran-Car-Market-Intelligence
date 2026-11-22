"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Search as SearchIcon } from "lucide-react";
import DiscoveryResults from "@/components/DiscoveryResults";
import {
  DEMO_QUERIES,
  fetchFacets,
  formatDateJalali,
  formatNumber,
} from "@/lib/api";

export default function Home() {
  return (
    <Suspense fallback={<div className="card p-6 text-sm">در حال بارگذاری…</div>}>
      <HomeContent />
    </Suspense>
  );
}

function HomeContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeQuery = searchParams.get("q") ?? "";
  const [query, setQuery] = useState(activeQuery);
  const [facets, setFacets] = useState<Awaited<ReturnType<typeof fetchFacets>> | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchFacets()
      .then(setFacets)
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
          جستجوی هوشمند آگهی‌ها از باما، دیوار، شیپور و همراه‌مکانیک — با فیلترهای
          دقیق و ارجاع به آگهی اصلی
        </p>
        <div className="mx-auto mt-4 flex max-w-xl gap-2">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit(query)}
            placeholder="مثلاً: ماشین اتوماتیک کم کارکرد خانوادگی"
            style={{ color: "var(--foreground)", background: "#fff" }}
          />
          <button className="btn shrink-0 flex items-center gap-1.5" onClick={() => submit(query)}>
            <SearchIcon size={16} />
            جستجو
          </button>
        </div>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {DEMO_QUERIES.slice(0, 5).map((dq) => (
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
            value={formatDateJalali(facets.freshness.newest_collection)}
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

      {/* results — browse newest or semantic search, with filters + load-more */}
      <Suspense fallback={<div className="card p-6 text-sm">در حال بارگذاری…</div>}>
        <DiscoveryResults />
      </Suspense>
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
