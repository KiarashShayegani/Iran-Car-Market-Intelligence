"use client";

import { useState } from "react";
import AskPanel from "@/components/AskPanel";
import FilterPanel from "@/components/FilterPanel";
import ListingModal from "@/components/ListingModal";
import ResultCard from "@/components/ResultCard";
import StatsPanel from "@/components/StatsPanel";
import {
  EMPTY_FILTERS,
  FiltersState,
  SearchApiResponse,
  searchAds,
} from "@/lib/api";

type Tab = "search" | "ask" | "stats";

const TABS: { key: Tab; label: string }[] = [
  { key: "search", label: "🔍 جستجو" },
  { key: "ask", label: "💬 پرسش هوشمند" },
  { key: "stats", label: "📊 داشبورد بازار" },
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("search");
  const [filters, setFilters] = useState<FiltersState>(EMPTY_FILTERS);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openListing, setOpenListing] = useState<string | null>(null);

  const runSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await searchAds(query.trim(), filters));
    } catch (e) {
      setError(e instanceof Error ? e.message : "خطای ناشناخته");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <header
        className="px-6 py-4 border-b flex items-center justify-between flex-wrap gap-2"
        style={{ background: "var(--card)", borderColor: "var(--border)" }}
      >
        <div>
          <h1 className="text-lg font-bold">آی‌سی‌ام‌آی — هوش بازار خودرو ایران</h1>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            جستجوی معنایی چندمنبعی · آمار بازار · پرسش‌وپاسخ مبتنی بر شواهد
          </p>
        </div>
        <nav className="flex gap-2 text-sm">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-1.5 rounded-lg font-medium ${
                tab === t.key ? "tab-active" : "btn-secondary"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="max-w-4xl mx-auto p-4">
        {tab === "search" && (
          <>
            <div className="card p-4 mb-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && runSearch()}
                  placeholder="مثلاً: ماشین اتوماتیک کم کارکرد خانوادگی"
                />
                <button className="btn shrink-0" onClick={runSearch} disabled={loading}>
                  {loading ? "…" : "جستجو"}
                </button>
              </div>
            </div>

            <FilterPanel
              filters={filters}
              onChange={setFilters}
              onReset={() => setFilters(EMPTY_FILTERS)}
            />

            {error && (
              <div className="card p-4 text-sm" style={{ color: "#991b1b" }}>
                ⚠️ {error} — مطمئن شوید سرور API روی پورت ۸۰۰۰ اجرا شده است
                (scripts/run_api.py).
              </div>
            )}

            {results && (
              <div className="mb-3 text-xs" style={{ color: "var(--muted)" }}>
                حالت: {results.mode === "hybrid" ? "ترکیبی (فیلتر + معنایی)" : "معنایی"} ·
                کل داده: <span className="num">{results.total_corpus}</span> ·
                پس از فیلتر: <span className="num">{results.candidates_after_filters}</span> ·
                نتایج: <span className="num">{results.results.length}</span>
              </div>
            )}

            {results &&
              results.results.length === 0 && (
                <div className="card p-6 text-sm text-center" style={{ color: "var(--muted)" }}>
                  هیچ نتیجه‌ای یافت نشد — فیلترها سخت‌گیرانه‌اند و شل نمی‌شوند؛
                  پرسش یا فیلترها را تغییر دهید.
                </div>
              )}

            {results?.results.map((r) => (
              <ResultCard
                key={r.listing_key}
                result={r}
                onOpenDetail={setOpenListing}
              />
            ))}
          </>
        )}

        {tab === "ask" && (
          <>
            <FilterPanel
              filters={filters}
              onChange={setFilters}
              onReset={() => setFilters(EMPTY_FILTERS)}
            />
            <AskPanel filters={filters} />
          </>
        )}

        {tab === "stats" && <StatsPanel />}
      </main>

      <footer
        className="text-center text-xs py-4"
        style={{ color: "var(--muted)" }}
      >
        ICMI v3 · داده‌ها از منابع عمومی باما، شیپور و همراه‌مکانیک · پاسخ‌های
        هوشمند فقط بر اساس آگهی‌های بازیابی‌شده ساخته می‌شوند
      </footer>

      {openListing && (
        <ListingModal listingKey={openListing} onClose={() => setOpenListing(null)} />
      )}
    </div>
  );
}
