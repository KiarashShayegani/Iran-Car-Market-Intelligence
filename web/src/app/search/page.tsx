"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import FilterPanel from "@/components/FilterPanel";
import ResultCard from "@/components/ResultCard";
import {
  EMPTY_FILTERS,
  FiltersState,
  SearchApiResponse,
  searchAds,
} from "@/lib/api";

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="card p-6 text-sm">در حال بارگذاری…</div>}>
      <SearchContent />
    </Suspense>
  );
}

function SearchContent() {
  const params = useSearchParams();
  const initialQuery = params.get("q") ?? "";

  const [query, setQuery] = useState(initialQuery);
  const [filters, setFilters] = useState<FiltersState>(EMPTY_FILTERS);
  const [sort, setSort] = useState("relevance");
  const [results, setResults] = useState<SearchApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (q: string, f: FiltersState, s: string, offset: number) => {
      if (!q.trim()) return;
      offset === 0 ? setLoading(true) : setLoadingMore(true);
      setError(null);
      try {
        const response = await searchAds(q.trim(), f, 12, offset, s);
        setResults((prev) =>
          offset === 0 || !prev
            ? response
            : {
                ...response,
                results: [...prev.results, ...response.results],
              }
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : "خطای ناشناخته");
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    },
    []
  );

  // auto-run on arrival from the homepage (?q=…)
  useEffect(() => {
    if (initialQuery.trim()) run(initialQuery, EMPTY_FILTERS, "relevance", 0);
  }, [initialQuery, run]);

  const canLoadMore =
    results && results.results.length < results.total_matched
      ? true
      : false;

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <div className="flex gap-2">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run(query, filters, sort, 0)}
            placeholder="جستجوی معنایی به فارسی یا انگلیسی…"
          />
          <select
            value={sort}
            onChange={(e) => {
              setSort(e.target.value);
              run(query, filters, e.target.value, 0);
            }}
            className="!w-36 shrink-0"
            aria-label="ترتیب نتایج"
          >
            <option value="relevance">مرتبط‌ترین</option>
            <option value="price_desc">گران‌ترین</option>
            <option value="price_asc">ارزان‌ترین</option>
            <option value="year_desc">سال نو</option>
          </select>
          <button className="btn shrink-0" onClick={() => run(query, filters, sort, 0)}>
            جستجو
          </button>
        </div>
      </div>

      <FilterPanel filters={filters} onChange={setFilters} onReset={() => setFilters(EMPTY_FILTERS)} />

      {error && (
        <div className="card p-4 text-sm" style={{ color: "#991b1b" }}>
          ⚠️ {error}
        </div>
      )}

      {results && (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          حالت: {results.mode === "hybrid" ? "ترکیبی (فیلتر + معنایی)" : "معنایی"} · کل داده:{" "}
          <span className="num">{results.total_corpus}</span> · پس از فیلتر:{" "}
          <span className="num">{results.candidates_after_filters}</span> · یافت‌شده:{" "}
          <span className="num">{results.total_matched}</span> · نمایش:{" "}
          <span className="num">{results.results.length}</span>
        </p>
      )}

      {results?.results.length === 0 && (
        <div className="card p-8 text-center text-sm" style={{ color: "var(--muted)" }}>
          هیچ نتیجه‌ای یافت نشد — فیلترها سخت‌گیرانه‌اند و شل نمی‌شوند؛ پرسش یا
          فیلترها را تغییر دهید.
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {results?.results.map((r) => (
          <ResultCard key={`${r.listing_key}-${r.chunk_index}`} result={r} />
        ))}
      </div>

      {canLoadMore && (
        <div className="text-center">
          <button
            className="btn-secondary !rounded-lg !px-6 !py-2 !text-sm"
            onClick={() => run(query, filters, sort, results!.results.length)}
            disabled={loadingMore}
          >
            {loadingMore ? "…" : "نمایش بیشتر"}
          </button>
        </div>
      )}
    </div>
  );
}
