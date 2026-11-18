"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import FilterPanel from "@/components/FilterPanel";
import ResultCard from "@/components/ResultCard";
import {
  EMPTY_FILTERS,
  FiltersState,
  SearchApiResponse,
  filtersActiveCount,
  searchAds,
} from "@/lib/api";

export default function DiscoveryResults() {
  const params = useSearchParams();
  const initialQuery = params.get("q") ?? "";

  const [filters, setFilters] = useState<FiltersState>(EMPTY_FILTERS);
  const [sort, setSort] = useState(initialQuery.trim() ? "relevance" : "newest");
  const [results, setResults] = useState<SearchApiResponse | null>(null);
  // skeletons from first paint when arriving with ?q= (auto-run effect
  // fires after hydration — this avoids a brief blank flash)
  const [loading, setLoading] = useState(Boolean(initialQuery));
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (q: string, f: FiltersState, s: string, offset: number) => {
      // empty query + no filters = browse mode over the newest ads
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

  // auto-run on arrival and on every query change; an empty query is a
  // browse-mode pass over the newest ads (Bama-like scrolling discovery)
  useEffect(() => {
    run(initialQuery, EMPTY_FILTERS, initialQuery.trim() ? "relevance" : "newest", 0);
  }, [initialQuery, run]);

  const canLoadMore =
    results && results.results.length < results.total_matched
      ? true
      : false;

  return (
    <div className="space-y-4">
      <FilterPanel filters={filters} onChange={setFilters} onReset={() => setFilters(EMPTY_FILTERS)} />

      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold" style={{ color: "var(--muted)" }}>
          {initialQuery.trim()
            ? `نتایج جستجو برای «${initialQuery}»`
            : "تازه‌ترین آگهی‌های جمع‌آوری‌شده"}
        </p>
        <select
          value={sort}
          onChange={(e) => {
            setSort(e.target.value);
            run(initialQuery, filters, e.target.value, 0);
          }}
          className="!w-40 !py-1.5 !text-xs"
          aria-label="ترتیب نتایج"
        >
          <option value="newest">جدیدترین</option>
          <option value="relevance">مرتبط‌ترین</option>
          <option value="price_desc">گران‌ترین</option>
          <option value="price_asc">ارزان‌ترین</option>
          <option value="year_desc">سال نو</option>
        </select>
      </div>

      {error && (
        <div className="card p-4 text-sm" style={{ color: "#991b1b" }}>
          ⚠️ {error}
        </div>
      )}

      {results && (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          حالت:{" "}
          {results.mode === "hybrid"
            ? "ترکیبی (فیلتر + معنایی)"
            : results.mode === "browse"
              ? "مرور با فیلتر"
              : "معنایی"}{" "}
          · کل داده:{" "}
          <span className="num">{results.total_corpus}</span> · پس از فیلتر:{" "}
          <span className="num">{results.candidates_after_filters}</span> · یافت‌شده:{" "}
          <span className="num">{results.total_matched}</span> · نمایش:{" "}
          <span className="num">{results.results.length}</span>
        </p>
      )}

      {loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card overflow-hidden">
              <div className="skeleton aspect-[4/3] w-full" />
              <div className="p-3 space-y-2">
                <div className="skeleton h-4 w-2/3" />
                <div className="skeleton h-4 w-1/3" />
                <div className="skeleton h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && results?.results.length === 0 && (
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
            onClick={() => run(initialQuery, filters, sort, results!.results.length)}
            disabled={loadingMore}
          >
            {loadingMore ? "…" : "نمایش بیشتر"}
          </button>
        </div>
      )}
    </div>
  );
}
