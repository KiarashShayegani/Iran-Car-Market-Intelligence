"use client";

import {
  PLATFORM_LABELS,
  SearchResultItem,
  formatNumber,
  formatToman,
} from "@/lib/api";

export default function ResultCard({
  result,
  onOpenDetail,
}: {
  result: SearchResultItem;
  onOpenDetail: (listingKey: string) => void;
}) {
  const platformLabel = PLATFORM_LABELS[result.source_platform] ?? result.source_platform;
  const firstLine = result.excerpt.split("\n")[0];

  return (
    <div className="card p-4 mb-3 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`chip platform-${result.source_platform}`}>{platformLabel}</span>
          <span className="chip score-chip num">مشابهت {result.score.toFixed(4)}</span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            شناسه: <span className="num">{result.listing_key}</span>
          </span>
        </div>
        <button
          onClick={() => onOpenDetail(result.listing_key)}
          className="text-xs underline shrink-0"
          style={{ color: "var(--accent)" }}
        >
          جزئیات
        </button>
      </div>

      <div className="mt-2 flex items-center gap-1.5 flex-wrap">
        {result.brand_raw && <span className="chip btn-secondary">{result.brand_raw}</span>}
        {result.model_raw && <span className="chip btn-secondary">{result.model_raw}</span>}
        {result.year_num !== null && (
          <span className="chip btn-secondary num">سال {formatNumber(result.year_num)}</span>
        )}
        {result.mileage_km !== null && (
          <span className="chip btn-secondary num">
            کارکرد {formatNumber(result.mileage_km)} کیلومتر
          </span>
        )}
        {result.price_amount !== null && (
          <span className="chip score-chip num">{formatToman(result.price_amount)}</span>
        )}
        {result.location_raw && (
          <span className="chip btn-secondary">{result.location_raw}</span>
        )}
      </div>

      <p className="mt-2 font-semibold text-sm">{firstLine}</p>
      {result.excerpt.includes("\n") && (
        <p className="mt-1 text-sm excerpt" style={{ color: "var(--foreground)" }}>
          {result.excerpt.split("\n").slice(1).join("\n").slice(0, 300)}
          {(result.excerpt.split("\n").slice(1).join("\n").length > 300) && "…"}
        </p>
      )}

      <div className="mt-3 flex items-center justify-between gap-2 flex-wrap">
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {result.original_url ? (
            <a href={result.original_url} target="_blank" rel="noopener noreferrer" className="underline">
              مشاهده آگهی اصلی ↗
            </a>
          ) : (
            "بدون لینک منبع"
          )}
        </span>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {result.chunk_count > 1
            ? `قطعه ${result.chunk_index + 1} از ${result.chunk_count}`
            : ""}
        </span>
      </div>
    </div>
  );
}
