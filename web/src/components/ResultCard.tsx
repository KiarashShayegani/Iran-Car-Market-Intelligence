"use client";

import Link from "next/link";
import { PLATFORM_LABELS, SearchResultItem, formatNumber, formatToman, scoreBand } from "@/lib/api";
import VehicleImage from "@/components/VehicleImage";

export default function ResultCard({ result }: { result: SearchResultItem }) {
  const [platform, id] = result.listing_key.split(":");
  const band = scoreBand(result.score);

  return (
    <Link href={`/listings/${platform}/${id}`} className="card card-hover overflow-hidden block">
      <VehicleImage
        src={result.image_url}
        alt={result.model_raw ?? result.listing_key}
        className="aspect-[4/3] w-full"
      />
      <div className="p-3">
        <div className="flex items-center justify-between gap-2">
          <span className={`chip platform-${result.source_platform}`}>
            {PLATFORM_LABELS[result.source_platform] ?? result.source_platform}
          </span>
          <span
            className={`chip ${band.className}`}
            title={`شباهت معنایی خام: ${result.score.toFixed(4)} (uncalibrated)`}
          >
            {band.label}
          </span>
        </div>

        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
          {result.brand_raw && <span className="chip btn-secondary">{result.brand_raw}</span>}
          {result.model_raw && <span className="chip btn-secondary">{result.model_raw}</span>}
          {result.year_num !== null && (
            <span className="chip btn-secondary num">سال {formatNumber(result.year_num)}</span>
          )}
        </div>

        <p className="mt-2 text-sm num font-bold" style={{ color: "var(--accent)" }}>
          {result.price_amount !== null ? formatToman(result.price_amount) : "توافقی"}
        </p>

        <p className="mt-1 text-[11px] num" style={{ color: "var(--muted)" }}>
          {result.mileage_km !== null
            ? `کارکرد ${formatNumber(result.mileage_km)} کیلومتر`
            : "کارکرد نامشخص"}
          {result.location_raw ? ` · ${result.location_raw}` : ""}
        </p>
      </div>
    </Link>
  );
}
