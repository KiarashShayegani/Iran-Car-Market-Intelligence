"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { GitCompare } from "lucide-react";
import {
  SearchResultItem,
  formatNumber,
  formatToman,
  getCompareKeys,
  scoreBand,
  toggleCompareKey,
} from "@/lib/api";
import VehicleImage from "@/components/VehicleImage";
import PlatformMark from "@/components/PlatformMark";

export default function ResultCard({ result }: { result: SearchResultItem }) {
  const [platform, id] = result.listing_key.split(":");
  const band = scoreBand(result.score);
  const [compareKeys, setCompareKeys] = useState<string[]>(getCompareKeys());

  useEffect(() => {
    const update = () => setCompareKeys(getCompareKeys());
    window.addEventListener("icmi-compare-change", update);
    return () => window.removeEventListener("icmi-compare-change", update);
  }, []);

  const inCompare = compareKeys.includes(result.listing_key);

  const title =
    [result.brand_raw, result.model_raw, result.trim_raw]
      .filter(Boolean)
      .join(" ") || result.listing_key;

  return (
    <div className="relative">
      <Link href={`/listings/${platform}/${id}`} className="card card-hover overflow-hidden block">
        <VehicleImage
          src={result.image_url}
          alt={result.model_raw ?? result.listing_key}
          className="aspect-[4/3] w-full"
        />
        <div className="p-3">
          <div className="flex items-center justify-between gap-2">
            <PlatformMark platform={result.source_platform} />
            <span
              className={`chip ${band.className}`}
              title={`شباهت معنایی خام: ${result.score.toFixed(4)} (uncalibrated)`}
            >
              {band.label}
            </span>
          </div>

          <p className="mt-2 text-sm font-bold truncate" title={title}>
            {title}
          </p>

          <div className="mt-2 flex items-center gap-1.5 flex-wrap">
            {result.year_num !== null && (
              <span className="chip btn-secondary num">
                سال {formatNumber(result.year_num)}
              </span>
            )}
            {result.mileage_km !== null && (
              <span className="chip btn-secondary num">
                کارکرد {formatNumber(result.mileage_km)} کیلومتر
              </span>
            )}
            {result.location_raw && (
              <span className="chip btn-secondary">{result.location_raw}</span>
            )}
          </div>

          <p className="mt-2 text-sm num font-bold" style={{ color: "var(--accent)" }}>
            {result.price_amount !== null ? formatToman(result.price_amount) : "توافقی"}
          </p>
        </div>
      </Link>
      <button
        onClick={() => setCompareKeys(toggleCompareKey(result.listing_key))}
        aria-label="افزودن/حذف از مقایسه"
        title="افزودن/حذف از مقایسه"
        className={`absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full ${
          getCompareKeys().includes(result.listing_key) ? "bg-teal-700 text-white" : "bg-white/90 text-slate-600"
        }`}
      >
        <GitCompare size={13} />
      </button>
    </div>
  );
}
