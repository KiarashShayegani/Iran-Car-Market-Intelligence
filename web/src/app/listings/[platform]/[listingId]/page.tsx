"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import VehicleImage from "@/components/VehicleImage";
import {
  CALENDAR_LABELS,
  PLATFORM_LABELS,
  fetchListing,
  fetchSimilar,
  formatNumber,
  formatToman,
  ListingDetail,
  SimilarApiResponse,
} from "@/lib/api";

export default function ListingDetailPage() {
  const params = useParams<{ platform: string; listingId: string }>();
  const [listing, setListing] = useState<ListingDetail | null>(null);
  const [similar, setSimilar] = useState<SimilarApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeImage, setActiveImage] = useState(0);

  useEffect(() => {
    fetchListing(`${params.platform}:${params.listingId}`)
      .then(setListing)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
    fetchSimilar(`${params.platform}:${params.listingId}`)
      .then(setSimilar)
      .catch(() => setSimilar(null));
  }, [params.platform, params.listingId]);

  if (error)
    return (
      <div className="card p-6 text-sm" style={{ color: "#991b1b" }}>
        ⚠️ {error}
      </div>
    );
  if (!listing) return <div className="card p-6 text-sm">در حال دریافت…</div>;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className={`chip platform-${listing.source_platform}`}>
            {PLATFORM_LABELS[listing.source_platform] ?? listing.source_platform}
          </span>
          {listing.year_calendar && (
            <span className="chip" style={{ background: "#e0e7ff", color: "#4338ca" }}>
              سال {CALENDAR_LABELS[listing.year_calendar]}
            </span>
          )}
        </div>
        {listing.original_url && (
          <a
            href={listing.original_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-sm underline"
            style={{ color: "var(--accent)" }}
          >
            آگهی اصلی <ExternalLink size={14} />
          </a>
        )}
      </div>

      {/* gallery */}
      <div className="card overflow-hidden">
        <VehicleImage
          src={listing.image_urls[activeImage]}
          alt={listing.title ?? listing.listing_key}
          className="aspect-[16/9] w-full"
        />
        {listing.image_urls.length > 1 && (
          <div className="flex gap-2 p-2 overflow-x-auto">
            {listing.image_urls.map((u, i) => (
              <button key={i} onClick={() => setActiveImage(i)} className="shrink-0">
                <VehicleImage
                  src={u}
                  alt={`${i}`}
                  className={`h-14 w-20 rounded border-2 ${
                    i === activeImage ? "" : "opacity-60"
                  }`}
                />
                <span className="sr-only">{i}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <h1 className="text-lg font-bold">{listing.title ?? "—"}</h1>

      {/* quick facts */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
        <Fact label="قیمت" value={formatToman(listing.price_amount)} highlight />
        <Fact label="سال (خام)" value={formatNumber(listing.year_raw)} />
        <Fact label="کارکرد" value={listing.mileage_raw ? `${formatNumber(listing.mileage_raw)}` : "—"} />
        <Fact label="موقعیت" value={listing.location_raw} />
        <Fact label="برند" value={listing.brand_raw} />
        <Fact label="مدل" value={listing.model_raw} />
        <Fact label="تیپ" value={listing.trim_raw} />
        <Fact label="رنگ" value={listing.color_raw} />
        <Fact label="سوخت" value={listing.fuel_raw} />
        <Fact label="گیربکس" value={listing.transmission_raw} />
        <Fact label="نوع بدنه" value={listing.body_type_raw} />
        <Fact label="وضعیت بدنه" value={listing.body_condition_raw} />
      </div>

      {listing.description && (
        <div>
          <h2 className="mb-1 text-sm font-semibold">متن آگهی (عیناً از منبع)</h2>
          <div className="card p-4 text-sm excerpt">{listing.description}</div>
        </div>
      )}

      {/* price history */}
      <div>
        <h2 className="mb-1 text-sm font-semibold">تاریخچه قیمت</h2>
        <div className="card p-4 text-sm space-y-1">
          {listing.price_history.length === 0 && (
            <p style={{ color: "var(--muted)" }}>مشاهده‌ای ثبت نشده است.</p>
          )}
          {listing.price_history.map((o, i) => (
            <p key={i} className="num flex items-center gap-2">
              <span style={{ color: "var(--muted)" }}>{o.observed_at.slice(0, 16)}</span>
              <span className="font-medium">
                {o.price_amount !== null ? formatToman(o.price_amount) : "بدون قیمت اعلامی"}
              </span>
              <span className="text-xs" style={{ color: "var(--muted)" }}>
                (اجرا: {o.run_id.slice(0, 15)}…)
              </span>
            </p>
          ))}
          {listing.price_history.length === 1 && (
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              تنها یک مشاهده ثبت شده — با اجراهای بعدی، تغییرات قیمت اینجا دیده
              می‌شود (هیچ تاریخی ساختگی اضافه نمی‌شود).
            </p>
          )}
        </div>
      </div>

      {/* similar ads */}
      {similar && similar.results.length > 0 && (
        <div>
          <h2 className="mb-2 text-sm font-semibold">آگهی‌های مشابه (بازیابی معنایی)</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {similar.results.slice(0, 6).map((r) => {
              const [p, id] = r.listing_key.split(":");
              return (
                <Link
                  key={r.listing_key}
                  href={`/listings/${p}/${id}`}
                  className="card card-hover overflow-hidden block"
                >
                  <VehicleImage
                    src={r.image_url ?? null}
                    alt={r.model_raw ?? r.listing_key}
                    className="aspect-[4/3] w-full"
                  />
                  <div className="p-2">
                    <p className="text-xs font-medium truncate">
                      {r.model_raw ?? r.listing_key}
                    </p>
                    <p className="text-xs num" style={{ color: "var(--accent)" }}>
                      {r.price_amount !== null ? formatToman(r.price_amount) : "توافقی"}
                    </p>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      {/* provenance */}
      <div className="card p-4 text-xs space-y-1" style={{ color: "var(--muted)" }}>
        <p>ردیابی داده: شناسه {listing.listing_key} · نسخهٔ استخراج {listing.extraction_version}</p>
        <p>
          ارجاع داده خام: <span className="num">{listing.raw_payload_ref_json.slice(0, 120)}</span>
        </p>
        <p>
          اولین مشاهده: {listing.first_seen_at.slice(0, 16)} · آخرین:{" "}
          {listing.last_seen_at.slice(0, 16)}
        </p>
      </div>
    </div>
  );
}

function Fact({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string | null;
  highlight?: boolean;
}) {
  return (
    <div className="card p-2.5">
      <p className="text-[11px]" style={{ color: "var(--muted)" }}>
        {label}
      </p>
      <p className={`num text-sm ${highlight ? "font-bold" : ""}`}
        style={highlight ? { color: "var(--accent)" } : {}}>
        {value ?? "—"}
      </p>
    </div>
  );
}
