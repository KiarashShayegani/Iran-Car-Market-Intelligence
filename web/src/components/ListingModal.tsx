"use client";

import { useEffect, useState } from "react";
import {
  CALENDAR_LABELS,
  ListingDetail,
  PLATFORM_LABELS,
  fetchListing,
  formatNumber,
  formatToman,
} from "@/lib/api";

export default function ListingModal({
  listingKey,
  onClose,
}: {
  listingKey: string;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<ListingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchListing(listingKey)
      .then(setListing)
      .catch((e) => setError(e instanceof Error ? e.message : "خطای ناشناخته"));
  }, [listingKey]);

  const [platform, id] = listingKey.split(":");

  return (
    <div
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{ background: "rgba(15,23,42,0.45)" }}
      onClick={onClose}
    >
      <div
        className="card p-5 max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {error && <p style={{ color: "#991b1b" }}>{error}</p>}
        {!listing && !error && <p className="text-sm">در حال دریافت…</p>}
        {listing && (
          <>
            <div className="flex items-center justify-between gap-2 mb-3">
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
              <button onClick={onClose} className="btn-secondary text-xs">
                بستن ✕
              </button>
            </div>

            <h2 className="font-bold text-base mb-1">{listing.title ?? "—"}</h2>
            <p className="text-xs mb-4 num" style={{ color: "var(--muted)" }}>
              شناسه مرجع: {listing.listing_key} · اولین مشاهده: {listing.first_seen_at}
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm">
              <Field label="برند" value={listing.brand_raw} />
              <Field label="مدل" value={listing.model_raw} />
              <Field label="تیپ" value={listing.trim_raw} />
              <Field label="سال (خام)" value={formatNumber(listing.year_raw)} />
              <Field label="کارکرد (خام)" value={formatNumber(listing.mileage_raw)} />
              <Field label="قیمت (خام)" value={formatNumber(listing.price_raw)} />
              <Field label="قیمت نرمال‌شده" value={formatToman(listing.price_amount)} />
              <Field label="رنگ" value={listing.color_raw} />
              <Field label="سوخت" value={listing.fuel_raw} />
              <Field label="گیربکس" value={listing.transmission_raw} />
              <Field label="نوع بدنه" value={listing.body_type_raw} />
              <Field label="وضعیت بدنه" value={listing.body_condition_raw} />
              <Field label="موقعیت" value={listing.location_raw} />
            </div>

            {listing.description && (
              <div className="mt-4">
                <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>
                  متن آگهی (عیناً از منبع):
                </p>
                <div className="card p-3 text-sm excerpt">{listing.description}</div>
              </div>
            )}

            <div className="mt-4 flex items-center justify-between gap-2 flex-wrap">
              {listing.original_url && (
                <a
                  href={listing.original_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline text-sm"
                  style={{ color: "var(--accent)" }}
                >
                  مشاهده آگهی اصلی ↗
                </a>
              )}
              <span className="text-xs num" style={{ color: "var(--muted)" }}>
                ارجاع داده خام: {listing.raw_payload_ref_json.slice(0, 80)}
              </span>
            </div>
            <p className="mt-2 text-xs" style={{ color: "var(--muted)" }}>
              شناسه منبع: {platform}:{id}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="p-2 rounded" style={{ background: "var(--background)" }}>
      <p className="text-xs" style={{ color: "var(--muted)" }}>
        {label}
      </p>
      <p className="num">{value ?? "—"}</p>
    </div>
  );
}
