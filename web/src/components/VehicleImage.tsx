"use client";

import { useState } from "react";
import { ImageOff } from "lucide-react";
import { API_BASE } from "@/lib/api";

/**
 * VehicleImage — direct CDN rendering first (G1 spike: verified working),
 * one retry through the cache-aside backend proxy on failure, then an
 * honest placeholder. Never a fake photo.
 */
export default function VehicleImage({
  src,
  alt,
  className = "",
}: {
  src: string | null | undefined;
  alt: string;
  className?: string;
}) {
  const [stage, setStage] = useState<"direct" | "proxy" | "failed">(src ? "direct" : "failed");
  const [loaded, setLoaded] = useState(false);

  if (!src || stage === "failed") {
    return (
      <div
        className={`flex items-center justify-center ${className}`}
        style={{ background: "var(--bg-muted)" }}
        aria-label={alt}
      >
        <div className="flex flex-col items-center gap-1" style={{ color: "var(--muted)" }}>
          <ImageOff size={22} strokeWidth={1.5} />
          <span className="text-[11px]">بدون تصویر</span>
        </div>
      </div>
    );
  }

  const url = stage === "direct" ? src : `${API_BASE}/images/proxy?url=${encodeURIComponent(src)}`;

  return (
    <div className={`relative overflow-hidden ${className}`} style={{ background: "var(--bg-muted)" }}>
      {!loaded && (
        <div className="absolute inset-0 animate-pulse" style={{ background: "var(--border)" }} />
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={alt}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        onError={() => {
          if (stage === "direct") setStage("proxy");
          else setStage("failed");
        }}
        className={`h-full w-full object-cover transition-opacity duration-300 ${
          loaded ? "opacity-100" : "opacity-0"
        }`}
      />
    </div>
  );
}
