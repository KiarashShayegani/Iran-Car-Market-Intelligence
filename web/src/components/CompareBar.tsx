"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { GitCompare, X } from "lucide-react";
import { clearCompareKeys, getCompareKeys, toggleCompareKey } from "@/lib/api";

/** Floating compare-selection bar (localStorage-backed, max 3 vehicles). */
export default function CompareBar() {
  const [keys, setKeys] = useState<string[]>([]);

  useEffect(() => {
    const update = () => setKeys(getCompareKeys());
    update();
    window.addEventListener("icmi-compare-change", update);
    return () => window.removeEventListener("icmi-compare-change", update);
  }, []);

  if (keys.length === 0) return null;

  return (
    <div
      className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 flex items-center gap-3 rounded-xl px-4 py-2.5 shadow-lg"
      style={{ background: "var(--foreground)", color: "#fff" }}
    >
      <GitCompare size={16} />
      <div className="flex items-center gap-1.5 flex-wrap max-w-md">
        {keys.map((k) => (
          <span key={k} className="chip num" style={{ background: "rgba(255,255,255,0.15)" }}>
            {k}
            <button
              onClick={() => setKeys(toggleCompareKey(k))}
              aria-label={`حذف ${k}`}
              className="mr-1"
            >
              <X size={12} />
            </button>
          </span>
        ))}
      </div>
      <Link
        href={`/compare?keys=${keys.join(",")}`}
        className="rounded-lg px-3 py-1.5 text-sm font-semibold"
        style={{ background: "var(--accent)" }}
      >
        مقایسه ({keys.length})
      </Link>
      <button onClick={() => clearCompareKeys()} aria-label="پاک‌کردن انتخاب">
        <X size={14} />
      </button>
    </div>
  );
}
