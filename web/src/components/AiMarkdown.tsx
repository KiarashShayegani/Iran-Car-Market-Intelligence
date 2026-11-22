"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ExternalLink } from "lucide-react";
import VehicleImage from "@/components/VehicleImage";
import {
  Citation,
  formatNumber,
  formatToman,
  formatYearText,
} from "@/lib/api";

const CITATION_HREF = /^#cite-(\d+)$/;

/** Turn bare [n] reference markers in AI text into interactive markdown
    links — rendered as numbered chips by the `a` override below. Only
    markers with a matching citation are converted; `(?!\()` keeps markdown
    links like [1](http…) untouched. */
function linkifyCitations(content: string, total: number): string {
  if (total <= 0) return content;
  // Persian digits → western so [۱] also matches
  const western = content.replace(
    /[۰-۹]/g,
    (d) => String("۰۱۲۳۴۵۶۷۸۹".indexOf(d))
  );
  return western.replace(
    /\[(\d{1,2})\](?!\()/g,
    (m, n: string) =>
      Number(n) >= 1 && Number(n) <= total ? `[${n}](#cite-${n})` : m
  );
}

/** Styled markdown renderer for AI-generated content — headings, tables,
    lists and key figures get a designed, RTL-aware presentation. When
    citations are supplied, inline [n] markers become hoverable numbered
    chips with an ad preview card; clicking one highlights the matching
    result row. */
export default function AiMarkdown({
  content,
  citations,
  onCitationClick,
}: {
  content: string;
  citations?: Citation[];
  onCitationClick?: (n: number) => void;
}) {
  const byIndex = new Map((citations ?? []).map((c) => [c.index, c]));

  return (
    <div className="ai-markdown text-sm prose-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="text-base font-extrabold mt-4 mb-2 pb-1 border-b" style={{ borderColor: "var(--border)" }} {...p} />,
          h2: (p) => <h2 className="text-sm font-extrabold mt-3 mb-1.5" {...p} />,
          h3: (p) => <h3 className="text-sm font-bold mt-2.5 mb-1" {...p} />,
          p: (p) => <p className="my-1.5 leading-6" {...p} />,
          strong: (p) => <strong className="font-extrabold" style={{ color: "var(--secondary)" }} {...p} />,
          ul: (p) => <ul className="my-1.5 space-y-1 list-disc pr-5" {...p} />,
          ol: (p) => <ol className="my-1.5 space-y-1 list-decimal pr-5" {...p} />,
          li: (p) => <li className="leading-6" {...p} />,
          table: (p) => (
            <div className="overflow-x-auto my-2">
              <table className="w-full text-xs border-collapse rounded-lg overflow-hidden" {...p} />
            </div>
          ),
          th: (p) => (
            <th
              className="border px-2.5 py-1.5 text-right font-bold"
              style={{ background: "var(--teal-soft)", borderColor: "var(--border)" }}
              {...p}
            />
          ),
          td: (p) => (
            <td className="border px-2.5 py-1.5" style={{ borderColor: "var(--border)" }} {...p} />
          ),
          blockquote: (p) => (
            <blockquote
              className="my-2 pr-3 border-r-2 text-xs"
              style={{ borderColor: "var(--secondary)", color: "var(--muted)" }}
              {...p}
            />
          ),
          code: (p) => (
            <code className="rounded px-1 py-0.5 text-[11px]" style={{ background: "var(--bg-muted)" }} {...p} />
          ),
          a: ({ href, children }) => {
            const match = typeof href === "string" ? CITATION_HREF.exec(href) : null;
            if (match && onCitationClick) {
              const n = Number(match[1]);
              const citation = byIndex.get(n);
              return <CitationChip n={n} citation={citation} onClick={() => onCitationClick(n)} />;
            }
            return (
              <a
                href={href}
                target={typeof href === "string" && href.startsWith("http") ? "_blank" : undefined}
                rel="noopener noreferrer"
                className="font-semibold underline"
                style={{ color: "var(--secondary)" }}
              >
                {children}
              </a>
            );
          },
        }}
      >
        {linkifyCitations(content, citations?.length ?? 0)}
      </ReactMarkdown>
    </div>
  );
}

/** Numbered inline reference chip: hover shows a compact ad preview card,
    click scrolls to + highlights the matching result row below the answer. */
function CitationChip({
  n,
  citation,
  onClick,
}: {
  n: number;
  citation?: Citation;
  onClick: () => void;
}) {
  const title = citation
    ? [citation.brand_raw, citation.model_raw].filter(Boolean).join(" ")
    : null;
  return (
    <span className="relative inline-block group align-baseline" onClick={onClick}>
      <button
        type="button"
        className="inline-flex items-center justify-center rounded-md align-super mx-0.5 num transition-transform group-hover:scale-110"
        style={{
          minWidth: 18,
          height: 18,
          fontSize: 10,
          padding: "0 3px",
          background: "var(--secondary)",
          color: "#fff",
          boxShadow: "0 1px 2px rgba(15,23,42,0.25)",
        }}
        title={citation ? title ?? "" : `ارجاع ${n}`}
        aria-label={`مشاهدهٔ آگهی ارجاع ${n}`}
      >
        {n.toLocaleString("fa-IR")}
      </button>
      {citation && (
        <span
          className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 w-60 rounded-xl border p-2 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity"
          style={{
            background: "#fff",
            borderColor: "var(--border)",
            display: "block",
          }}
        >
          <span className="flex items-start gap-2 text-right" style={{ display: "flex" }}>
            <VehicleImage
              src={citation.image_url}
              alt={title ?? ""}
              className="w-14 h-11 rounded-lg object-cover shrink-0"
            />
            <span className="min-w-0 flex-1 block">
              <span className="block text-[11px] font-bold truncate">{title}</span>
              <span className="block text-[10px] mt-0.5" style={{ color: "var(--muted)" }}>
                {[
                  citation.year_num !== null && citation.year_num !== undefined
                    ? `سال ${formatYearText(citation.year_num, citation.year_calendar ?? null, null)}`
                    : null,
                  citation.mileage_km !== null && citation.mileage_km !== undefined
                    ? `کارکرد ${formatNumber(citation.mileage_km)}`
                    : null,
                  citation.location_raw,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
              <span className="block text-[11px] font-bold mt-0.5" style={{ color: "var(--accent)" }}>
                {citation.price_amount !== null && citation.price_amount !== undefined
                  ? formatToman(citation.price_amount)
                  : "توافقی"}
              </span>
            </span>
          </span>
          <span
            className="mt-1.5 pt-1.5 border-t flex items-center justify-between text-[10px]"
            style={{ borderColor: "var(--border)", color: "var(--muted)", display: "flex" }}
          >
            <span>برای هایلایت، کلیک کنید</span>
            {citation.original_url && (
              <a
                href={citation.original_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-0.5"
                style={{ color: "var(--secondary)" }}
                onClick={(e) => e.stopPropagation()}
              >
                منبع <ExternalLink size={9} />
              </a>
            )}
          </span>
        </span>
      )}
    </span>
  );
}
