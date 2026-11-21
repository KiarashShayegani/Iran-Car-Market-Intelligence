"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Styled markdown renderer for AI-generated content — headings, tables,
    lists and key figures get a designed, RTL-aware presentation. */
export default function AiMarkdown({ content }: { content: string }) {
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
          a: (p) => (
            <a className="font-semibold underline" style={{ color: "var(--secondary)" }} {...p} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
