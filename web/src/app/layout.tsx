import type { Metadata } from "next";
import Link from "next/link";
import CompareBar from "@/components/CompareBar";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "آی‌سی‌ام‌آی — هوش بازار خودرو ایران",
  description:
    "بازار هوشمند آگهی‌های خودرو: جستجوی معنایی چندمنبعی، آمار بازار، تخمین قیمت و پرسش‌وپاسخ مبتنی بر شواهد",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fa" dir="rtl">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css"
        />
      </head>
      <body className="antialiased">
        <header
          className="sticky top-0 z-40 border-b"
          style={{ background: "rgba(255,255,255,0.92)", backdropFilter: "blur(8px)", borderColor: "var(--border)" }}
        >
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3 flex-wrap">
            <Link href="/" className="flex items-center gap-2">
              <span
                className="flex h-8 w-8 items-center justify-center rounded-lg text-white"
                style={{ background: "var(--accent)" }}
              >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 17h-2v-4l2-5h14l2 5v4h-2" />
                  <circle cx="7.5" cy="17" r="1.5" />
                  <circle cx="16.5" cy="17" r="1.5" />
                  <path d="M9 17h6" />
                </svg>
              </span>
              <div>
                <p className="text-sm font-bold leading-4">آی‌سی‌ام‌آی</p>
                <p className="text-[10px]" style={{ color: "var(--muted)" }}>
                  هوش بازار خودرو ایران
                </p>
              </div>
            </Link>
            <Nav />
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-5">{children}</main>
        <CompareBar />
        <footer
          className="border-t py-5 text-center text-xs"
          style={{ color: "var(--muted)", borderColor: "var(--border)" }}
        >
          ICMI v3 · داده‌ها از منابع عمومی باما، شیپور و همراه‌مکانیک · پاسخ‌های
          هوشمند فقط بر اساس آگهی‌های بازیابی‌شده ساخته می‌شوند · نمونهٔ دادهٔ
          محدود (نه کل بازار)
        </footer>
      </body>
    </html>
  );
}
