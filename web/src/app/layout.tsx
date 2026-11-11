import type { Metadata } from "next";
import Link from "next/link";
import CompareBar from "@/components/CompareBar";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "خودروبان | ICMI — هوش بازار خودرو ایران",
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
          style={{ background: "rgba(255,255,255,0.85)", backdropFilter: "blur(10px)", borderColor: "var(--border)" }}
        >
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3 flex-wrap">
            <Link href="/" className="flex items-center gap-2.5">
              <img src="/khodroban-mark.svg" alt="خودروبان" className="h-9 w-9" />
              <div>
                <p className="text-base font-extrabold leading-5 tracking-tight">خودروبان</p>
                <p className="text-[10px] font-medium tracking-widest" style={{ color: "var(--accent-text)" }}>
                  ICMI
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
