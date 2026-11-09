import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "آی‌سی‌ام‌آی — هوش بازار خودرو ایران",
  description:
    "جستجوی معنایی آگهی‌های خودرو از چند منبع، آمار بازار، و پرسش‌وپاسخ مبتنی بر شواهد",
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
      <body className="antialiased">{children}</body>
    </html>
  );
}
