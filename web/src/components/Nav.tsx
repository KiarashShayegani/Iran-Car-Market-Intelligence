"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, Sparkles, Banknote, BarChart3 } from "lucide-react";

const LINKS = [
  { href: "/", label: "خانه", icon: null },
  { href: "/search", label: "جستجو", icon: Search },
  { href: "/ask", label: "پرسش هوشمند", icon: Sparkles },
  { href: "/estimate", label: "تخمین قیمت", icon: Banknote },
  { href: "/stats", label: "داشبورد", icon: BarChart3 },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex items-center gap-1 flex-wrap text-sm">
      {LINKS.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-medium transition-colors ${
              active
                ? "bg-teal-700 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            {Icon && <Icon size={15} strokeWidth={2} />}
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
