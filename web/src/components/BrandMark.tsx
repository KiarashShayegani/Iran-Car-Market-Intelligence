export default function BrandMark({ className = "h-9 w-9" }: { className?: string }) {
  return (
    <svg viewBox="0 0 44 44" className={className} role="img" aria-label="خودروبان">
      <defs>
        <linearGradient id="kbm" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#0B2E2E" />
          <stop offset="0.55" stopColor="#14514F" />
          <stop offset="1" stopColor="#1B6B62" />
        </linearGradient>
      </defs>
      {/* market gauge: open at the bottom, needle pointing up-right (growth) */}
      <path
        d="M 7.9 27.3 A 15 15 0 1 1 36.1 27.3"
        fill="none"
        stroke="url(#kbm)"
        strokeWidth="3.6"
        strokeLinecap="round"
      />
      {/* faint amber ticks flanking the dial */}
      <g stroke="#E6A23C" strokeWidth="1.9" strokeLinecap="round" opacity="0.55">
        <line x1="10.8" y1="22.9" x2="15.2" y2="22.9" />
        <line x1="33.2" y1="22.9" x2="28.8" y2="22.9" />
      </g>
      {/* the needle: بازار هوشمند — rising to the top-right */}
      <line
        x1="22" y1="23.8" x2="29.9" y2="15"
        stroke="#E6A23C" strokeWidth="3.4" strokeLinecap="round"
      />
      {/* hub */}
      <circle cx="22" cy="24.2" r="3.2" fill="#E6A23C" />
      <circle cx="22" cy="24.2" r="1.3" fill="#ffffff" />
    </svg>
  );
}
