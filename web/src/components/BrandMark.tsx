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
      <path
        d="M 33.8 10.2 A 15.5 15.5 0 1 0 38 22"
        fill="none"
        stroke="url(#kbm)"
        strokeWidth="3.4"
        strokeLinecap="round"
      />
      <g stroke="#14514F" strokeWidth="2.6" strokeLinecap="round">
        <line x1="22" y1="22" x2="22" y2="10.5" />
        <line x1="22" y1="22" x2="12" y2="29" />
        <line x1="22" y1="22" x2="32" y2="29" />
      </g>
      <line x1="22" y1="22" x2="35.5" y2="8.5" stroke="#1B6B62" strokeWidth="3" strokeLinecap="round" />
      <path d="M 35.5 8.5 L 31.2 10.4 L 34.6 13.2 Z" fill="#1B6B62" />
      <circle cx="22" cy="22" r="3.1" fill="#E6A23C" />
      <circle cx="22" cy="22" r="1.3" fill="#ffffff" />
      <circle cx="36.8" cy="7.2" r="2.2" fill="#E6A23C" />
    </svg>
  );
}
