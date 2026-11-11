"use client";

/** Consistent source icon + readable name (no external logo hotlinking).
    Local letter-marks in platform-tinted circles — accessible via label. */
const PLATFORMS: Record<string, { label: string; letter: string; bg: string; fg: string }> = {
  bama: { label: "باما", letter: "ب", bg: "#fdf3e3", fg: "#8a5a10" },
  sheypoor: { label: "شیپور", letter: "ش", bg: "#e3f1f0", fg: "#123c46" },
  hamrah_mechanic: { label: "همراه مکانیک", letter: "هـ", bg: "#e8f0e6", fg: "#42632e" },
};

export function PlatformMark({ platform, withLabel = true }: { platform: string; withLabel?: boolean }) {
  const spec = PLATFORMS[platform] ?? { label: platform, letter: platform.slice(0, 1), bg: "#eef1f0", fg: "#5e6e76" };
  return (
    <span className="chip" style={{ background: spec.bg, color: spec.fg }}>
      <span
        aria-hidden
        className="flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold"
        style={{ background: spec.fg, color: spec.bg }}
      >
        {spec.letter}
      </span>
      {withLabel && spec.label}
    </span>
  );
}

export default PlatformMark;
