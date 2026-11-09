"use client";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface SearchResultItem {
  listing_key: string;
  source_platform: string;
  score: number;
  excerpt: string;
  original_url: string | null;
  raw_payload_ref_json: string;
  chunk_index: number;
  chunk_count: number;
  brand_raw: string | null;
  model_raw: string | null;
  year_num: number | null;
  year_calendar: string | null;
  price_amount: number | null;
  mileage_km: number | null;
  location_raw: string | null;
}

export interface SearchApiResponse {
  query: string;
  mode: "semantic" | "hybrid";
  total_corpus: number;
  candidates_after_filters: number;
  results: SearchResultItem[];
}

export interface Citation {
  index: number;
  listing_key: string;
  source_platform: string;
  original_url: string | null;
  score: number;
}

export interface AskApiResponse {
  question: string;
  answer: string;
  grounded: boolean;
  insufficient_evidence: boolean;
  model_id: string | null;
  evidence_count: number;
  candidates_after_filters: number;
  citations: Citation[];
}

export interface StatsApiResponse {
  meta: Record<string, unknown>;
  overview: Record<string, string | number>[];
  by_brand: Record<string, string | number>[];
  by_year: Record<string, string | number>[];
}

export interface ListingDetail {
  listing_key: string;
  source_platform: string;
  source_listing_id: string;
  original_url: string | null;
  title: string | null;
  description: string | null;
  brand_raw: string | null;
  model_raw: string | null;
  trim_raw: string | null;
  year_raw: string | null;
  year_calendar: string | null;
  mileage_raw: string | null;
  price_raw: string | null;
  price_amount: number | null;
  price_currency_raw: string | null;
  location_raw: string | null;
  fuel_raw: string | null;
  transmission_raw: string | null;
  body_type_raw: string | null;
  body_condition_raw: string | null;
  color_raw: string | null;
  first_seen_at: string;
  last_seen_at: string;
  raw_payload_ref_json: string;
}

export interface FiltersState {
  platforms: string[];
  brands: string;
  price_min: string;
  price_max: string;
  year_min: string;
  year_max: string;
  year_calendar: string;
  mileage_max_km: string;
  location_contains: string;
}

export const EMPTY_FILTERS: FiltersState = {
  platforms: [],
  brands: "",
  price_min: "",
  price_max: "",
  year_min: "",
  year_max: "",
  year_calendar: "",
  mileage_max_km: "",
  location_contains: "",
};

export function filtersToParams(f: FiltersState): URLSearchParams {
  const p = new URLSearchParams();
  if (f.platforms.length) p.set("platforms", f.platforms.join(","));
  if (f.brands.trim()) p.set("brands", f.brands.trim());
  if (f.price_min) p.set("price_min", f.price_min);
  if (f.price_max) p.set("price_max", f.price_max);
  if (f.year_min) p.set("year_min", f.year_min);
  if (f.year_max) p.set("year_max", f.year_max);
  if (f.year_calendar) p.set("year_calendar", f.year_calendar);
  if (f.mileage_max_km) p.set("mileage_max_km", f.mileage_max_km);
  if (f.location_contains.trim())
    p.set("location_contains", f.location_contains.trim());
  return p;
}

export function filtersToBody(f: FiltersState): Record<string, unknown> {
  const body: Record<string, unknown> = { platforms: f.platforms };
  if (f.price_min) body.price_min = Number(f.price_min);
  if (f.price_max) body.price_max = Number(f.price_max);
  if (f.year_min) body.year_min = Number(f.year_min);
  if (f.year_max) body.year_max = Number(f.year_max);
  if (f.year_calendar) body.year_calendar = f.year_calendar;
  if (f.mileage_max_km) body.mileage_max_km = Number(f.mileage_max_km);
  return body;
}

export async function searchAds(
  query: string,
  filters: FiltersState,
  topK = 12
): Promise<SearchApiResponse> {
  const params = filtersToParams(filters);
  params.set("q", query);
  params.set("top_k", String(topK));
  const res = await fetch(`${API_BASE}/search?${params.toString()}`);
  if (!res.ok) throw new Error(`خطای جستجو (${res.status})`);
  return res.json();
}

export async function askQuestion(
  question: string,
  filters: FiltersState,
  topK = 6
): Promise<AskApiResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: topK,
      ...filtersToBody(filters),
    }),
  });
  if (res.status === 503) {
    const detail = (await res.json())?.detail;
    throw new Error(detail || "LLM پیکربندی نشده است");
  }
  if (!res.ok) {
    const detail = (await res.json())?.detail;
    throw new Error(detail || `خطای پرسش‌وپاسخ (${res.status})`);
  }
  return res.json();
}

export async function fetchStats(): Promise<StatsApiResponse> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error(`خطای آمار (${res.status})`);
  return res.json();
}

export async function fetchListing(
  listingKey: string
): Promise<ListingDetail> {
  const [platform, id] = listingKey.split(":");
  const res = await fetch(`${API_BASE}/listings/${platform}/${id}`);
  if (!res.ok) throw new Error(`خطای دریافت آگهی (${res.status})`);
  return res.json();
}

export const DEMO_QUERIES = [
  "ماشین اتوماتیک کم کارکرد خانوادگی",
  "پژو 206 مدل ۹۸ به بالا",
  "شاسی بلند وارداتی کم کارکرد",
  "ماشین صفر با گارانتی",
  "خودروی اقتصادی با مصرف کم",
];

export const PLATFORM_LABELS: Record<string, string> = {
  bama: "باما",
  sheypoor: "شیپور",
  hamrah_mechanic: "همراه مکانیک",
};

export const CALENDAR_LABELS: Record<string, string> = {
  shamsi: "شمسی",
  gregorian: "میلادی",
  unknown: "نامشخص",
};

export function formatNumber(n: number | string | null): string {
  if (n === null || n === undefined || n === "") return "—";
  const num = typeof n === "string" ? Number(n.replace(/,/g, "")) : n;
  if (Number.isNaN(num)) return String(n);
  return new Intl.NumberFormat("fa-IR").format(num);
}

export function formatToman(amount: number | null): string {
  if (amount === null || amount === undefined) return "—";
  const b = amount / 1_000_000_000;
  if (b >= 0.1) return `${new Intl.NumberFormat("fa-IR", { maximumFractionDigits: 2 }).format(b)} میلیارد تومان`;
  const m = amount / 1_000_000;
  return `${new Intl.NumberFormat("fa-IR", { maximumFractionDigits: 0 }).format(m)} میلیون تومان`;
}
