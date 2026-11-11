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
  trim_raw?: string | null;
  year_num: number | null;
  year_calendar: string | null;
  price_amount: number | null;
  mileage_km: number | null;
  location_raw: string | null;
  image_url?: string | null;
  fuel_raw?: string | null;
  transmission_raw?: string | null;
  body_type_raw?: string | null;
}

export interface ListingCard {
  listing_key: string;
  source_platform: string;
  original_url: string | null;
  title: string | null;
  brand_raw: string | null;
  model_raw: string | null;
  price_amount: number | null;
  year_num: number | null;
  year_calendar: string | null;
  mileage_km: number | null;
  location_raw: string | null;
  image_url: string | null;
  first_seen_at: string;
}

export interface LatestApiResponse {
  listings: ListingCard[];
}

export interface BrandFacet {
  key: string;
  label: string;
  count: number;
  models: { label: string; count: number }[];
}

export interface FacetsApiResponse {
  brands: BrandFacet[];
  fuel: string[];
  transmission: string[];
  body_type: string[];
  locations: { label: string; count: number }[];
  platforms: { platform: string; count: number }[];
  bounds: {
    min_price: number | null;
    max_price: number | null;
    min_year: number | null;
    max_year: number | null;
  };
  freshness: {
    oldest_collection: string | null;
    newest_collection: string | null;
    total_listings: number;
    priced_listings: number;
  };
}

export interface SearchApiResponse {
  query: string;
  mode: "semantic" | "hybrid" | "browse";
  total_corpus: number;
  candidates_after_filters: number;
  total_matched: number;
  offset: number;
  limit: number;
  sort: string;
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
  by_mileage: Record<string, string | number>[];
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
  extraction_version: string;
  raw_payload_ref_json: string;
  price_history: {
    observed_at: string;
    price_raw: string | null;
    price_amount: number | null;
    price_currency_raw: string | null;
    run_id: string;
  }[];
  image_urls: string[];
}

export interface FiltersState {
  platforms: string[];
  brands: string;      // comma-separated normalized keys
  models: string;      // comma-separated model labels
  price_min: string;
  price_max: string;
  year_min: string;
  year_max: string;
  year_calendar: string;
  mileage_max_km: string;
  location_contains: string;
  fuel: string;        // comma-separated
  transmission: string;
  body_type: string;
}

export const EMPTY_FILTERS: FiltersState = {
  platforms: [],
  brands: "",
  models: "",
  price_min: "",
  price_max: "",
  year_min: "",
  year_max: "",
  year_calendar: "",
  mileage_max_km: "",
  location_contains: "",
  fuel: "",
  transmission: "",
  body_type: "",
};

export function filtersToParams(f: FiltersState): URLSearchParams {
  const p = new URLSearchParams();
  if (f.platforms.length) p.set("platforms", f.platforms.join(","));
  if (f.brands.trim()) p.set("brands", f.brands.trim());
  if (f.models.trim()) p.set("models", f.models.trim());
  if (f.price_min) p.set("price_min", f.price_min);
  if (f.price_max) p.set("price_max", f.price_max);
  if (f.year_min) p.set("year_min", f.year_min);
  if (f.year_max) p.set("year_max", f.year_max);
  if (f.year_calendar) p.set("year_calendar", f.year_calendar);
  if (f.mileage_max_km) p.set("mileage_max_km", f.mileage_max_km);
  if (f.location_contains.trim())
    p.set("location_contains", f.location_contains.trim());
  if (f.fuel.trim()) p.set("fuel", f.fuel.trim());
  if (f.transmission.trim()) p.set("transmission", f.transmission.trim());
  if (f.body_type.trim()) p.set("body_type", f.body_type.trim());
  return p;
}

export function filtersActiveCount(f: FiltersState): number {
  let n = 0;
  if (f.platforms.length) n += f.platforms.length;
  if (f.brands.trim()) n += 1;
  if (f.models.trim()) n += 1;
  if (f.price_min) n += 1;
  if (f.price_max) n += 1;
  if (f.year_min) n += 1;
  if (f.year_max) n += 1;
  if (f.year_calendar) n += 1;
  if (f.mileage_max_km) n += 1;
  if (f.location_contains.trim()) n += 1;
  if (f.fuel.trim()) n += 1;
  if (f.transmission.trim()) n += 1;
  if (f.body_type.trim()) n += 1;
  return n;
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
  topK = 12,
  offset = 0,
  sort = "relevance"
): Promise<SearchApiResponse> {
  const params = filtersToParams(filters);
  params.set("q", query);
  params.set("top_k", String(topK));
  params.set("offset", String(offset));
  params.set("sort", sort);
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

export async function fetchLatest(limit = 8): Promise<LatestApiResponse> {
  const res = await fetch(`${API_BASE}/latest?limit=${limit}`);
  if (!res.ok) throw new Error(`خطا (${res.status})`);
  return res.json();
}

export async function fetchFacets(): Promise<FacetsApiResponse> {
  const res = await fetch(`${API_BASE}/facets`);
  if (!res.ok) throw new Error(`خطا (${res.status})`);
  return res.json();
}

export interface SimilarApiResponse {
  listing_key: string;
  results: SearchResultItem[];
}

export async function fetchSimilar(listingKey: string, limit = 6): Promise<SimilarApiResponse> {
  const [platform, id] = listingKey.split(":");
  const res = await fetch(`${API_BASE}/listings/${platform}/${id}/similar?limit=${limit}`);
  if (!res.ok) throw new Error(`خطا (${res.status})`);
  return res.json();
}

/**
 * Qualitative relevance bands (owner decision: never present raw cosine as
 * a calibrated percentage). Thresholds are corpus-relative, derived from the
 * F3c evaluation distribution (relevant top hits ≥0.84; weak <0.80) — must
 * be recalibrated as the corpus grows.
 */
export function scoreBand(score: number): { label: string; className: string } {
  if (score >= 0.84) return { label: "تطابق بالا", className: "score-high" };
  if (score >= 0.8) return { label: "تطابق متوسط", className: "score-mid" };
  return { label: "تطابق کم", className: "score-low" };
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


export interface CompareVehicle {
  listing: ListingDetail & Record<string, unknown>;
  image_url: string | null;
  market_range: {
    count: number;
    median: number | null;
    p25: number | null;
    p75: number | null;
    quality: string;
    quality_note: string;
  };
}

export interface CompareApiResponse {
  vehicles: CompareVehicle[];
}

export async function fetchCompare(keys: string[]): Promise<CompareApiResponse> {
  const res = await fetch(`${API_BASE}/compare?keys=${keys.join(",")}`);
  if (!res.ok) {
    const detail = (await res.json())?.detail;
    throw new Error(detail || `خطای مقایسه (${res.status})`);
  }
  return res.json();
}

/* --- compare selection (localStorage, max 3) --- */
const COMPARE_KEY = "icmi-compare-keys";

export function getCompareKeys(): string[] {
  try {
    return JSON.parse(localStorage.getItem(COMPARE_KEY) ?? "[]") as string[];
  } catch {
    return [];
  }
}

export function toggleCompareKey(key: string): string[] {
  const current = getCompareKeys();
  const next = current.includes(key)
    ? current.filter((k) => k !== key)
    : [...current, key].slice(-3);
  localStorage.setItem(COMPARE_KEY, JSON.stringify(next));
  window.dispatchEvent(new Event("icmi-compare-change"));
  return next;
}

export function clearCompareKeys(): void {
  localStorage.removeItem(COMPARE_KEY);
  window.dispatchEvent(new Event("icmi-compare-change"));
}
