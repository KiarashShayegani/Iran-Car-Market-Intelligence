import { redirect } from "next/navigation";

/** /search merged into the Home discovery experience — route compatibility:
    existing /search links land on Home with the query preserved. */
export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { q } = await searchParams;
  redirect(q ? `/?q=${encodeURIComponent(q)}` : "/");
}
