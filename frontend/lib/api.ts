/* Typed client for the VisuRAG FastAPI backend.
 *
 * Mirrors `api/schemas.py`. The backend returns evidence patch paths as
 * `data/...` repo-relative paths (e.g. `data/cache/patches/<doc>/….png`);
 * `evidenceUrl` maps them onto `GET /files/...`.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_VISURAG_API_URL ?? "http://localhost:8000";

export interface Health {
  status: string;
  qdrant_mode: string;
  vlm_provider: string;
  vlm_model: string;
  counts: Record<string, number>;
}

export interface IngestedDoc {
  document_id: string;
  source: string;
  pages: number;
  text_points: number;
  visual_points: number;
  cache_hit: boolean;
}

export interface Citation {
  source: string | null;
  page_num: number | null;
  chunk_id: string | null;
  text_snippet: string | null;
  image_patch_paths: string[];
  fused_score: number | null;
  rerank_score: number | null;
}

export interface QueryAnswer {
  answer: string;
  model: string;
  provider: string;
  source: string | null;
  page_num: number | null;
  image_patch_paths: string[];
  citations: Citation[];
  prompt_chars: number;
  images_sent: number;
}

/** Map a backend ``data/...`` patch path to a servable ``/files/...`` URL. */
export function evidenceUrl(apiUrl: string, imagePath: string): string {
  let sub = imagePath;
  if (sub.startsWith("data/")) sub = sub.slice("data/".length);
  sub = sub.replace(/^\/+/, "");
  return `${apiUrl.replace(/\/+$/, "")}/files/${sub}`;
}

async function checked(res: Response, what: string): Promise<unknown> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(`${what} failed (${res.status}: ${detail})`);
  }
  return res.json() as Promise<unknown>;
}

export async function fetchHealth(apiUrl: string = API_URL): Promise<Health> {
  const res = await fetch(`${apiUrl}/health`);
  return (await checked(res, "health check")) as Health;
}

export async function ingestPdf(
  file: File,
  apiUrl: string = API_URL,
): Promise<IngestedDoc> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiUrl}/ingest`, { method: "POST", body: form });
  return (await checked(res, "ingest")) as IngestedDoc;
}

export async function queryRag(
  query: string,
  opts: { document_id?: string | null; top_k?: number; max_images?: number } = {},
  apiUrl: string = API_URL,
): Promise<QueryAnswer> {
  const res = await fetch(`${apiUrl}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      document_id: opts.document_id ?? null,
      top_k: opts.top_k ?? 5,
      max_images: opts.max_images ?? 4,
    }),
  });
  return (await checked(res, "query")) as QueryAnswer;
}
