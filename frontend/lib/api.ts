/* Typed client for the VisuRAG FastAPI backend.
 *
 * Mirrors `api/schemas.py`. The backend returns evidence patch paths as
 * `data/...` repo-relative paths (e.g. `data/cache/patches/<doc>/….png`);
 * `evidenceUrl` maps them onto `GET /files/...`.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_VISURAG_API_URL ?? "http://localhost:8000";

/** Optional backend key (`VISURAG_API_KEY`). Only set this for trusted
 * self-hosted deployments — NEXT_PUBLIC_ values ship to the browser. */
export const API_KEY = process.env.NEXT_PUBLIC_VISURAG_API_KEY ?? "";

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return API_KEY ? { ...extra, "X-API-Key": API_KEY } : extra;
}

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
  const base = `${apiUrl.replace(/\/+$/, "")}/files/${sub}`;
  // <img> tags cannot send headers, so the key travels as a query param.
  return API_KEY ? `${base}?api_key=${encodeURIComponent(API_KEY)}` : base;
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
  const res = await fetch(`${apiUrl}/health`, { headers: authHeaders() });
  return (await checked(res, "health check")) as Health;
}

export async function ingestPdf(
  file: File,
  apiUrl: string = API_URL,
): Promise<IngestedDoc> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiUrl}/ingest`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return (await checked(res, "ingest")) as IngestedDoc;
}

export async function queryRag(  query: string,
  opts: { document_id?: string | null; top_k?: number; max_images?: number } = {},
  apiUrl: string = API_URL,
): Promise<QueryAnswer> {
  const res = await fetch(`${apiUrl}/query`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      query,
      document_id: opts.document_id ?? null,
      top_k: opts.top_k ?? 5,
      max_images: opts.max_images ?? 4,
    }),
  });
  return (await checked(res, "query")) as QueryAnswer;
}

export async function deleteDocument(
  document_id: string,
  apiUrl: string = API_URL,
): Promise<void> {
  const res = await fetch(
    `${apiUrl}/documents/${encodeURIComponent(document_id)}`,
    { method: "DELETE", headers: authHeaders() },
  );
  await checked(res, "delete document");
}

export interface StreamHandlers {
  onRetrieval?: (info: {
    source: string | null;
    page_num: number | null;
    citations: Citation[];
    images_sent: number;
  }) => void;
  onToken?: (text: string) => void;
  onDone?: (answer: QueryAnswer) => void;
  onError?: (message: string) => void;
}

/** POST /query/stream: progressive answer tokens + final attribution. */
export async function queryRagStream(
  query: string,
  opts: { document_id?: string | null; top_k?: number; max_images?: number } = {},
  handlers: StreamHandlers = {},
  apiUrl: string = API_URL,
): Promise<void> {
  const res = await fetch(`${apiUrl}/query/stream`, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    }),
    body: JSON.stringify({
      query,
      document_id: opts.document_id ?? null,
      top_k: opts.top_k ?? 5,
      max_images: opts.max_images ?? 4,
    }),
  });
  if (!res.ok || !res.body) {
    await checked(res, "query stream");
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let event = "";
  const dispatch = (data: string) => {
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      handlers.onError?.("malformed stream frame");
      return;
    }
    if (event === "retrieval")
      handlers.onRetrieval?.(payload as Parameters<NonNullable<StreamHandlers["onRetrieval"]>>[0]);
    else if (event === "token")
      handlers.onToken?.((payload as { text: string }).text ?? "");
    else if (event === "done")
      handlers.onDone?.(payload as QueryAnswer);
    else if (event === "error")
      handlers.onError?.((payload as { detail: string }).detail ?? "stream failed");
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) dispatch(data);
    }
  }
}
