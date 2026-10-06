"use client";

import { useEffect, useRef, useState } from "react";
import EvidencePane from "@/components/EvidencePane";
import {
  API_URL,
  fetchHealth,
  ingestPdf,
  queryRag,
  type Health,
  type IngestedDoc,
  type QueryAnswer,
} from "@/lib/api";

interface ChatMessage {
  id: number;
  query: string;
  answer?: QueryAnswer;
  error?: string;
}

let nextId = 1;

export default function Home() {
  const [apiUrl] = useState(API_URL);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [docs, setDocs] = useState<IngestedDoc[]>([]);
  const [scope, setScope] = useState<string>(""); // "" = all documents
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [asking, setAsking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth(apiUrl)
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((e: unknown) => {
        setHealth(null);
        setHealthError(e instanceof Error ? e.message : String(e));
      });
  }, [apiUrl]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  const selected = messages.find((m) => m.id === selectedId)?.answer ?? null;

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const doc = await ingestPdf(file, apiUrl);
      setDocs((prev) =>
        prev.some((d) => d.document_id === doc.document_id)
          ? prev.map((d) => (d.document_id === doc.document_id ? doc : d))
          : [...prev, doc],
      );
      setScope(doc.document_id); // scope new questions to the fresh doc
      setHealth((h) =>
        h
          ? {
              ...h,
              counts: {
                ...h.counts,
                text: (h.counts.text ?? 0) + doc.text_points,
                visual: (h.counts.visual ?? 0) + doc.visual_points,
              },
            }
          : h,
      );
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function onAsk(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || asking) return;
    const id = nextId++;
    setInput("");
    setMessages((prev) => [...prev, { id, query: q }]);
    setSelectedId(id);
    setAsking(true);
    try {
      const answer = await queryRag(
        q,
        { document_id: scope || null, top_k: 5, max_images: 4 },
        apiUrl,
      );
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, answer } : m)),
      );
    } catch (err: unknown) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? { ...m, error: err instanceof Error ? err.message : String(err) }
            : m,
        ),
      );
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 font-bold text-white">
          V
        </div>
        <div>
          <h1 className="text-lg font-bold leading-tight">VisuRAG</h1>
          <p className="text-xs text-slate-500">
            Multimodal RAG for datasheets &amp; schematics
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs">
          {health ? (
            <>
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-500" />
              <span className="text-slate-600">
                {health.vlm_model} · {health.counts.text ?? 0} text /{" "}
                {health.counts.visual ?? 0} visual
              </span>
            </>
          ) : (
            <>
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
              <span className="text-slate-600" title={healthError ?? ""}>
                backend unreachable — start it with{" "}
                <code className="rounded bg-slate-100 px-1 font-mono">
                  uvicorn api.main:app --port 8000
                </code>
              </span>
            </>
          )}
        </div>
      </header>

      {/* Upload bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 text-sm">
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf"
          disabled={uploading}
          onChange={(e) => void onUpload(e.target.files?.[0])}
          className="text-xs"
        />
        {uploading && <span className="text-xs text-slate-500">Ingesting…</span>}
        {uploadError && <span className="text-xs text-red-600">{uploadError}</span>}
        {docs.length > 0 && (
          <label className="ml-auto flex items-center gap-1.5 text-xs text-slate-600">
            Scope
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              className="rounded-md border border-slate-300 bg-white px-2 py-1"
            >
              <option value="">All documents</option>
              {docs.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.source} ({d.pages}p)
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {/* Split pane */}
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        {/* Left: chat */}
        <main className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-3 overflow-y-auto p-4">
            {messages.length === 0 && (
              <div className="mx-auto mt-10 max-w-md rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
                <p className="mb-1 text-base font-medium text-slate-700">
                  Ask about your datasheets
                </p>
                <p>
                  Upload a PDF above, then ask e.g.{" "}
                  <em>“Which pin is VCC?”</em> Answers arrive with cited pages
                  and schematic patches on the right.
                </p>
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className="flex flex-col gap-1.5">
                <div className="self-end rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2 text-sm text-white">
                  {m.query}
                </div>
                {m.answer && (
                  <button
                    onClick={() => setSelectedId(m.id)}
                    className={`self-start rounded-2xl rounded-bl-sm border px-4 py-2 text-left text-sm shadow-sm transition ${
                      selectedId === m.id
                        ? "border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300"
                        : "border-slate-200 bg-white hover:border-indigo-300"
                    }`}
                    title="Show source attribution →"
                  >
                    <p className="whitespace-pre-wrap leading-relaxed">
                      {m.answer.answer}
                    </p>
                    <p className="mt-1.5 font-mono text-xs text-slate-500">
                      {m.answer.source ?? "?"} · p.
                      {m.answer.page_num ?? "?"} ·{" "}
                      {m.answer.citations.length} citation(s) ·{" "}
                      {m.answer.image_patch_paths.length} patch(es)
                    </p>
                  </button>
                )}
                {m.error && (
                  <div className="self-start rounded-2xl rounded-bl-sm border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
                    {m.error}
                  </div>
                )}
              </div>
            ))}
            {asking && (
              <div className="self-start rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-2 text-sm text-slate-500 shadow-sm">
                Retrieving + generating…
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <form
            onSubmit={(e) => void onAsk(e)}
            className="flex gap-2 border-t border-slate-200 bg-white p-3"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a technical question…"
              disabled={asking}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <button
              type="submit"
              disabled={asking || !input.trim()}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              Ask
            </button>
          </form>
        </main>

        {/* Right: visual source attribution */}
        <aside className="min-h-0 w-full border-t border-slate-200 bg-slate-50 md:w-[380px] md:border-l md:border-t-0">
          <EvidencePane answer={selected} apiUrl={apiUrl} />
        </aside>
      </div>
    </div>
  );
}
