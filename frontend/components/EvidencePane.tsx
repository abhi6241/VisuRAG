"use client";

import { useCallback, useEffect, useState } from "react";
import { evidenceUrl, type QueryAnswer } from "@/lib/api";

interface Props {
  answer: QueryAnswer | null;
  apiUrl: string;
}

/** Right split-pane: visual source attribution for one grounded answer. */
export default function EvidencePane({ answer, apiUrl }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  const close = useCallback(() => setLightbox(null), []);
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, close]);

  if (!answer) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
        <div className="text-4xl">📑</div>
        <p className="font-medium text-slate-600">No answer selected</p>
        <p className="max-w-xs text-sm text-slate-500">
          Ask a question and the cited datasheet pages and schematic patches
          will appear here.
        </p>
      </div>
    );
  }

  const topImages = answer.image_patch_paths ?? [];

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      {/* Top attribution */}
      <section className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Top source
        </h3>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="rounded-md bg-slate-900 px-2 py-1 font-mono text-xs text-white">
            {answer.source ?? "unknown file"}
          </span>
          {answer.page_num !== null && (
            <span className="rounded-md bg-indigo-100 px-2 py-1 text-xs font-medium text-indigo-800">
              page {answer.page_num}
            </span>
          )}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {answer.model} · {answer.images_sent} image(s) sent to VLM
        </p>
      </section>

      {/* Evidence patches */}
      <section className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Schematic patches ({topImages.length})
        </h3>
        {topImages.length === 0 ? (
          <p className="text-sm text-slate-500">No image evidence attached.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {topImages.map((p) => {
              const url = evidenceUrl(apiUrl, p);
              return (
                <button
                  key={p}
                  onClick={() => setLightbox(url)}
                  className="group overflow-hidden rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  title={p}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={url}
                    alt={`Evidence patch ${p}`}
                    loading="lazy"
                    className="aspect-square w-full object-cover transition group-hover:opacity-90"
                  />
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* Citations */}
      <section className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Citations ({answer.citations.length})
        </h3>
        <ol className="flex flex-col gap-2">
          {answer.citations.map((c, i) => (
            <li
              key={`${c.chunk_id ?? i}`}
              className="rounded-lg bg-slate-50 p-2 text-sm"
            >
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="font-mono font-bold text-indigo-700">
                  [S{i + 1}]
                </span>
                <span className="font-mono text-xs text-slate-600">
                  {c.source ?? "?"} · p.{c.page_num ?? "?"}
                </span>
                {c.rerank_score !== null && (
                  <span className="ml-auto text-xs text-slate-400">
                    rerank {c.rerank_score.toFixed(2)}
                  </span>
                )}
              </div>
              {c.text_snippet && (
                <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-slate-600">
                  {c.text_snippet}
                </p>
              )}
              {c.image_patch_paths.length > 0 && (
                <div className="mt-1.5 flex gap-1.5 overflow-x-auto">
                  {c.image_patch_paths.slice(0, 4).map((p) => {
                    const url = evidenceUrl(apiUrl, p);
                    return (
                      <button
                        key={p}
                        onClick={() => setLightbox(url)}
                        className="h-14 w-14 shrink-0 overflow-hidden rounded border border-slate-200"
                        title={p}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={url}
                          alt={`Patch ${p}`}
                          loading="lazy"
                          className="h-full w-full object-cover"
                        />
                      </button>
                    );
                  })}
                </div>
              )}
            </li>
          ))}
        </ol>
      </section>

      {/* Lightbox modal */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={close}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="max-h-full max-w-4xl overflow-auto rounded-xl bg-white p-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-xs text-slate-500">
                {lightbox}
              </span>
              <button
                onClick={close}
                className="rounded-md bg-slate-900 px-3 py-1 text-sm text-white hover:bg-slate-700"
              >
                Close
              </button>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={lightbox}
              alt="Evidence patch full size"
              className="max-h-[75vh] w-auto"
            />
          </div>
        </div>
      )}
    </div>
  );
}
