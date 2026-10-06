import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisuRAG — Multimodal RAG for Datasheets",
  description:
    "Ask technical questions over engineering datasheets and schematics with visual source attribution.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-100 text-slate-900 antialiased">{children}</body>
    </html>
  );
}
