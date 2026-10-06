"""Vision-Language Model integration.

Providers:

- :class:`EchoVLMProvider` — offline deterministic fallback. Extracts the
  top retrieved snippet so the full ``/query`` path works with zero
  downloads and no Ollama server (tests, CI, demos).
- :class:`OllamaVLMProvider` — local Ollama (``llama3.2-vision``,
  ``qwen2-vl``, …) via the OpenAI-compatible ``/v1/chat/completions``
  endpoint, with fallback to native ``/api/chat``.
- :class:`OpenAICompatibleVLMProvider` — any OpenAI-style vision endpoint
  (vLLM, LM Studio, …) via ``/chat/completions``.
- :class:`GroqVLMProvider` — hosted Groq inference (OpenAI-compatible,
  free API key, no local server) with a vision-capable model.

``build_rag_prompt()`` constructs the grounded multimodal prompt: numbered
``[Sn]`` context blocks (source + page + chunk text) plus an instruction to
cite ``[Sn]`` ids, so answers stay attributable to PDF pages/patches.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Protocol

from PIL import Image

SYSTEM_PROMPT = (
    "You are VisuRAG, an engineering-datasheet assistant. "
    "Answer ONLY from the retrieved context blocks below. "
    "Cite sources inline as [S1], [S2], etc. "
    "If the context does not contain the answer, say so explicitly."
)


def build_rag_prompt(query: str, hits: list, *, max_chars: int = 6000) -> str:
    """Render retrieved hits into a grounded generation prompt."""
    lines = [SYSTEM_PROMPT, "", f"User question: {query}", "", "Context:"]
    budget = max_chars
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        source = payload.get("source", "?")
        page = payload.get("page_num", "?")
        text = (h.text or "").strip().replace("\n", " ")
        block = f"[S{i}] ({source} p.{page}) {text}"
        if len(block) > budget and i > 1:
            lines.append(f"[S{i}] ({source} p.{page}) [truncated: context budget]")
            break
        if len(block) > budget:  # single huge hit: hard-truncate its text
            block = block[: max(budget - 20, 100)] + " …[truncated]"
        lines.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    lines += ["", "Answer with inline [Sn] citations:"]
    return "\n".join(lines)


def encode_image_base64(path: str | Path, *, max_side: int = 1024) -> str:
    """Load a patch PNG, downscale to ``max_side``, return base64 PNG."""
    p = Path(path)
    with Image.open(p) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pick_evidence_images(hits: list, *, max_images: int) -> list[str]:
    """First ``max_images`` unique patch paths across hits (rank order)."""
    seen: list[str] = []
    for h in hits:
        for ev in h.visual_evidence or ():
            p = ev.get("image_path") if isinstance(ev, dict) else None
            if p and p not in seen:
                seen.append(p)
                if len(seen) >= max_images:
                    return seen
    return seen


class VLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, images: list[str]) -> str:
        """Generate an answer. ``images`` are base64-encoded PNGs."""
        ...


def chunk_words(text: str, *, words_per_chunk: int = 8):
    """Split text into word chunks for simulated streaming."""
    words = text.split(" ")
    for i in range(0, len(words), words_per_chunk):
        piece = " ".join(words[i : i + words_per_chunk])
        yield piece + (" " if i + words_per_chunk < len(words) else "")


class EchoVLMProvider:
    """Offline fallback: extractive answer from the top context block."""

    name = "echo-fallback"

    def generate(self, prompt: str, images: list[str]) -> str:  # noqa: ARG002
        # Pull the [S1] block back out of the prompt for a grounded snippet.
        snippet = ""
        for line in prompt.splitlines():
            if line.startswith("[S1]"):
                snippet = line[5:].strip()
                break
        if not snippet:
            return "No relevant context was retrieved, so I cannot answer. [no citations]"
        trimmed = snippet[:500] + ("…" if len(snippet) > 500 else "")
        note = f" (+{len(images)} schematic image(s) attached)" if images else ""
        return f"Based on the retrieved context [S1]: {trimmed}{note}"

    def generate_stream(self, prompt: str, images: list[str]):
        """Yield the answer in word chunks (simulated streaming)."""
        yield from chunk_words(self.generate(prompt, images))


def _chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
) -> str:
    """POST an OpenAI-style ``/chat/completions`` vision request."""
    import requests

    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        headers=headers,
        timeout=timeout_s,
    )
    try:
        resp.raise_for_status()
    except Exception as e:
        # Surface the provider's message (e.g. unknown model, bad key)
        # instead of a bare HTTP status.
        raise RuntimeError(f"{e} — {resp.text[:300]}") from e
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _chat_completion_stream(
    base_url: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
):
    """Yield answer deltas from an OpenAI-style ``stream: true`` SSE request."""
    import json

    import requests

    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        headers=headers,
        timeout=timeout_s,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError):
                continue
            piece = delta.get("content")
            if piece:
                yield piece


class OllamaVLMProvider:
    """Local Ollama vision models (``llama3.2-vision``, ``qwen2-vl``, …)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2-vision",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, images: list[str]) -> str:
        # 1) OpenAI-compatible endpoint (Ollama ≥ 0.4 serves /v1/*).
        try:
            return _chat_completion(
                f"{self.base_url}/v1", self.model, prompt, images,
                timeout_s=self.timeout_s,
            )
        except Exception:
            pass  # fall through to native /api/chat
        # 2) Native Ollama chat API.
        import requests

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images,  # native API takes raw base64
                    },
                ],
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def generate_stream(self, prompt: str, images: list[str]):
        """True token streaming via ``/v1/chat/completions``; falls back to a
        single chunk (native API / errors) so callers always get an answer."""
        try:
            yield from _chat_completion_stream(
                f"{self.base_url}/v1", self.model, prompt, images,
                timeout_s=self.timeout_s,
            )
            return
        except Exception:
            pass
        yield self.generate(prompt, images)


class OpenAICompatibleVLMProvider:
    """Any OpenAI-style vision endpoint (vLLM, LM Studio, hosted)."""

    def __init__(
        self, base_url: str, model: str,
        api_key: str | None = None, timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.name = f"openai-compatible:{model}"

    def generate(self, prompt: str, images: list[str]) -> str:
        return _chat_completion(
            self.base_url, self.model, prompt, images,
            api_key=self.api_key, timeout_s=self.timeout_s,
        )

    def generate_stream(self, prompt: str, images: list[str]):
        """True token streaming; single-chunk fallback on any SSE failure."""
        try:
            yield from _chat_completion_stream(
                self.base_url, self.model, prompt, images,
                api_key=self.api_key, timeout_s=self.timeout_s,
            )
            return
        except Exception:
            pass
        yield self.generate(prompt, images)


class GroqVLMProvider:
    """Hosted Groq inference — real vision answers with no local server.

    Needs an API key (free at https://console.groq.com/keys)::

        VISURAG_VLM_PROVIDER=groq VISURAG_GROQ_API_KEY=gsk_... \\
            uvicorn api.main:app --port 8000

    Uses Groq's OpenAI-compatible ``/chat/completions`` with a
    vision-capable model (default ``qwen/qwen3.8-27b``; swap via
    ``VISURAG_GROQ_MODEL``, e.g. ``meta-llama/llama-4-scout-17b-16e-instruct``
    if available on your account).
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self, api_key: str | None, model: str,
        base_url: str | None = None, timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Groq API key required: set VISURAG_GROQ_API_KEY "
                "(or GROQ_API_KEY). Get one free at "
                "https://console.groq.com/keys"
            )
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.name = f"groq:{model}"

    def generate(self, prompt: str, images: list[str]) -> str:
        return _chat_completion(
            self.base_url, self.model, prompt, images,
            api_key=self.api_key, timeout_s=self.timeout_s,
        )

    def generate_stream(self, prompt: str, images: list[str]):
        """True token streaming; single-chunk fallback on any SSE failure."""
        try:
            yield from _chat_completion_stream(
                self.base_url, self.model, prompt, images,
                api_key=self.api_key, timeout_s=self.timeout_s,
            )
            return
        except Exception:
            pass
        yield self.generate(prompt, images)


def create_vlm_provider(settings) -> VLMProvider:
    """Factory from :class:`APISettings` (import-cycle safe: duck-typed)."""
    import os

    provider = (settings.vlm_provider or "echo").lower()
    if provider == "ollama":
        return OllamaVLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_s=settings.vlm_timeout_s,
        )
    if provider in ("openai-compatible", "openai_compatible", "vllm"):
        if not settings.openai_compatible_base_url:
            raise ValueError(
                "VISURAG_OPENAI_COMPATIBLE_BASE_URL is required for "
                "vlm_provider='openai-compatible'"
            )
        return OpenAICompatibleVLMProvider(
            base_url=settings.openai_compatible_base_url,
            model=settings.openai_compatible_model or "default",
            api_key=settings.openai_compatible_api_key,
            timeout_s=settings.vlm_timeout_s,
        )
    if provider == "groq":
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        return GroqVLMProvider(
            api_key=api_key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
            timeout_s=settings.vlm_timeout_s,
        )
    return EchoVLMProvider()
