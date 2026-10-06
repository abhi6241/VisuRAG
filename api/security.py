"""Request security: optional API-key auth + per-IP rate limiting.

Both are dependency-free (stdlib only) and safe by default:

- Auth is **open** when no key is configured (current dev behavior,
  unchanged). Set ``VISURAG_API_KEY`` to require ``X-API-Key`` on every
  route except ``/health`` (liveness) and the OpenAPI docs. ``/files``
  additionally accepts ``?api_key=`` because ``<img>`` tags cannot send
  custom headers. Keys are compared with :func:`hmac.compare_digest`.
- Rate limiting is a sliding-window counter per client IP
  (``VISURAG_RATE_LIMIT_PER_MIN``, default 120, ``0`` disables).
  ``/health`` is exempt so monitors never trip it. Over-limit requests
  get 429 + a ``Retry-After`` header.

Note: the limiter is in-memory per process — with multiple uvicorn
workers each enforces its own budget. Good enough for self-hosting;
use a gateway (nginx, Cloudflare) for strict global limits.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class AuthMiddleware(BaseHTTPMiddleware):
    """Require an API key when one is configured, otherwise pass through."""

    def __init__(self, app, api_key: str | None) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        if not self.api_key or request.url.path in OPEN_PATHS:
            return await call_next(request)
        provided = request.headers.get("x-api-key") or request.query_params.get(
            "api_key"
        )
        if not provided or not hmac.compare_digest(provided, self.api_key):
            return JSONResponse(
                {"detail": "invalid or missing API key"}, status_code=401
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP limiter (``per_min <= 0`` disables)."""

    def __init__(self, app, per_min: int) -> None:
        super().__init__(app)
        self.per_min = per_min
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        # Drop idle clients so the table cannot grow without bound.
        if len(self._hits) < 10000:
            return
        cutoff = now - 60.0
        for ip in [ip for ip, dq in self._hits.items() if not dq or dq[-1] <= cutoff]:
            del self._hits[ip]

    async def dispatch(self, request, call_next):
        if not self.per_min or request.url.path in OPEN_PATHS:
            return await call_next(request)
        now = time.monotonic()
        client = request.client.host if request.client else "unknown"
        with self._lock:
            dq = self._hits.setdefault(client, deque())
            while dq and dq[0] <= now - 60.0:
                dq.popleft()
            if len(dq) >= self.per_min:
                retry = max(1, int(dq[0] + 60.0 - now))
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
            dq.append(now)
            self._prune(now)
        return await call_next(request)
