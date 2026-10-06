"""Run the API: ``python -m api`` → uvicorn on port 8000."""

from __future__ import annotations

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
