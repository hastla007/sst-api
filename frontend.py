import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


frontend = FastAPI(
    title="STT Frontend",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
index_file = os.path.join(static_dir, "index.html")

frontend.mount("/static", StaticFiles(directory=static_dir), name="static")


@frontend.get("/health")
async def health() -> dict:
    """Lightweight health endpoint for the static frontend service."""
    return {"status": "ok"}


@frontend.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serve the main dashboard HTML from the static directory."""
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend assets not found.")
