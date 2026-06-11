"""
FastAPI application factory.

Singletons (LLM client, retriever with its model cache, session store) are
built once at startup; each request composes a RAGChain from them plus the
caller's session memory. Sessions are in-process: run a single worker, or
put a sticky load balancer in front when scaling out.

Run locally:
    python run_api.py
    # or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.api.deps import ApiSessionStore
from src.api.routes import admin_router, protected_router, public_router
from src.rag.chain import RAGChain
from src.rag.llm_client import OllamaLLM
from src.rag.memory import ConversationMemory
from src.rag.retriever import Retriever

API_VERSION = "1.0.0"


def default_ingest_runner(
    vendor: Optional[str] = None,
    clear: bool = False,
    force: bool = False,
) -> dict:
    """Run the ingestion pipeline synchronously; returns its stats dict."""
    from src.ingest import IngestionPipeline

    pipeline = IngestionPipeline()
    if clear:
        pipeline.store.clear()
    pipeline.ingest_directory(vendor=vendor, skip_existing=not force)
    return dict(pipeline.stats)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = OllamaLLM()
    retriever = Retriever()

    app.state.llm = llm
    app.state.retriever = retriever
    app.state.sessions = ApiSessionStore()
    app.state.chain_factory = lambda memory: RAGChain(
        llm=llm, retriever=retriever, memory=memory or ConversationMemory()
    )
    app.state.ingest_runner = default_ingest_runner
    app.state.ingest_lock = threading.Lock()
    app.state.ingest_status = {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Surveillance Design Assistant API",
        description=(
            "Local-first RAG over vendor camera documentation. "
            "PoE budgets come from verified metadata, never LLM generation."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=Settings.ALLOWED_ORIGINS,
        # Credentialed CORS is incompatible with a wildcard origin
        allow_credentials=Settings.ALLOWED_ORIGINS != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(public_router)
    app.include_router(protected_router)
    app.include_router(admin_router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Ollama down mid-request, ChromaDB I/O errors, etc. land here.
        return JSONResponse(
            status_code=500,
            content={"detail": f"internal error: {type(exc).__name__}: {exc}"},
        )

    return app


app = create_app()
