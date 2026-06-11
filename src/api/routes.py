"""
API routes.

Three routers with different auth postures:
- public: health probe (load balancers shouldn't need a key)
- protected: query/search endpoints (key required when configured)
- admin: ingestion (disabled entirely until a key is configured)
"""
import json
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.deps import verify_admin_key, verify_api_key
from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestStatusResponse,
    ModelsResponse,
    PoeBudgetRequest,
    PoeBudgetResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.rag.memory import ConversationMemory
from src.rag.prompts import classify_query

public_router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(verify_api_key)])
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(verify_admin_key)])


def _last_turn_meta(memory: ConversationMemory) -> tuple[list[str], list[str]]:
    """
    Pull the sources and models recorded for the most recent exchange.

    RAGChain._update_memory stores cited source files on the assistant
    message and referenced models on the user message.
    """
    sources: list[str] = []
    models: list[str] = []
    for msg in reversed(memory.messages):
        if msg.role == "assistant" and not sources:
            sources = msg.metadata.get("sources") or []
        elif msg.role == "user" and not models:
            models = msg.metadata.get("models") or []
        if sources and models:
            break
    return sources, models


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@public_router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state

    try:
        state.llm.client.list()
        ollama_reachable = True
    except Exception:
        ollama_reachable = False

    chat_ok = state.llm.check_model_available() if ollama_reachable else False

    try:
        chunks = state.retriever.store.count()
    except Exception:
        chunks = 0

    healthy = ollama_reachable and chat_ok and chunks > 0
    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama_reachable=ollama_reachable,
        chat_model_available=chat_ok,
        chat_model=state.llm.model,
        embedding_model=state.retriever.embedder.model,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Protected
# ---------------------------------------------------------------------------

@protected_router.post("/query")
def query(req: QueryRequest, request: Request):
    state = request.app.state
    memory = state.sessions.get(req.session_id)
    chain = state.chain_factory(memory)
    query_type = classify_query(req.question)

    if req.stream:
        def event_stream():
            try:
                for token in chain.query(req.question, vendor=req.vendor, stream=True):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                sources, models = _last_turn_meta(memory)
                yield "data: " + json.dumps({
                    "done": True,
                    "query_type": query_type,
                    "sources": sources,
                    "models": models,
                    "session_id": req.session_id,
                }) + "\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': f'{type(exc).__name__}: {exc}'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    answer = chain.query(req.question, vendor=req.vendor)
    sources, models = _last_turn_meta(memory)
    return QueryResponse(
        answer=answer,
        query_type=query_type,
        models=models,
        sources=sources,
        session_id=req.session_id,
    )


@protected_router.post("/poe/budget", response_model=PoeBudgetResponse)
def poe_budget(req: PoeBudgetRequest, request: Request) -> PoeBudgetResponse:
    """
    Deterministic PoE budget from verified metadata — no LLM involved.
    Partial model references resolve to stored tags; unresolvable ones are
    reported in `missing`.
    """
    retriever = request.app.state.retriever
    budget = retriever.store.calculate_poe_budget(
        retriever.resolve_model_references(req.models)
    )
    return PoeBudgetResponse(
        total_watts=round(budget["total_watts"], 2),
        by_model=budget["by_model"],
        missing=budget["missing"],
    )


@protected_router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    models = sorted(request.app.state.retriever.store.list_model_numbers())
    return ModelsResponse(models=models, count=len(models))


@protected_router.get("/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    return StatsResponse(**request.app.state.retriever.store.get_stats())


@protected_router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request) -> dict:
    """Idempotent: deleting an unknown session is not an error."""
    existed = request.app.state.sessions.delete(session_id)
    return {"session_id": session_id, "deleted": existed}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def _run_ingest(app, vendor, clear, force) -> None:
    state = app.state
    try:
        result = state.ingest_runner(vendor=vendor, clear=clear, force=force)
        state.retriever.refresh_model_cache()
        with state.ingest_lock:
            state.ingest_status = {
                **state.ingest_status,
                "running": False,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "result": result,
                "error": None,
            }
    except Exception as exc:
        with state.ingest_lock:
            state.ingest_status = {
                **state.ingest_status,
                "running": False,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


@admin_router.post("/ingest", response_model=IngestStatusResponse, status_code=202)
def trigger_ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> IngestStatusResponse:
    state = request.app.state
    with state.ingest_lock:
        if state.ingest_status.get("running"):
            raise HTTPException(status_code=409, detail="ingestion already running")
        state.ingest_status = {
            "running": True,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(_run_ingest, request.app, req.vendor, req.clear, req.force)
    return IngestStatusResponse(**state.ingest_status)


@admin_router.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status(request: Request) -> IngestStatusResponse:
    with request.app.state.ingest_lock:
        return IngestStatusResponse(**request.app.state.ingest_status)
