"""
Pydantic request/response models for the API.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    vendor: Optional[str] = Field(
        default=None,
        description=f"Restrict retrieval to one vendor: {', '.join(Settings.VENDORS)}",
    )
    session_id: str = Field(default="default", min_length=1, max_length=128)
    stream: bool = Field(
        default=False,
        description="When true the response is a text/event-stream of tokens",
    )

    @field_validator("vendor")
    @classmethod
    def vendor_must_be_known(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vendor = v.lower()
        if vendor not in Settings.VENDORS:
            raise ValueError(f"unknown vendor {v!r}; expected one of {Settings.VENDORS}")
        return vendor


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    models: list[str] = Field(default_factory=list, description="Models referenced this turn")
    sources: list[str] = Field(default_factory=list, description="Source documents cited")
    session_id: str


class PoeBudgetRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=64)

    @field_validator("models")
    @classmethod
    def models_must_be_nonempty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip() for m in v if m and m.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty model reference required")
        return cleaned


class PoeBudgetResponse(BaseModel):
    total_watts: float
    by_model: dict[str, float]
    missing: list[str] = Field(
        default_factory=list,
        description="Models with no verified wattage in the corpus",
    )


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chat_model_available: bool
    chat_model: str
    embedding_model: str
    chunks: int


class StatsResponse(BaseModel):
    total_chunks: int
    by_vendor: dict[str, int]
    by_doc_type: dict[str, int]


class ModelsResponse(BaseModel):
    models: list[str]
    count: int


class IngestRequest(BaseModel):
    vendor: Optional[str] = None
    clear: bool = Field(default=False, description="Wipe the collection before ingesting")
    force: bool = Field(default=False, description="Re-ingest chunks that already exist")

    @field_validator("vendor")
    @classmethod
    def vendor_must_be_known(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vendor = v.lower()
        if vendor not in Settings.VENDORS:
            raise ValueError(f"unknown vendor {v!r}; expected one of {Settings.VENDORS}")
        return vendor


class IngestStatusResponse(BaseModel):
    running: bool
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
