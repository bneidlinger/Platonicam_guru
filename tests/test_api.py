"""
API tests. All external dependencies (Ollama, ChromaDB) are faked so these
run anywhere; the real Retriever is exercised against the fake store so
model-resolution logic gets covered end to end.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from config.settings import Settings
from src.api.deps import ApiSessionStore
from src.api.main import create_app
from src.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

FAKE_WATTS = {"M1075-L": 12.95, "M1135-E": 7.2, "Q6075-E": 74.0}


class FakeStore:
    def count(self):
        return 2608

    def get_stats(self):
        return {
            "total_chunks": 2608,
            "by_vendor": {"axis": 2608},
            "by_doc_type": {"datasheet": 831, "accessory": 251},
        }

    def list_model_numbers(self):
        return set(FAKE_WATTS)

    def calculate_poe_budget(self, model_nums):
        result = {"total_watts": 0.0, "by_model": {}, "missing": []}
        for model in model_nums:
            if model in FAKE_WATTS:
                result["by_model"][model] = FAKE_WATTS[model]
                result["total_watts"] += FAKE_WATTS[model]
            else:
                result["missing"].append(model)
        return result


class FakeEmbedder:
    model = "fake-embed"

    def embed_query(self, text):
        return [0.0] * 8


class FakeLLM:
    model = "fake-chat"

    @property
    def client(self):
        return self

    def list(self):
        return {"models": []}

    def check_model_available(self):
        return True


class FakeChain:
    """Mimics RAGChain's memory side effects without touching Ollama."""

    def __init__(self, memory):
        self.memory = memory

    def query(self, question, vendor=None, use_memory=True, stream=False):
        if stream:
            def gen():
                yield "Hello"
                yield " world"
                self._update(question)
            return gen()
        self._update(question)
        return f"stub answer: {question}"

    def _update(self, question):
        self.memory.add_user_message(question, models=["M1075-L"])
        self.memory.add_assistant_message("stub", sources=["M1075-L Box.pdf"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        app.state.llm = FakeLLM()
        app.state.retriever = Retriever(embedder=FakeEmbedder(), store=FakeStore())
        app.state.sessions = ApiSessionStore(ttl_minutes=60, max_sessions=10)
        app.state.chain_factory = lambda memory: FakeChain(memory)
        app.state.ingest_runner = lambda vendor=None, clear=False, force=False: {
            "pdfs_processed": 2,
            "vendor": vendor,
        }
        yield test_client


@pytest.fixture
def app(client):
    return client.app


# ---------------------------------------------------------------------------
# Public + protected endpoints (no API key configured)
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ollama_reachable"] is True
        assert body["chunks"] == 2608
        assert body["chat_model"] == "fake-chat"


class TestQuery:
    def test_query_returns_answer_with_provenance(self, client):
        response = client.post("/query", json={"question": "power of the M1075-L?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"].startswith("stub answer")
        assert body["query_type"] == "poe"
        assert body["models"] == ["M1075-L"]
        assert body["sources"] == ["M1075-L Box.pdf"]
        assert body["session_id"] == "default"

    def test_query_stream_emits_sse(self, client):
        response = client.post(
            "/query",
            json={"question": "tell me about the M1075-L", "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = response.text
        assert 'data: {"token": "Hello"}' in text
        assert '"done": true' in text
        assert "M1075-L Box.pdf" in text

    def test_unknown_vendor_rejected(self, client):
        response = client.post(
            "/query", json={"question": "anything", "vendor": "sony"}
        )
        assert response.status_code == 422

    def test_sessions_are_isolated(self, client, app):
        client.post("/query", json={"question": "q1", "session_id": "alpha"})
        client.post("/query", json={"question": "q2", "session_id": "beta"})

        sessions = app.state.sessions.manager.sessions
        assert len(sessions["alpha"].messages) == 2
        assert len(sessions["beta"].messages) == 2
        assert sessions["alpha"].messages[0].content == "q1"
        assert sessions["beta"].messages[0].content == "q2"

    def test_delete_session_is_idempotent(self, client):
        client.post("/query", json={"question": "q", "session_id": "gone"})

        first = client.delete("/sessions/gone")
        assert first.status_code == 200
        assert first.json()["deleted"] is True

        second = client.delete("/sessions/gone")
        assert second.json()["deleted"] is False


class TestPoeBudget:
    def test_budget_resolves_partials_and_reports_missing(self, client):
        response = client.post(
            "/poe/budget", json={"models": ["m1075", "UNKNOWN-99"]}
        )
        assert response.status_code == 200
        body = response.json()
        # Partial "m1075" resolves to the stored M1075-L tag
        assert body["by_model"] == {"M1075-L": 12.95}
        assert body["total_watts"] == 12.95
        assert body["missing"] == ["UNKNOWN-99"]

    def test_empty_models_rejected(self, client):
        response = client.post("/poe/budget", json={"models": ["  "]})
        assert response.status_code == 422


class TestCatalog:
    def test_models_endpoint(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert body["models"] == sorted(FAKE_WATTS)

    def test_stats_endpoint(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        assert response.json()["total_chunks"] == 2608


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_key_enforced_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(Settings, "API_KEY", "secret")

        assert client.post("/query", json={"question": "q"}).status_code == 401
        assert client.post(
            "/query", json={"question": "q"}, headers={"X-API-Key": "wrong"}
        ).status_code == 401
        assert client.post(
            "/query", json={"question": "q"}, headers={"X-API-Key": "secret"}
        ).status_code == 200

    def test_health_stays_public_with_key(self, client, monkeypatch):
        monkeypatch.setattr(Settings, "API_KEY", "secret")
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_admin_disabled_without_key(self, client):
        response = client.post("/admin/ingest", json={})
        assert response.status_code == 403
        assert "PLATONICAM_API_KEY" in response.json()["detail"]

    def test_ingest_runs_and_reports_status(self, client, monkeypatch):
        monkeypatch.setattr(Settings, "API_KEY", "secret")
        headers = {"X-API-Key": "secret"}

        response = client.post(
            "/admin/ingest", json={"vendor": "axis"}, headers=headers
        )
        assert response.status_code == 202
        assert response.json()["running"] is True

        # TestClient executes background tasks before returning, so the
        # follow-up status reflects completion.
        status = client.get("/admin/ingest/status", headers=headers).json()
        assert status["running"] is False
        assert status["result"] == {"pdfs_processed": 2, "vendor": "axis"}
        assert status["error"] is None
