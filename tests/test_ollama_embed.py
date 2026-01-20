"""
Tests for OllamaEmbedder.

Note: Tests that require Ollama are marked with @pytest.mark.ollama
and can be skipped if Ollama is not running.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.ollama_embed import OllamaEmbedder


@pytest.fixture
def embedder():
    return OllamaEmbedder()


def ollama_available():
    """Check if Ollama is running and model is available."""
    try:
        embedder = OllamaEmbedder()
        return embedder.check_model_available()
    except Exception:
        return False


# Mark for tests requiring Ollama
requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running or model not available"
)


class TestOllamaEmbedderInit:
    """Test embedder initialization."""

    def test_default_model(self, embedder):
        from config.settings import Settings
        assert embedder.model == Settings.EMBEDDING_MODEL

    def test_default_host(self, embedder):
        from config.settings import Settings
        assert embedder.host == Settings.OLLAMA_HOST

    def test_custom_config(self):
        custom = OllamaEmbedder(
            model="custom-model",
            host="http://custom:11434",
        )
        assert custom.model == "custom-model"
        assert custom.host == "http://custom:11434"


class TestContentHash:
    """Test content hashing for deduplication."""

    def test_hash_consistency(self, embedder):
        text = "Test content"
        hash1 = embedder.content_hash(text)
        hash2 = embedder.content_hash(text)
        assert hash1 == hash2

    def test_hash_different_content(self, embedder):
        hash1 = embedder.content_hash("Content A")
        hash2 = embedder.content_hash("Content B")
        assert hash1 != hash2

    def test_hash_format(self, embedder):
        hash_value = embedder.content_hash("Test")
        # MD5 hash is 32 hex characters
        assert len(hash_value) == 32
        assert all(c in "0123456789abcdef" for c in hash_value)


@requires_ollama
class TestOllamaEmbedderWithServer:
    """Tests that require Ollama to be running."""

    def test_model_available(self, embedder):
        assert embedder.check_model_available()

    def test_embed_text(self, embedder):
        embedding = embedder.embed_text("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    def test_embedding_dimension(self, embedder):
        dim = embedder.get_embedding_dimension()

        # nomic-embed-text produces 768-dim embeddings
        assert dim == 768

    def test_embed_batch(self, embedder):
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embedder.embed_batch(texts, show_progress=False)

        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)

    def test_embed_chunks(self, embedder):
        chunks = [
            {"content": "Chunk one", "metadata": {}},
            {"content": "Chunk two", "metadata": {}},
        ]

        result = embedder.embed_chunks(chunks, show_progress=False)

        assert len(result) == 2
        assert "embedding" in result[0]
        assert "embedding" in result[1]
        assert len(result[0]["embedding"]) == 768

    def test_similar_texts_similar_embeddings(self, embedder):
        """Semantically similar texts should have similar embeddings."""
        emb1 = embedder.embed_text("camera power consumption")
        emb2 = embedder.embed_text("power usage of the camera")
        emb3 = embedder.embed_text("weather forecast tomorrow")

        # Calculate cosine similarity (simplified)
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_12 = cosine_sim(emb1, emb2)
        sim_13 = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13
