"""
Ollama Embedding module for generating vector embeddings locally.

Uses nomic-embed-text model (768 dimensions) via Ollama API.
"""
import hashlib
from typing import Optional

import ollama

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class OllamaEmbedder:
    """
    Generates embeddings using Ollama's local embedding models.

    nomic-embed-text is trained with task prefixes: queries and documents must
    be embedded with "search_query: " / "search_document: " respectively or
    retrieval quality degrades. Prefixes are applied automatically for nomic
    models and affect only the embedding input, never stored content.
    """

    QUERY_PREFIX = "search_query: "
    DOC_PREFIX = "search_document: "

    def __init__(
        self,
        model: str = Settings.EMBEDDING_MODEL,
        host: str = Settings.OLLAMA_HOST,
    ):
        self.model = model
        self.host = host
        self._client = None

    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def _needs_prefix(self) -> bool:
        return "nomic" in self.model.lower()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single raw text (no task prefix).

        Prefer embed_query() / embed_documents() so the correct task prefix
        is applied.

        Args:
            text: Text to embed.

        Returns:
            List of floats (embedding vector).
        """
        response = self.client.embeddings(
            model=self.model,
            prompt=text,
        )
        return response["embedding"]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the model's query task prefix."""
        prefix = self.QUERY_PREFIX if self._needs_prefix() else ""
        return self.embed_text(prefix + text)

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Embed documents for storage with the model's document task prefix."""
        prefix = self.DOC_PREFIX if self._needs_prefix() else ""
        return self._embed_many([prefix + t for t in texts], show_progress)

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Generate embeddings for multiple raw texts (no task prefix).

        Args:
            texts: List of texts to embed.
            show_progress: Whether to print progress.

        Returns:
            List of embedding vectors.
        """
        return self._embed_many(texts, show_progress)

    def _embed_many(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Batch-embed via /api/embed when available, else one at a time."""
        total = len(texts)
        embeddings: list[list[float]] = []

        if hasattr(self.client, "embed"):
            step = max(1, Settings.EMBED_BATCH_SIZE)
            for i in range(0, total, step):
                response = self.client.embed(model=self.model, input=texts[i:i + step])
                batch = getattr(response, "embeddings", None)
                if batch is None:
                    batch = response["embeddings"]
                embeddings.extend(list(batch))
                if show_progress:
                    print(f"  Embedded {min(i + step, total)}/{total} chunks")
        else:
            for i, text in enumerate(texts):
                embeddings.append(self.embed_text(text))
                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Embedded {i + 1}/{total} chunks")
            if show_progress:
                print(f"  Embedded {total}/{total} chunks (complete)")

        return embeddings

    def embed_chunks(self, chunks: list[dict], show_progress: bool = True) -> list[dict]:
        """
        Add embeddings to chunk dicts (uses the document task prefix).

        Args:
            chunks: List of chunk dicts with 'content' key.
            show_progress: Whether to print progress.

        Returns:
            Same chunks with 'embedding' key added.
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_documents(texts, show_progress=show_progress)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks

    @staticmethod
    def content_hash(text: str) -> str:
        """
        Generate a hash for text content (for deduplication).

        Args:
            text: Text to hash.

        Returns:
            MD5 hash string.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def check_model_available(self) -> bool:
        """
        Check if the embedding model is available in Ollama.

        Returns:
            True if model is available.
        """
        try:
            response = self.client.list()
            # Handle both old dict format and new object format
            if hasattr(response, 'models'):
                model_names = [m.model for m in response.models]
            else:
                model_names = [m["name"] for m in response.get("models", [])]
            # Check both exact match and with :latest tag
            return (
                self.model in model_names or
                f"{self.model}:latest" in model_names
            )
        except Exception as e:
            print(f"Error checking Ollama models: {e}")
            return False

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension (768 for nomic-embed-text).
        """
        # Test with a simple embedding
        test_embedding = self.embed_text("test")
        return len(test_embedding)


if __name__ == "__main__":
    # Test the embedder
    print("Testing OllamaEmbedder...")
    print("-" * 40)

    embedder = OllamaEmbedder()

    # Check model availability
    print(f"Model: {embedder.model}")
    print(f"Host: {embedder.host}")

    if embedder.check_model_available():
        print("Model is available!")

        # Test embedding
        test_text = "XNV-8080R camera specifications"
        embedding = embedder.embed_text(test_text)

        print(f"\nTest embedding:")
        print(f"  Input: '{test_text}'")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    else:
        print(f"Model '{embedder.model}' not found!")
        print("Run: ollama pull nomic-embed-text")
