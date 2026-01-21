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
    """

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

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

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

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            show_progress: Whether to print progress.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)

            if show_progress and (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{total} chunks")

        if show_progress:
            print(f"  Embedded {total}/{total} chunks (complete)")

        return embeddings

    def embed_chunks(self, chunks: list[dict], show_progress: bool = True) -> list[dict]:
        """
        Add embeddings to chunk dicts.

        Args:
            chunks: List of chunk dicts with 'content' key.
            show_progress: Whether to print progress.

        Returns:
            Same chunks with 'embedding' key added.
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_batch(texts, show_progress=show_progress)

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
