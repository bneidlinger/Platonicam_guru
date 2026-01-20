"""
ChromaDB Vector Store module for persistent semantic search.

Handles document storage, retrieval, and metadata filtering.
"""
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class ChromaStore:
    """
    ChromaDB vector store for camera documentation.

    Supports:
    - Persistent storage
    - Metadata filtering by vendor, doc_type, model_num
    - Incremental updates (skip existing content)
    - Tiered metadata schema
    """

    COLLECTION_NAME = "camera_docs"

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir or Settings.CHROMA_DIR
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Surveillance camera documentation",
                    "hnsw:space": "cosine",  # Use cosine similarity
                },
            )
        return self._collection

    @staticmethod
    def generate_id(content: str, source_file: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.

        Uses content hash + source info for deduplication.
        """
        hash_input = f"{source_file}_{chunk_index}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def add_chunks(
        self,
        chunks: list[dict],
        skip_existing: bool = True,
        show_progress: bool = True,
    ) -> dict:
        """
        Add chunks with embeddings to the collection.

        Args:
            chunks: List of chunk dicts with 'content', 'embedding', 'metadata'.
            skip_existing: Skip chunks that already exist (by ID).
            show_progress: Print progress updates.

        Returns:
            Dict with 'added' and 'skipped' counts.
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        existing_ids = set()
        if skip_existing:
            # Get existing IDs to check for duplicates
            existing = self.collection.get()
            existing_ids = set(existing.get("ids", []))

        skipped = 0
        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            embedding = chunk.get("embedding")

            if embedding is None:
                raise ValueError("Chunk missing 'embedding' - run embedder first")

            # Generate unique ID
            chunk_id = self.generate_id(
                content,
                metadata.get("source_file", "unknown"),
                metadata.get("chunk_index", 0),
            )

            # Skip if exists
            if skip_existing and chunk_id in existing_ids:
                skipped += 1
                continue

            # Prepare for insertion
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(content)

            # Clean metadata for ChromaDB (only primitive types)
            clean_metadata = self._clean_metadata(metadata)
            metadatas.append(clean_metadata)

        # Batch insert
        if ids:
            # ChromaDB has batch size limits, so chunk if needed
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                )

                if show_progress:
                    print(f"  Added batch {i//batch_size + 1}: {end - i} chunks")

        result = {"added": len(ids), "skipped": skipped}

        if show_progress:
            print(f"  Total: {result['added']} added, {result['skipped']} skipped")

        return result

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Clean metadata for ChromaDB storage.

        ChromaDB only supports str, int, float, bool values.
        Lists are converted to comma-separated strings.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                clean[key] = ",".join(str(v) for v in value)
            else:
                # Convert other types to string
                clean[key] = str(value)
        return clean

    def search(
        self,
        query_embedding: list[float],
        n_results: int = Settings.TOP_K,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Metadata filter (e.g., {"vendor": "hanwha"}).
            where_document: Document content filter.

        Returns:
            List of result dicts with 'content', 'metadata', 'distance'.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return formatted

    def search_by_vendor(
        self,
        query_embedding: list[float],
        vendor: str,
        n_results: int = Settings.TOP_K,
    ) -> list[dict]:
        """
        Search within a specific vendor's documents.
        """
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where={"vendor": vendor},
        )

    def search_by_model(
        self,
        query_embedding: list[float],
        model_num: str,
        n_results: int = Settings.TOP_K,
    ) -> list[dict]:
        """
        Search for documents about a specific model.
        """
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where={"model_num": model_num},
        )

    def search_accessories(
        self,
        query_embedding: list[float],
        n_results: int = Settings.TOP_K,
    ) -> list[dict]:
        """
        Search only accessory documents.
        """
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where={"doc_type": "accessory"},
        )

    def get_all_by_model(self, model_num: str) -> list[dict]:
        """
        Get all chunks for a specific model (for POE calculation, etc.).
        """
        results = self.collection.get(
            where={"model_num": model_num},
            include=["documents", "metadatas"],
        )

        formatted = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return formatted

    def get_poe_wattage(self, model_num: str) -> Optional[float]:
        """
        Get POE wattage for a model from metadata.

        Design Principle: Use metadata for computation, not LLM.
        """
        docs = self.get_all_by_model(model_num)
        for doc in docs:
            wattage = doc.get("metadata", {}).get("poe_wattage")
            if wattage is not None:
                try:
                    return float(wattage)
                except (ValueError, TypeError):
                    continue
        return None

    def calculate_poe_budget(self, model_nums: list[str]) -> dict:
        """
        Calculate total POE budget for a list of camera models.

        Returns:
            Dict with 'total_watts', 'by_model', and 'missing' keys.
        """
        result = {
            "total_watts": 0.0,
            "by_model": {},
            "missing": [],
        }

        for model in model_nums:
            wattage = self.get_poe_wattage(model)
            if wattage is not None:
                result["total_watts"] += wattage
                result["by_model"][model] = wattage
            else:
                result["missing"].append(model)

        return result

    def count(self) -> int:
        """Get total document count."""
        return self.collection.count()

    def get_stats(self) -> dict:
        """Get collection statistics."""
        all_docs = self.collection.get(include=["metadatas"])

        vendors = {}
        doc_types = {}

        for meta in all_docs.get("metadatas", []):
            vendor = meta.get("vendor", "unknown")
            vendors[vendor] = vendors.get(vendor, 0) + 1

            doc_type = meta.get("doc_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            "total_chunks": self.count(),
            "by_vendor": vendors,
            "by_doc_type": doc_types,
        }

    def delete_by_source(self, source_file: str) -> int:
        """
        Delete all chunks from a specific source file.

        Useful for re-ingesting updated PDFs.
        """
        # Get IDs of chunks from this source
        results = self.collection.get(
            where={"source_file": source_file},
        )

        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def clear(self) -> int:
        """Delete all documents from the collection."""
        count = self.count()
        self.client.delete_collection(self.collection_name)
        self._collection = None  # Reset to recreate on next access
        return count


if __name__ == "__main__":
    # Test the store
    print("Testing ChromaStore...")
    print("-" * 40)

    store = ChromaStore()

    print(f"Persist directory: {store.persist_dir}")
    print(f"Collection name: {store.collection_name}")
    print(f"Current count: {store.count()}")

    stats = store.get_stats()
    print(f"\nStats: {stats}")
