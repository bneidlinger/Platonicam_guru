"""
RAG Retriever - Handles context retrieval and preparation.

Retrieves relevant documents from ChromaDB and prepares context for LLM.
"""
import re
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.embeddings.ollama_embed import OllamaEmbedder
from src.vectorstore.chroma_store import ChromaStore
from src.rag.prompts import format_context, format_poe_data, format_metadata_summary


class Retriever:
    """
    Retrieves relevant context from vector store for RAG.
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        store: Optional[ChromaStore] = None,
        top_k: int = Settings.TOP_K,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.store = store or ChromaStore()
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
        model_num: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question.
            n_results: Override default top_k.
            vendor: Filter by vendor.
            doc_type: Filter by document type.
            model_num: Filter by model number.

        Returns:
            List of relevant documents with content and metadata.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Build filter
        where = {}
        if vendor:
            where["vendor"] = vendor.lower()
        if doc_type:
            where["doc_type"] = doc_type.lower()
        if model_num:
            where["model_num"] = model_num.upper()

        # Search
        results = self.store.search(
            query_embedding=query_embedding,
            n_results=n_results or self.top_k,
            where=where if where else None,
        )

        return results

    def retrieve_for_models(self, models: list[str], query: str = "") -> list[dict]:
        """
        Retrieve documents for specific camera models.

        Args:
            models: List of model numbers.
            query: Optional additional query context.

        Returns:
            Combined results for all models.
        """
        all_results = []
        seen_ids = set()

        for model in models:
            # Search with model filter
            search_query = f"{model} {query}".strip()
            results = self.retrieve(
                query=search_query,
                model_num=model,
                n_results=3,  # Fewer per model when combining
            )

            for result in results:
                # Deduplicate
                content_id = hash(result.get("content", "")[:100])
                if content_id not in seen_ids:
                    seen_ids.add(content_id)
                    all_results.append(result)

        return all_results

    def retrieve_with_context(
        self,
        query: str,
        vendor: Optional[str] = None,
        max_context_length: int = 4000,
    ) -> dict:
        """
        Retrieve documents and prepare formatted context.

        Args:
            query: User's question.
            vendor: Optional vendor filter.
            max_context_length: Maximum context string length.

        Returns:
            Dict with 'results', 'context', 'metadata_summary'.
        """
        results = self.retrieve(query, vendor=vendor)

        return {
            "results": results,
            "context": format_context(results, max_length=max_context_length),
            "metadata_summary": format_metadata_summary(results),
        }

    def retrieve_poe_context(
        self,
        query: str,
        models: Optional[list[str]] = None,
    ) -> dict:
        """
        Retrieve context specifically for POE queries.

        Extracts model numbers from query if not provided,
        and includes verified POE data from metadata.

        Args:
            query: User's question about power consumption.
            models: Optional list of model numbers.

        Returns:
            Dict with context, results, and POE data.
        """
        # Extract model numbers from query if not provided
        if not models:
            models = self._extract_model_numbers(query)

        # Get relevant documents
        if models:
            results = self.retrieve_for_models(models, query)
        else:
            results = self.retrieve(query)

        # Get verified POE data from metadata
        if models:
            poe_info = self.store.calculate_poe_budget(models)
        else:
            # Extract models from results
            result_models = []
            for r in results:
                model = r.get("metadata", {}).get("model_num")
                if model and model not in result_models:
                    result_models.append(model)
            poe_info = self.store.calculate_poe_budget(result_models)

        return {
            "results": results,
            "context": format_context(results),
            "poe_data": format_poe_data(poe_info),
            "poe_info": poe_info,
            "models": models,
        }

    def retrieve_accessory_context(
        self,
        query: str,
        model_num: Optional[str] = None,
    ) -> dict:
        """
        Retrieve context for accessory queries.

        Searches accessory documents and includes image references.

        Args:
            query: User's question about accessories.
            model_num: Optional model to find accessories for.

        Returns:
            Dict with context, results, and image references.
        """
        # Search in accessory documents
        results = self.retrieve(
            query=query,
            doc_type="accessory",
            model_num=model_num,
        )

        # If not enough results, also search general docs
        if len(results) < 3:
            general_results = self.retrieve(query=query, model_num=model_num)
            # Merge, avoiding duplicates
            seen = set(r.get("content", "")[:50] for r in results)
            for r in general_results:
                if r.get("content", "")[:50] not in seen:
                    results.append(r)
                    if len(results) >= self.top_k:
                        break

        # Extract image references
        image_refs = []
        for r in results:
            imgs = r.get("metadata", {}).get("image_refs", "")
            if imgs:
                if isinstance(imgs, str):
                    image_refs.extend(imgs.split(","))
                else:
                    image_refs.extend(imgs)

        return {
            "results": results,
            "context": format_context(results),
            "image_refs": list(set(image_refs)),
        }

    def _extract_model_numbers(self, text: str) -> list[str]:
        """
        Extract camera model numbers from text.

        Args:
            text: Text that may contain model numbers.

        Returns:
            List of found model numbers.
        """
        # Pattern for common camera model formats
        pattern = r"\b([A-Z]{1,4}[-]?[A-Z0-9]{3,10}(?:[-][A-Z0-9]+)?)\b"
        matches = re.findall(pattern, text.upper())

        # Filter out common false positives
        false_positives = {"POE", "IEEE", "RTSP", "HTTP", "HTTPS", "IP66", "IP67"}
        return [m for m in matches if m not in false_positives and len(m) >= 5]


if __name__ == "__main__":
    print("Testing Retriever...")
    print("-" * 40)

    retriever = Retriever()

    print(f"Top-k: {retriever.top_k}")
    print(f"Store count: {retriever.store.count()}")

    if retriever.store.count() > 0:
        # Test retrieval
        results = retriever.retrieve("camera power consumption")
        print(f"\nFound {len(results)} results for 'camera power consumption'")

        if results:
            print(f"First result source: {results[0].get('metadata', {}).get('source_file')}")
    else:
        print("\nStore is empty. Run ingestion first.")
