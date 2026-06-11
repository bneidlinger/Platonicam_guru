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
from src.parser.metadata_extractor import MetadataExtractor
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
        self._extractor = MetadataExtractor()
        self._model_cache: Optional[set[str]] = None

    @staticmethod
    def _build_where(
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
        model_num: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Build a ChromaDB where filter.

        ChromaDB rejects flat multi-key dicts; two or more conditions must be
        wrapped in {"$and": [...]}.
        """
        conditions = []
        if vendor:
            conditions.append({"vendor": vendor.lower()})
        if doc_type:
            conditions.append({"doc_type": doc_type.lower()})
        if model_num:
            conditions.append({"model_num": model_num.upper()})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
        model_num: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question.
            n_results: Override default top_k.
            vendor: Filter by vendor.
            doc_type: Filter by document type.
            model_num: Filter by model number (exact metadata match).
            contains: Require this string in the document text.

        Returns:
            List of relevant documents with content and metadata.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.store.search(
            query_embedding=query_embedding,
            n_results=n_results or self.top_k,
            where=self._build_where(vendor=vendor, doc_type=doc_type, model_num=model_num),
            where_document={"$contains": contains} if contains else None,
        )

        return results

    def _known_model_nums(self) -> set[str]:
        """Cached set of model_num values present in the store."""
        if self._model_cache is None:
            self._model_cache = self.store.list_model_numbers()
        return self._model_cache

    def refresh_model_cache(self) -> None:
        """Drop the cached model list (call after ingestion)."""
        self._model_cache = None

    def _resolve_models(self, model: str, limit: int = 4) -> list[str]:
        """
        Map a (possibly partial) model reference to model_num values that
        actually exist in the store.

        "M1075-L" -> ["M1075-L"]; "M1075" -> ["M1075-L"]; "P3265" -> all
        P3265 variants. Returns [] when nothing matches.
        """
        model = model.upper()
        known = self._known_model_nums()

        if model in known:
            return [model]

        # Partial reference: user typed a prefix of the stored model
        prefixed = sorted(m for m in known if m.startswith(model))
        if prefixed:
            return prefixed[:limit]

        # Reference longer than the stored tag (e.g. trailing revision suffix)
        contained = sorted(
            (m for m in known if model.startswith(m)),
            key=len,
            reverse=True,
        )
        return contained[:1]

    def resolve_model_references(self, models: list[str]) -> list[str]:
        """
        Resolve (possibly partial) model references to stored model tags.

        Unresolvable references pass through uppercased so callers can report
        them as missing instead of silently dropping them.
        """
        resolved_list = []
        for model in models:
            for resolved in self._resolve_models(model) or [model.upper()]:
                if resolved not in resolved_list:
                    resolved_list.append(resolved)
        return resolved_list

    def retrieve_for_models(self, models: list[str], query: str = "") -> list[dict]:
        """
        Retrieve documents for specific camera models.

        Fallback tiers per model, so a recognized model reference never
        produces an empty context:
        1. Exact/resolved model_num metadata filter.
        2. Document-text $contains match (covers variants that only appear
           in prose or related_models).
        3. Plain semantic search.

        Args:
            models: List of model numbers (may be partial references).
            query: Optional additional query context.

        Returns:
            Combined deduplicated results for all models.
        """
        all_results = []
        seen_ids = set()
        per_model = self.top_k if len(models) == 1 else 3

        for model in models:
            model = model.upper()
            search_query = f"{model} {query}".strip()

            results = []
            resolved = self._resolve_models(model)
            per_resolved = per_model if len(resolved) <= 1 else max(2, per_model // len(resolved))
            for resolved_model in resolved:
                results.extend(self.retrieve(
                    query=search_query,
                    model_num=resolved_model,
                    n_results=per_resolved,
                ))

            if not results:
                results = self.retrieve(search_query, n_results=per_model, contains=model)
            if not results:
                results = self.retrieve(search_query, n_results=per_model)

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
            # Resolve partial references; unresolvable ones surface in the
            # "missing" list instead of vanishing.
            poe_info = self.store.calculate_poe_budget(
                self.resolve_model_references(models)
            )
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
        # Resolve partial model references to stored tags
        if model_num:
            resolved = self._resolve_models(model_num)
            model_num = resolved[0] if resolved else model_num

        # Search in accessory documents
        results = self.retrieve(
            query=query,
            doc_type="accessory",
            model_num=model_num,
        )

        # If not enough results, widen: model-only, then unfiltered semantic
        for fallback_kwargs in ({"model_num": model_num}, {}):
            if len(results) >= 3:
                break
            general_results = self.retrieve(query=query, **fallback_kwargs)
            # Merge, avoiding duplicates
            seen = set(r.get("content", "")[:50] for r in results)
            for r in general_results:
                if r.get("content", "")[:50] not in seen:
                    results.append(r)
                    seen.add(r.get("content", "")[:50])
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

        Uses MetadataExtractor for consistent pattern matching.

        Args:
            text: Text that may contain model numbers.

        Returns:
            List of found model numbers.
        """
        return self._extractor.extract_model_numbers(text)


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
