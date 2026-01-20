"""
RAG Chain - Main orchestrator for Retrieval-Augmented Generation.

Combines retrieval, prompt construction, and LLM generation into a unified pipeline.
"""
from typing import Generator, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.rag.llm_client import OllamaLLM
from src.rag.retriever import Retriever
from src.rag.memory import ConversationMemory
from src.rag.prompts import (
    SYSTEM_PROMPT,
    RAG_TEMPLATE,
    RAG_TEMPLATE_WITH_METADATA,
    POE_QUERY_TEMPLATE,
    ACCESSORY_QUERY_TEMPLATE,
    COMPARISON_TEMPLATE,
    FOLLOWUP_TEMPLATE,
    classify_query,
    format_context,
)


class RAGChain:
    """
    Main RAG pipeline for the Surveillance Design Assistant.

    Orchestrates:
    1. Query classification
    2. Context retrieval
    3. Prompt construction
    4. LLM generation
    5. Response formatting
    """

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        retriever: Optional[Retriever] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        self.llm = llm or OllamaLLM()
        self.retriever = retriever or Retriever()
        self.memory = memory or ConversationMemory()

    def query(
        self,
        question: str,
        vendor: Optional[str] = None,
        use_memory: bool = True,
        stream: bool = False,
    ):
        """
        Process a user query through the RAG pipeline.

        Args:
            question: User's question.
            vendor: Optional vendor filter.
            use_memory: Whether to use conversation memory.
            stream: Whether to stream the response.

        Returns:
            Response string, or generator if streaming.
        """
        # Use vendor from memory if not specified
        if not vendor and use_memory:
            vendor = self.memory.get_context_vendor()

        # Classify query type
        query_type = classify_query(question)

        # Route to specialized handler
        if query_type == "poe":
            return self._handle_poe_query(question, vendor, stream)
        elif query_type == "accessory":
            return self._handle_accessory_query(question, vendor, stream)
        elif query_type == "comparison":
            return self._handle_comparison_query(question, vendor, stream)
        else:
            return self._handle_general_query(question, vendor, use_memory, stream)

    def _handle_general_query(
        self,
        question: str,
        vendor: Optional[str],
        use_memory: bool,
        stream: bool,
    ):
        """Handle general queries with standard RAG."""
        # Check for follow-up
        is_followup = use_memory and self.memory.is_followup(question)

        # Retrieve context
        context_data = self.retriever.retrieve_with_context(question, vendor=vendor)

        # Build prompt
        if is_followup and self.memory.messages:
            prompt = FOLLOWUP_TEMPLATE.format(
                history=self.memory.get_history_string(last_n=3),
                context=context_data["context"],
                question=question,
            )
        else:
            prompt = RAG_TEMPLATE_WITH_METADATA.format(
                context=context_data["context"],
                metadata_summary=context_data["metadata_summary"],
                question=question,
            )

        # Generate response
        if stream:
            return self._stream_response(prompt, question, context_data)
        else:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            self._update_memory(question, response, context_data)
            return response

    def _handle_poe_query(
        self,
        question: str,
        vendor: Optional[str],
        stream: bool,
    ):
        """Handle POE/power consumption queries."""
        # Get POE-specific context with verified metadata
        context_data = self.retriever.retrieve_poe_context(question)

        prompt = POE_QUERY_TEMPLATE.format(
            context=context_data["context"],
            poe_data=context_data["poe_data"],
            question=question,
        )

        if stream:
            return self._stream_response(prompt, question, context_data)
        else:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            self._update_memory(question, response, context_data)
            return response

    def _handle_accessory_query(
        self,
        question: str,
        vendor: Optional[str],
        stream: bool,
    ):
        """Handle accessory/mounting queries."""
        # Extract model if present
        models = self.retriever._extract_model_numbers(question)
        model_num = models[0] if models else None

        # Get accessory-specific context
        context_data = self.retriever.retrieve_accessory_context(
            question,
            model_num=model_num,
        )

        prompt = ACCESSORY_QUERY_TEMPLATE.format(
            context=context_data["context"],
            question=question,
        )

        # Add image reference note if available
        if context_data.get("image_refs"):
            prompt += f"\n\nNote: {len(context_data['image_refs'])} related images are available for visual verification."

        if stream:
            return self._stream_response(prompt, question, context_data)
        else:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            self._update_memory(question, response, context_data)
            return response

    def _handle_comparison_query(
        self,
        question: str,
        vendor: Optional[str],
        stream: bool,
    ):
        """Handle comparison queries between models."""
        # Extract models to compare
        models = self.retriever._extract_model_numbers(question)

        if len(models) < 2:
            # Not enough models found, treat as general query
            return self._handle_general_query(question, vendor, True, stream)

        # Get context for all models
        results = self.retriever.retrieve_for_models(models, question)

        prompt = COMPARISON_TEMPLATE.format(
            context=format_context(results),
            models=", ".join(models),
            question=question,
        )

        context_data = {"results": results, "context": format_context(results)}

        if stream:
            return self._stream_response(prompt, question, context_data)
        else:
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            self._update_memory(question, response, context_data)
            return response

    def _stream_response(
        self,
        prompt: str,
        question: str,
        context_data: dict,
    ) -> Generator[str, None, None]:
        """Stream response and update memory when complete."""
        full_response = []

        for token in self.llm.stream(prompt, system=SYSTEM_PROMPT):
            full_response.append(token)
            yield token

        # Update memory with complete response
        self._update_memory(question, "".join(full_response), context_data)

    def _update_memory(
        self,
        question: str,
        response: str,
        context_data: dict,
    ) -> None:
        """Update conversation memory."""
        # Extract models from context
        models = []
        for result in context_data.get("results", []):
            model = result.get("metadata", {}).get("model_num")
            if model and model not in models:
                models.append(model)

        # Extract sources
        sources = []
        for result in context_data.get("results", []):
            source = result.get("metadata", {}).get("source_file")
            if source and source not in sources:
                sources.append(source)

        self.memory.add_user_message(question, models=models)
        self.memory.add_assistant_message(response, sources=sources)

    def set_vendor_filter(self, vendor: str) -> None:
        """Set vendor filter for subsequent queries."""
        self.memory.set_vendor_context(vendor)

    def clear_vendor_filter(self) -> None:
        """Clear vendor filter."""
        self.memory.clear_vendor_context()

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()

    def get_conversation_summary(self) -> dict:
        """Get summary of current conversation state."""
        return self.memory.get_summary()

    def calculate_poe_budget(self, models: list[str]) -> dict:
        """
        Direct POE budget calculation from metadata.

        Design Principle: Use metadata for computation, not LLM.
        """
        return self.retriever.store.calculate_poe_budget(
            [m.upper() for m in models]
        )


def create_chain() -> RAGChain:
    """Factory function to create a configured RAG chain."""
    return RAGChain()


if __name__ == "__main__":
    print("Testing RAGChain...")
    print("-" * 40)

    chain = RAGChain()

    # Check prerequisites
    if not chain.llm.check_model_available():
        print(f"LLM model '{chain.llm.model}' not available!")
        print(f"Run: ollama pull {chain.llm.model}")
        exit(1)

    if chain.retriever.store.count() == 0:
        print("Vector store is empty. Run ingestion first.")
        print("  python -m src.ingest")
        exit(1)

    print(f"LLM: {chain.llm.model}")
    print(f"Store: {chain.retriever.store.count()} chunks")

    # Test query
    print("\nTest query: 'What is the power consumption of cameras?'")
    print("-" * 40)

    response = chain.query("What is the power consumption of cameras?")
    print(response)
