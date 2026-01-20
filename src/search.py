"""
Search CLI - Query the vector store for camera documentation.

Supports:
- Natural language queries
- Vendor filtering
- Model-specific lookups
- POE budget calculations
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.embeddings.ollama_embed import OllamaEmbedder
from src.vectorstore.chroma_store import ChromaStore


class SearchEngine:
    """
    Search interface for camera documentation.
    """

    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.store = ChromaStore()

    def search(
        self,
        query: str,
        n_results: int = Settings.TOP_K,
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for documents matching the query.

        Args:
            query: Natural language query.
            n_results: Number of results to return.
            vendor: Filter by vendor (hanwha, axis, bosch).
            doc_type: Filter by document type.

        Returns:
            List of matching documents with metadata.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Build filter
        where = {}
        if vendor:
            where["vendor"] = vendor
        if doc_type:
            where["doc_type"] = doc_type

        # Search
        results = self.store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where if where else None,
        )

        return results

    def search_model(self, model_num: str, query: str = "") -> list[dict]:
        """
        Search for information about a specific model.

        Args:
            model_num: Model number (e.g., XNV-8080R).
            query: Optional additional query context.

        Returns:
            Matching documents for this model.
        """
        search_query = f"{model_num} {query}".strip()
        query_embedding = self.embedder.embed_text(search_query)

        return self.store.search_by_model(
            query_embedding=query_embedding,
            model_num=model_num.upper(),
        )

    def get_poe_info(self, model_num: str) -> dict:
        """
        Get POE information for a model from metadata.

        Design Principle: Use extracted metadata, not LLM generation.
        """
        wattage = self.store.get_poe_wattage(model_num.upper())

        return {
            "model": model_num.upper(),
            "poe_wattage": wattage,
            "found": wattage is not None,
        }

    def calculate_project_poe(self, model_nums: list[str]) -> dict:
        """
        Calculate total POE budget for a project.

        Args:
            model_nums: List of camera model numbers.

        Returns:
            POE budget breakdown.
        """
        return self.store.calculate_poe_budget(
            [m.upper() for m in model_nums]
        )

    def get_stats(self) -> dict:
        """Get store statistics."""
        return self.store.get_stats()


def format_result(result: dict, index: int) -> str:
    """Format a single search result for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Result {index + 1} (distance: {result.get('distance', 'N/A'):.4f})")
    lines.append(f"{'='*60}")

    meta = result.get("metadata", {})
    lines.append(f"Source: {meta.get('source_file', 'unknown')}")
    lines.append(f"Vendor: {meta.get('vendor', 'unknown')} | Page: {meta.get('page_num', '?')}")

    if meta.get("model_num"):
        lines.append(f"Model: {meta.get('model_num')}")
    if meta.get("poe_wattage"):
        lines.append(f"POE: {meta.get('poe_wattage')}W (Class {meta.get('poe_class', '?')})")

    lines.append(f"\nContent:")
    lines.append("-" * 40)

    # Truncate long content
    content = result.get("content", "")
    if len(content) > 500:
        content = content[:500] + "..."
    lines.append(content)

    return "\n".join(lines)


def interactive_mode(engine: SearchEngine):
    """Run interactive search session."""
    print("\n" + "=" * 60)
    print("SURVEILLANCE DESIGN ASSISTANT - Search Mode")
    print("=" * 60)
    print("Commands:")
    print("  <query>         - Search all documents")
    print("  /vendor <name>  - Filter by vendor (hanwha, axis, bosch)")
    print("  /model <num>    - Search specific model")
    print("  /poe <models>   - Calculate POE budget (comma-separated)")
    print("  /stats          - Show database statistics")
    print("  /quit           - Exit")
    print("-" * 60)

    current_vendor = None

    while True:
        try:
            prompt = f"\n[{current_vendor or 'all'}]> "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break

                elif cmd == "/vendor":
                    if arg.lower() in Settings.VENDORS:
                        current_vendor = arg.lower()
                        print(f"Filtering by vendor: {current_vendor}")
                    elif arg.lower() == "all" or not arg:
                        current_vendor = None
                        print("Showing all vendors")
                    else:
                        print(f"Unknown vendor. Options: {', '.join(Settings.VENDORS)}")

                elif cmd == "/model":
                    if arg:
                        results = engine.search_model(arg)
                        for i, result in enumerate(results):
                            print(format_result(result, i))
                    else:
                        print("Usage: /model <model_number>")

                elif cmd == "/poe":
                    if arg:
                        models = [m.strip() for m in arg.split(",")]
                        budget = engine.calculate_project_poe(models)
                        print(f"\nPOE Budget Calculation")
                        print("-" * 40)
                        for model, watts in budget["by_model"].items():
                            print(f"  {model}: {watts}W")
                        if budget["missing"]:
                            print(f"  Missing data: {', '.join(budget['missing'])}")
                        print("-" * 40)
                        print(f"  TOTAL: {budget['total_watts']:.1f}W")
                    else:
                        print("Usage: /poe <model1>,<model2>,...")

                elif cmd == "/stats":
                    stats = engine.get_stats()
                    print(f"\nDatabase Statistics")
                    print("-" * 40)
                    print(f"  Total chunks: {stats['total_chunks']}")
                    print(f"  By vendor: {stats['by_vendor']}")
                    print(f"  By doc type: {stats['by_doc_type']}")

                else:
                    print(f"Unknown command: {cmd}")

            else:
                # Regular search query
                results = engine.search(user_input, vendor=current_vendor)

                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, result in enumerate(results):
                        print(format_result(result, i))
                else:
                    print("No results found.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search camera documentation"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit for interactive mode)",
    )
    parser.add_argument(
        "--vendor",
        choices=Settings.VENDORS,
        help="Filter by vendor",
    )
    parser.add_argument(
        "--model",
        help="Search specific model",
    )
    parser.add_argument(
        "--poe",
        help="Calculate POE budget (comma-separated model numbers)",
    )
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=Settings.TOP_K,
        help="Number of results",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive search mode",
    )

    args = parser.parse_args()

    engine = SearchEngine()

    # Check if store has data
    if engine.store.count() == 0:
        print("Warning: Vector store is empty. Run ingestion first:")
        print("  python -m src.ingest")
        print()

    # Handle different modes
    if args.poe:
        models = [m.strip() for m in args.poe.split(",")]
        budget = engine.calculate_project_poe(models)
        print(f"POE Budget: {budget['total_watts']:.1f}W")
        for model, watts in budget["by_model"].items():
            print(f"  {model}: {watts}W")
        if budget["missing"]:
            print(f"  Missing: {', '.join(budget['missing'])}")

    elif args.model:
        results = engine.search_model(args.model, args.query or "")
        for i, result in enumerate(results):
            print(format_result(result, i))

    elif args.query and not args.interactive:
        results = engine.search(
            args.query,
            n_results=args.num_results,
            vendor=args.vendor,
        )
        for i, result in enumerate(results):
            print(format_result(result, i))

    else:
        # Interactive mode
        interactive_mode(engine)


if __name__ == "__main__":
    main()
