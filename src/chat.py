"""
Chat CLI - Interactive RAG-powered conversation interface.

Full-featured chat with:
- Conversation memory
- Vendor filtering
- POE budget calculations
- Streaming responses
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.rag.chain import RAGChain


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("  SURVEILLANCE DESIGN ASSISTANT")
    print("  RAG-Powered Technical Support")
    print("=" * 60)
    print("\nCommands:")
    print("  /vendor <name>    - Filter by vendor (hanwha, axis, bosch, all)")
    print("  /poe <models>     - Calculate POE budget (comma-separated)")
    print("  /clear            - Clear conversation history")
    print("  /status           - Show current context")
    print("  /help             - Show this help")
    print("  /quit             - Exit")
    print("\nAsk any question about camera specifications, accessories,")
    print("power requirements, or system design.")
    print("-" * 60)


def print_status(chain: RAGChain):
    """Print current conversation status."""
    summary = chain.get_conversation_summary()
    vendor = chain.memory.get_context_vendor()

    print("\n--- Current Status ---")
    print(f"  Vendor filter: {vendor or 'all'}")
    print(f"  Messages: {summary['message_count']}")
    print(f"  Models discussed: {', '.join(summary['current_models']) or 'none'}")
    print(f"  Store chunks: {chain.retriever.store.count()}")
    print("-" * 22)


def handle_poe_command(chain: RAGChain, arg: str):
    """Handle POE budget calculation."""
    if not arg:
        print("Usage: /poe <model1>,<model2>,...")
        print("Example: /poe XNV-8080R,P3265-LVE")
        return

    models = [m.strip().upper() for m in arg.split(",")]
    budget = chain.calculate_poe_budget(models)

    print("\n--- POE Budget Calculation ---")
    print("(Values from document metadata - verified)")
    print()

    for model, watts in budget["by_model"].items():
        print(f"  {model}: {watts}W")

    if budget["missing"]:
        print(f"\n  Missing data: {', '.join(budget['missing'])}")

    print("-" * 30)
    print(f"  TOTAL: {budget['total_watts']:.1f}W")
    print()


def interactive_chat(chain: RAGChain, stream: bool = True):
    """Run interactive chat session."""
    print_welcome()

    # Check store
    if chain.retriever.store.count() == 0:
        print("\n⚠️  Warning: Vector store is empty!")
        print("   Run: python -m src.ingest")
        print("   Then add PDFs to data/pdfs/<vendor>/\n")

    while True:
        try:
            # Build prompt
            vendor = chain.memory.get_context_vendor()
            prompt_prefix = f"[{vendor}]" if vendor else "[all]"
            user_input = input(f"\n{prompt_prefix} You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    print("\nGoodbye!")
                    break

                elif cmd == "/help":
                    print_welcome()

                elif cmd == "/clear":
                    chain.clear_memory()
                    print("Conversation cleared.")

                elif cmd == "/status":
                    print_status(chain)

                elif cmd == "/vendor":
                    if arg.lower() in Settings.VENDORS:
                        chain.set_vendor_filter(arg.lower())
                        print(f"Filtering by vendor: {arg.lower()}")
                    elif arg.lower() == "all" or not arg:
                        chain.clear_vendor_filter()
                        print("Showing all vendors.")
                    else:
                        print(f"Unknown vendor. Options: {', '.join(Settings.VENDORS)}, all")

                elif cmd == "/poe":
                    handle_poe_command(chain, arg)

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands.")

                continue

            # Process query
            print(f"\n{prompt_prefix} Assistant: ", end="", flush=True)

            if stream:
                # Stream response
                for token in chain.query(user_input, stream=True):
                    print(token, end="", flush=True)
                print()  # Newline after streaming
            else:
                response = chain.query(user_input, stream=False)
                print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Try again or type /help for commands.")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG-powered chat for camera documentation"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable response streaming",
    )
    parser.add_argument(
        "--vendor",
        choices=Settings.VENDORS,
        help="Start with vendor filter",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Single query (non-interactive mode)",
    )

    args = parser.parse_args()

    # Initialize chain
    chain = RAGChain()

    # Check LLM availability
    if not chain.llm.check_model_available():
        print(f"Error: LLM model '{chain.llm.model}' not available!")
        print(f"Run: ollama pull {chain.llm.model}")
        sys.exit(1)

    # Set initial vendor if specified
    if args.vendor:
        chain.set_vendor_filter(args.vendor)

    # Single query mode
    if args.query:
        response = chain.query(args.query, stream=False)
        print(response)
        return

    # Interactive mode
    interactive_chat(chain, stream=not args.no_stream)


if __name__ == "__main__":
    main()
