# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise Surveillance Design Assistant - A local-first RAG (Retrieval-Augmented Generation) system for parsing vendor PDFs (Hanwha, Axis, Bosch) and providing design-level intelligence for Physical Security Systems Engineering.

## Design Principle

**LLMs for summarization, Metadata for computation.** POE budgets and numerical calculations use extracted metadata fields—never LLM-generated numbers. The `MetadataExtractor` uses regex patterns to extract model numbers, wattage, and PoE class, which are stored in ChromaDB metadata and used directly for calculations via `store.calculate_poe_budget()`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (required before first run)
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Ingest PDFs into vector store
python -m src.ingest
python -m src.ingest --vendor hanwha
python -m src.ingest --clear --force

# CLI Chat (streaming)
python -m src.chat
python -m src.chat "What mount fits the XNV-8080R?"

# CLI Search
python -m src.search "power consumption"
python -m src.search --poe "XNV-8080R,P3265-LVE"

# Streamlit Web UI
python run_app.py

# Run tests
pytest
pytest tests/test_metadata_extractor.py -v
pytest tests/test_metadata_extractor.py::TestPoEWattageExtraction -v  # single test class
pytest -k "test_hanwha"  # run tests matching pattern
```

## Architecture

```
PDF → PyMuPDF → Text + Images → MetadataExtractor (regex) → Chunking
    → Ollama nomic-embed-text → ChromaDB

Query → RAGChain.query() → classify_query() → route to handler
    → Retriever → ChromaDB → context + metadata
    → Prompt template → Ollama llama3.1:8b → Response
```

**Query Classification** (`src/rag/prompts.py:classify_query`): Routes queries to specialized handlers:
- `poe` - Power consumption queries → uses `POE_QUERY_TEMPLATE` with verified metadata
- `accessory` - Mount/bracket queries → uses `ACCESSORY_QUERY_TEMPLATE`
- `comparison` - Model comparisons → extracts models, retrieves context for each
- `general` - Standard RAG with conversation memory support

## Key Modules

- `RAGChain` (`src/rag/chain.py`) - Main orchestrator; routes queries to specialized handlers based on classification
- `ChromaStore` (`src/vectorstore/chroma_store.py`) - Vector storage, metadata filtering, `calculate_poe_budget()` for direct computation
- `MetadataExtractor` (`src/parser/metadata_extractor.py`) - Regex patterns extract `model_num`, `poe_wattage`, `poe_class`, `brand` from text
- `ConversationMemory` (`src/rag/memory.py`) - Tracks models discussed; detects follow-up questions
- `Retriever` (`src/rag/retriever.py`) - Context retrieval with specialized methods for POE, accessories, comparisons

## Configuration

Edit `config/settings.py` for:
- `CHUNK_SIZE` (1200) / `CHUNK_OVERLAP` (150)
- `EMBEDDING_MODEL`, `CHAT_MODEL`, `TEMPERATURE`
- `TOP_K` retrieval count (5)
- `OLLAMA_HOST` (http://localhost:11434)

## Data Layout

- `data/pdfs/{hanwha,axis,bosch}/` - Source PDFs (gitignored)
- `chroma_db/` - Persistent vector store (gitignored)
- `assets/images/` - Extracted images (gitignored)
