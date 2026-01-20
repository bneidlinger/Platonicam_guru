# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise Surveillance Design Assistant - A local-first RAG (Retrieval-Augmented Generation) system for parsing vendor PDFs (Hanwha, Axis, Bosch) and providing design-level intelligence for Physical Security Systems Engineering.

## Architecture

```
[PDF Library] -> [PyMuPDF Parser] -> [Text + Image Extraction]
-> [Regex Metadata Extraction] -> [Recursive Character Splitting]
-> [Ollama Embedding (nomic-embed-text)] -> [ChromaDB Vector Store]
-> [User Query] -> [RAG Chain] -> [Llama 3.1 8B] -> [Streamlit UI]
```

## Design Principle

**LLMs for summarization, Metadata for computation.** POE budgets and numerical calculations use extracted metadata fields—never LLM-generated numbers.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (required)
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Ingest PDFs into vector store
python -m src.ingest
python -m src.ingest --vendor hanwha
python -m src.ingest --clear --force

# CLI Chat (streaming)
python -m src.chat

# CLI Search
python -m src.search "power consumption"
python -m src.search --poe "XNV-8080R,P3265-LVE"

# Streamlit Web UI
python run_app.py
# or: streamlit run app/streamlit_app.py

# Run tests
pytest
pytest tests/test_metadata_extractor.py -v
```

## Tech Stack

- **Backend**: Python 3.11+
- **LLM Engine**: Ollama (local) - `llama3.1:8b` for chat, `nomic-embed-text` for embeddings
- **Vector DB**: ChromaDB (persistent, stored in `./chroma_db/`)
- **UI**: Streamlit (multi-page app)
- **PDF Parsing**: PyMuPDF (fitz) - text and image extraction
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter (chunk_size=1200, overlap=150)

## Project Structure

```
platonicam_guru/
├── app/
│   ├── streamlit_app.py     # Main chat UI
│   └── pages/
│       ├── 1_📁_Ingestion.py  # PDF upload & processing
│       └── 2_🔍_Database.py   # Search & browse
├── src/
│   ├── parser/
│   │   ├── pdf_parser.py         # Text + image extraction
│   │   ├── metadata_extractor.py # Regex field extraction
│   │   └── batch_processor.py    # Batch processing
│   ├── embeddings/
│   │   └── ollama_embed.py       # Embedding generation
│   ├── vectorstore/
│   │   └── chroma_store.py       # ChromaDB operations
│   ├── rag/
│   │   ├── llm_client.py         # Ollama chat client
│   │   ├── retriever.py          # Context retrieval
│   │   ├── prompts.py            # System prompts
│   │   ├── memory.py             # Conversation memory
│   │   └── chain.py              # RAG orchestrator
│   ├── ingest.py                 # Ingestion pipeline CLI
│   ├── search.py                 # Search CLI
│   └── chat.py                   # Chat CLI
├── config/
│   └── settings.py               # Configuration
├── tests/
├── data/pdfs/{vendor}/           # Source PDFs (gitignored)
├── assets/images/                # Extracted images (gitignored)
├── chroma_db/                    # Vector store (gitignored)
└── docs/                         # Setup guides
```

## Metadata Schema (Tiered)

- **Tier 1 (Document)**: `vendor`, `doc_type`, `source_file`
- **Tier 2 (Engineering)**: `model_num`, `poe_wattage`, `poe_class`, `brand`
- **Tier 3 (Visual)**: `image_refs`, `page_num`, `chunk_index`

## Key Modules

- `RAGChain` - Main orchestrator with query classification (POE, accessory, comparison, spec, general)
- `ChromaStore` - Vector storage with metadata filtering and POE budget calculation
- `MetadataExtractor` - Regex patterns for model numbers, wattage, PoE class
- `ConversationMemory` - Multi-turn support with follow-up detection

## Streamlit Pages

1. **Chat** (main) - RAG-powered conversation with source citations
2. **Ingestion** - PDF upload, batch processing, database management
3. **Database** - Search, browse, POE lookup

## Development Notes

- Query classification routes to specialized prompts (POE, accessory, comparison)
- POE calculations use `store.calculate_poe_budget()` - metadata only
- Conversation memory tracks models discussed for context
- Embeddings: 768 dimensions (nomic-embed-text)
- Top-k retrieval: 5 chunks default
