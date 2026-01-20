# 📹 Surveillance Design Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **local-first RAG (Retrieval-Augmented Generation) system** for Physical Security Systems Engineers. Parse vendor PDFs from Hanwha, Axis, and Bosch to get instant design intelligence—all running on your machine with no cloud dependencies.

> **Design Principle:** LLMs handle summarization and natural language. Metadata handles computation. POE budgets come from extracted data—never hallucinated.

---

## ✨ Features

### 🤖 RAG-Powered Chat
- Natural language queries against your camera documentation
- Automatic query classification (POE, accessories, comparisons, specs)
- Source citations with every response
- Conversation memory for follow-up questions

### ⚡ POE Budget Calculator
- Extracts power consumption from datasheets via regex
- Calculates project totals from verified metadata—not LLM generation
- Tracks PoE class (Class 0-4) for switch compatibility

### 📷 Project Mode
- Build camera lists for system designs
- Real-time POE budget tracking
- Export to CSV/JSON for BOMs and proposals

### 🔍 Smart Search
- Semantic search across all vendor documentation
- Filter by vendor, document type, or model number
- Image extraction for visual accessory verification

### 🏠 100% Local
- Runs entirely on your machine
- Your PDFs never leave your network
- No API costs or rate limits
- Works offline once models are downloaded

---

## 🖥️ Screenshots

<details>
<summary>Click to expand screenshots</summary>

### Chat Interface
```
┌─────────────────────────────────────────────────────────────┐
│  📹 Surveillance Design Assistant                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  You: What is the power consumption of the XNV-8080R?       │
│                                                             │
│  Assistant: The Hanwha XNV-8080R has a maximum power        │
│  consumption of **25.5W** (PoE++ Class 4).                  │
│                                                             │
│  [Source: XNV-8080R_Datasheet.pdf, Page 2]                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  📋 Project Mode          │  ⚡ POE Budget                  │
│  ─────────────────────    │  ───────────────                │
│  • XNV-8080R    x4        │  XNV-8080R: 25.5W × 4 = 102W   │
│  • P3265-LVE   x8        │  P3265-LVE: 12.9W × 8 = 103.2W │
│                           │  ───────────────                │
│  [Export CSV] [JSON]      │  TOTAL: 205.2W                  │
└─────────────────────────────────────────────────────────────┘
```

</details>

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** - [Download here](https://ollama.com/download)
- **8GB+ RAM** (16GB recommended for larger models)
- **GPU optional** but recommended for faster inference

### 1. Clone & Install

```bash
git clone https://github.com/bneidlinger/platonicam_guru.git
cd platonicam_guru

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Ollama Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Chat model (choose based on your hardware)
ollama pull llama3.1:8b        # Recommended (6GB VRAM)
# OR
ollama pull phi3:mini          # Lower VRAM option (3GB)
# OR
ollama pull llama3.2:3b        # Budget GPU (4GB)
```

### 3. Add Your PDFs

```
data/pdfs/
├── hanwha/
│   ├── XNV-8080R_Datasheet.pdf
│   └── XNP-6400RW_Installation.pdf
├── axis/
│   └── P3265-LVE_Datasheet.pdf
└── bosch/
    └── NBE-3502-AL_Manual.pdf
```

### 4. Ingest Documents

```bash
# Process all PDFs
python -m src.ingest

# Or specific vendor
python -m src.ingest --vendor hanwha
```

### 5. Launch the App

```bash
python run_app.py
```

Open http://localhost:8501 in your browser.

---

## 📖 Usage

### Web UI (Recommended)

```bash
python run_app.py
```

**Pages:**
- **Chat** - Ask questions, get answers with citations
- **Ingestion** - Upload PDFs, manage database
- **Database** - Search, browse, POE lookup

### CLI Chat

```bash
# Interactive mode with streaming
python -m src.chat

# Single query
python -m src.chat "What mount fits the XNV-8080R?"

# With vendor filter
python -m src.chat --vendor hanwha
```

### CLI Search

```bash
# Semantic search
python -m src.search "outdoor vandal dome 4K"

# POE budget calculation
python -m src.search --poe "XNV-8080R,P3265-LVE,NBE-3502-AL"

# Interactive mode
python -m src.search -i
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY CLASSIFICATION                          │
│         (POE / Accessory / Comparison / Spec / General)         │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  EMBEDDING       │ │  METADATA    │ │  CONVERSATION    │
│  (nomic-embed)   │ │  LOOKUP      │ │  MEMORY          │
└──────────────────┘ └──────────────┘ └──────────────────┘
              │               │               │
              ▼               ▼               │
┌─────────────────────────────────────────────────────────────────┐
│                      CHROMADB VECTOR STORE                       │
│                   (Semantic Search + Filtering)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTEXT INJECTION                            │
│              (Retrieved docs + Verified metadata)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OLLAMA LLM                                  │
│                    (llama3.1:8b)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE + CITATIONS                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Metadata Extraction

The system extracts structured data from PDFs using regex patterns:

| Field | Pattern | Example |
|-------|---------|---------|
| Model Number | `[A-Z]{1,4}-[A-Z0-9]{4,10}` | XNV-8080R, P3265-LVE |
| POE Wattage | `\d{1,2}\.?\d?\s?W` | 25.5W, 12.9 W |
| POE Class | `Class\s?[0-4]` | Class 4 |
| IP Rating | `IP[0-9]{2}` | IP66, IP67 |
| Brand | Hanwha, Axis, Bosch, etc. | Wisenet → Hanwha |

**Tiered Schema:**
- **Tier 1 (Document):** vendor, doc_type, source_file
- **Tier 2 (Engineering):** model_num, poe_wattage, poe_class, brand
- **Tier 3 (Visual):** image_refs, page_num, chunk_index

---

## 🛠️ Configuration

Edit `config/settings.py`:

```python
class Settings:
    # Chunk settings
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150

    # Models
    EMBEDDING_MODEL = "nomic-embed-text"
    CHAT_MODEL = "llama3.1:8b"
    TEMPERATURE = 0.2

    # Retrieval
    TOP_K = 5
```

---

## 📁 Project Structure

```
platonicam_guru/
├── app/
│   ├── streamlit_app.py          # Main chat UI
│   └── pages/
│       ├── 1_📁_Ingestion.py     # PDF upload
│       └── 2_🔍_Database.py      # Search & browse
├── src/
│   ├── parser/
│   │   ├── pdf_parser.py         # PyMuPDF extraction
│   │   ├── metadata_extractor.py # Regex patterns
│   │   └── batch_processor.py    # Bulk processing
│   ├── embeddings/
│   │   └── ollama_embed.py       # Vector generation
│   ├── vectorstore/
│   │   └── chroma_store.py       # ChromaDB operations
│   ├── rag/
│   │   ├── chain.py              # Main orchestrator
│   │   ├── retriever.py          # Context retrieval
│   │   ├── prompts.py            # System prompts
│   │   ├── memory.py             # Conversation state
│   │   └── llm_client.py         # Ollama interface
│   ├── ingest.py                 # Ingestion CLI
│   ├── search.py                 # Search CLI
│   └── chat.py                   # Chat CLI
├── config/
│   └── settings.py               # Configuration
├── tests/                        # pytest suite
├── docs/                         # Setup guides (HTML)
├── data/pdfs/                    # Your PDFs (gitignored)
├── assets/images/                # Extracted images (gitignored)
├── chroma_db/                    # Vector store (gitignored)
└── requirements.txt
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src

# Specific module
pytest tests/test_metadata_extractor.py -v

# Tests requiring Ollama (skipped if unavailable)
pytest tests/test_ollama_embed.py -v
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://langchain.com/) - Text splitting
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF parsing
- [Streamlit](https://streamlit.io/) - Web UI framework

---

## 📬 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/platonicam_guru/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/platonicam_guru/discussions)

---

<p align="center">
  Built for Physical Security Engineers who need answers, not guesses.
</p>
