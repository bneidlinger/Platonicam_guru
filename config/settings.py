"""
Configuration settings for the Surveillance Design Assistant.
"""
from pathlib import Path


class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "pdfs"
    ASSETS_DIR = PROJECT_ROOT / "assets" / "images"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"

    # Vendor directories
    VENDORS = ["hanwha", "axis", "bosch"]

    # Text splitting configuration
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150

    # Embedding configuration
    EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_DIMENSION = 768
    EMBED_BATCH_SIZE = 32

    # LLM configuration
    CHAT_MODEL = "llama3.1:8b"
    TEMPERATURE = 0.2
    TOP_K = 5
    # Context window for chat calls. Ollama's default (2048-4096) silently
    # truncates RAG prompts; llama3.1 supports far more.
    NUM_CTX = 8192

    # Ollama API
    OLLAMA_HOST = "http://localhost:11434"

    # Image extraction
    IMAGE_FORMATS = ["png", "jpeg", "jpg"]

    # Document types for classification
    DOC_TYPES = ["datasheet", "installation", "accessory", "manual", "guide"]
