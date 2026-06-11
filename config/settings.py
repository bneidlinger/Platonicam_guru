"""
Configuration settings for the Surveillance Design Assistant.

Deployment-relevant values can be overridden via environment variables
(or a .env file in the project root) so containerized installs can point
at an Ollama sidecar, set an API key, etc. without code changes.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_DIMENSION = 768
    EMBED_BATCH_SIZE = 32

    # LLM configuration
    CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
    TEMPERATURE = 0.2
    TOP_K = 5
    # Context window for chat calls. Ollama's default (2048-4096) silently
    # truncates RAG prompts; llama3.1 supports far more.
    NUM_CTX = 8192

    # Ollama API
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # API server
    # Empty key = open access for local development; admin endpoints stay
    # disabled until a key is configured.
    API_KEY = os.getenv("PLATONICAM_API_KEY", "")
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "60"))
    MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))

    # Image extraction
    IMAGE_FORMATS = ["png", "jpeg", "jpg"]

    # Document types for classification
    DOC_TYPES = ["datasheet", "installation", "accessory", "manual", "guide"]
