"""
Ingestion Pipeline - Full workflow from PDF to ChromaDB.

Combines:
- PDF parsing (text + image extraction)
- Metadata extraction (regex-based)
- Embedding generation (Ollama)
- Vector storage (ChromaDB)
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.parser.pdf_parser import PDFParser
from src.parser.metadata_extractor import MetadataExtractor
from src.embeddings.ollama_embed import OllamaEmbedder
from src.vectorstore.chroma_store import ChromaStore


class IngestionPipeline:
    """
    End-to-end pipeline for ingesting PDF documents into vector store.
    """

    def __init__(self):
        self.parser = PDFParser()
        self.extractor = MetadataExtractor()
        self.embedder = OllamaEmbedder()
        self.store = ChromaStore()

        # Stats tracking
        self.stats = {
            "pdfs_processed": 0,
            "pdfs_failed": 0,
            "chunks_created": 0,
            "chunks_embedded": 0,
            "chunks_stored": 0,
            "chunks_skipped": 0,
            "images_extracted": 0,
        }

    def check_prerequisites(self) -> bool:
        """
        Check that Ollama is running and model is available.
        """
        print("Checking prerequisites...")

        if not self.embedder.check_model_available():
            print(f"ERROR: Embedding model '{self.embedder.model}' not found!")
            print(f"Run: ollama pull {self.embedder.model}")
            return False

        print(f"  Embedding model: {self.embedder.model} (available)")
        print(f"  Vector store: {self.store.persist_dir}")
        print(f"  Current chunks in store: {self.store.count()}")

        return True

    def ingest_pdf(
        self,
        pdf_path: Path,
        vendor: str = "",
        skip_existing: bool = True,
    ) -> dict:
        """
        Ingest a single PDF file.

        Args:
            pdf_path: Path to PDF file.
            vendor: Vendor name for metadata.
            skip_existing: Skip chunks already in store.

        Returns:
            Dict with processing stats for this file.
        """
        pdf_path = Path(pdf_path)
        file_stats = {"chunks": 0, "images": 0, "added": 0, "skipped": 0}

        # Step 1: Parse PDF (extract text, images, chunk)
        chunks = self.parser.process_pdf(pdf_path, vendor=vendor)
        file_stats["chunks"] = len(chunks)

        # Count images
        for chunk in chunks:
            file_stats["images"] += len(chunk.get("metadata", {}).get("image_refs", []))

        # Step 2: Enrich with extracted metadata
        for chunk in chunks:
            self.extractor.enrich_chunk(chunk)

        # Step 3: Generate embeddings
        chunks = self.embedder.embed_chunks(chunks, show_progress=False)

        # Step 4: Store in ChromaDB
        result = self.store.add_chunks(chunks, skip_existing=skip_existing, show_progress=False)
        file_stats["added"] = result["added"]
        file_stats["skipped"] = result["skipped"]

        return file_stats

    def ingest_directory(
        self,
        data_dir: Optional[Path] = None,
        vendor: Optional[str] = None,
        skip_existing: bool = True,
    ) -> dict:
        """
        Ingest all PDFs from data directory.

        Args:
            data_dir: Directory containing PDFs (default: Settings.DATA_DIR).
            vendor: Filter by vendor (or process all).
            skip_existing: Skip chunks already in store.

        Returns:
            Processing statistics.
        """
        data_dir = data_dir or Settings.DATA_DIR

        # Collect PDF files
        pdf_files = []
        vendors_to_process = [vendor] if vendor else Settings.VENDORS

        for v in vendors_to_process:
            vendor_dir = data_dir / v
            if vendor_dir.exists():
                for pdf_path in vendor_dir.glob("*.pdf"):
                    pdf_files.append((pdf_path, v))

        # Also check root for unorganized PDFs
        if not vendor:
            for pdf_path in data_dir.glob("*.pdf"):
                pdf_files.append((pdf_path, "unknown"))

        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return self.stats

        print(f"\nFound {len(pdf_files)} PDF files to ingest")
        print("=" * 60)

        for pdf_path, vendor_name in pdf_files:
            print(f"\nProcessing: {pdf_path.name}")
            print(f"  Vendor: {vendor_name}")

            try:
                file_stats = self.ingest_pdf(pdf_path, vendor=vendor_name, skip_existing=skip_existing)

                self.stats["pdfs_processed"] += 1
                self.stats["chunks_created"] += file_stats["chunks"]
                self.stats["chunks_stored"] += file_stats["added"]
                self.stats["chunks_skipped"] += file_stats["skipped"]
                self.stats["images_extracted"] += file_stats["images"]

                print(f"  Chunks: {file_stats['chunks']} created, {file_stats['added']} stored, {file_stats['skipped']} skipped")
                print(f"  Images: {file_stats['images']} extracted")

            except Exception as e:
                print(f"  ERROR: {e}")
                self.stats["pdfs_failed"] += 1

        self._print_summary()
        return self.stats

    def _print_summary(self):
        """Print ingestion summary."""
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"  PDFs processed: {self.stats['pdfs_processed']}")
        print(f"  PDFs failed: {self.stats['pdfs_failed']}")
        print(f"  Chunks created: {self.stats['chunks_created']}")
        print(f"  Chunks stored: {self.stats['chunks_stored']}")
        print(f"  Chunks skipped (existing): {self.stats['chunks_skipped']}")
        print(f"  Images extracted: {self.stats['images_extracted']}")
        print(f"\n  Total chunks in store: {self.store.count()}")

        # Show breakdown by vendor
        store_stats = self.store.get_stats()
        if store_stats["by_vendor"]:
            print("\n  By vendor:")
            for vendor, count in store_stats["by_vendor"].items():
                print(f"    {vendor}: {count} chunks")


def main():
    """CLI entry point for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into vector store"
    )
    parser.add_argument(
        "--vendor",
        choices=Settings.VENDORS + ["all"],
        default="all",
        help="Process specific vendor or all",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if chunks exist",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingestion",
    )

    args = parser.parse_args()

    pipeline = IngestionPipeline()

    # Check prerequisites
    if not pipeline.check_prerequisites():
        sys.exit(1)

    # Clear if requested
    if args.clear:
        count = pipeline.store.clear()
        print(f"\nCleared {count} existing chunks")

    # Run ingestion
    vendor = None if args.vendor == "all" else args.vendor
    pipeline.ingest_directory(vendor=vendor, skip_existing=not args.force)


if __name__ == "__main__":
    main()
