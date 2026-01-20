"""
Batch processor for ingesting PDF library.

Iterates through vendor PDF folders, extracts text/images, enriches with metadata,
and outputs chunks ready for vector embedding.
"""
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import Settings
from src.parser.pdf_parser import PDFParser
from src.parser.metadata_extractor import MetadataExtractor


class BatchProcessor:
    """
    Processes all PDFs in the data directory, organized by vendor.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or Settings.DATA_DIR
        self.output_dir = output_dir or Settings.PROJECT_ROOT / "processed"
        self.parser = PDFParser()
        self.extractor = MetadataExtractor()

        # Stats tracking
        self.stats = {
            "pdfs_processed": 0,
            "pdfs_failed": 0,
            "total_chunks": 0,
            "total_images": 0,
            "by_vendor": {},
        }

    def get_pdf_files(self, vendor: Optional[str] = None) -> list[tuple[Path, str]]:
        """
        Get all PDF files from data directory.

        Args:
            vendor: Optional vendor filter (hanwha, axis, bosch).

        Returns:
            List of (pdf_path, vendor_name) tuples.
        """
        pdf_files = []

        vendors = [vendor] if vendor else Settings.VENDORS

        for v in vendors:
            vendor_dir = self.data_dir / v
            if vendor_dir.exists():
                for pdf_path in vendor_dir.glob("*.pdf"):
                    pdf_files.append((pdf_path, v))

        # Also check root data dir for unorganized PDFs
        for pdf_path in self.data_dir.glob("*.pdf"):
            pdf_files.append((pdf_path, "unknown"))

        return pdf_files

    def process_single(self, pdf_path: Path, vendor: str) -> list[dict]:
        """
        Process a single PDF file.

        Returns:
            List of enriched chunk dicts.
        """
        # Parse PDF (extract text, images, chunk)
        chunks = self.parser.process_pdf(pdf_path, vendor=vendor)

        # Enrich each chunk with extracted metadata
        enriched_chunks = []
        for chunk in chunks:
            enriched = self.extractor.enrich_chunk(chunk)
            enriched_chunks.append(enriched)

        return enriched_chunks

    def process_all(
        self,
        vendor: Optional[str] = None,
        save_output: bool = True,
    ) -> list[dict]:
        """
        Process all PDFs in the library.

        Args:
            vendor: Optional vendor filter.
            save_output: Whether to save results to JSON.

        Returns:
            List of all enriched chunks.
        """
        pdf_files = self.get_pdf_files(vendor)
        all_chunks = []

        print(f"Found {len(pdf_files)} PDF files to process")
        print("-" * 50)

        for pdf_path, vendor_name in pdf_files:
            print(f"Processing: {pdf_path.name} ({vendor_name})")

            try:
                chunks = self.process_single(pdf_path, vendor_name)
                all_chunks.extend(chunks)

                # Update stats
                self.stats["pdfs_processed"] += 1
                self.stats["total_chunks"] += len(chunks)

                # Count images
                image_count = sum(
                    len(c.get("metadata", {}).get("image_refs", []))
                    for c in chunks
                )
                self.stats["total_images"] += image_count

                # Track by vendor
                if vendor_name not in self.stats["by_vendor"]:
                    self.stats["by_vendor"][vendor_name] = {"pdfs": 0, "chunks": 0}
                self.stats["by_vendor"][vendor_name]["pdfs"] += 1
                self.stats["by_vendor"][vendor_name]["chunks"] += len(chunks)

                print(f"  -> {len(chunks)} chunks, {image_count} images")

            except Exception as e:
                print(f"  ERROR: {e}")
                self.stats["pdfs_failed"] += 1
                continue

        print("-" * 50)
        self._print_stats()

        if save_output and all_chunks:
            self._save_output(all_chunks)

        return all_chunks

    def _print_stats(self):
        """Print processing statistics."""
        print("\nProcessing Complete!")
        print(f"  PDFs processed: {self.stats['pdfs_processed']}")
        print(f"  PDFs failed: {self.stats['pdfs_failed']}")
        print(f"  Total chunks: {self.stats['total_chunks']}")
        print(f"  Total images: {self.stats['total_images']}")

        if self.stats["by_vendor"]:
            print("\n  By vendor:")
            for vendor, data in self.stats["by_vendor"].items():
                print(f"    {vendor}: {data['pdfs']} PDFs, {data['chunks']} chunks")

    def _save_output(self, chunks: list[dict]):
        """Save processed chunks to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "processed_chunks.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"\n  Output saved to: {output_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDF library for vector embedding"
    )
    parser.add_argument(
        "--vendor",
        choices=Settings.VENDORS + ["all"],
        default="all",
        help="Process specific vendor or all",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output to JSON",
    )

    args = parser.parse_args()

    vendor = None if args.vendor == "all" else args.vendor

    processor = BatchProcessor()
    processor.process_all(vendor=vendor, save_output=not args.no_save)


if __name__ == "__main__":
    main()
