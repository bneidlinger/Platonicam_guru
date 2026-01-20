"""
PDF Parser module for extracting text and images from vendor datasheets.
"""
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class PDFParser:
    """
    Extracts text and images from PDF files, chunks text for vector storage.
    """

    def __init__(
        self,
        chunk_size: int = Settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.CHUNK_OVERLAP,
        assets_dir: Optional[Path] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.assets_dir = assets_dir or Settings.ASSETS_DIR
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Ensure assets directory exists
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def extract_text(self, pdf_path: Path) -> list[dict]:
        """
        Extract text from all pages of a PDF.

        Returns:
            List of dicts with 'page_num' and 'text' keys.
        """
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({
                    "page_num": page_num + 1,  # 1-indexed for display
                    "text": text,
                })

        doc.close()
        return pages

    def extract_images(self, pdf_path: Path, vendor: str = "") -> list[dict]:
        """
        Extract images from a PDF and save to assets directory.

        Args:
            pdf_path: Path to PDF file.
            vendor: Vendor name for organizing images.

        Returns:
            List of dicts with 'page_num', 'image_index', and 'image_path' keys.
        """
        doc = fitz.open(pdf_path)
        extracted_images = []
        filename_stem = pdf_path.stem

        # Create vendor subdirectory if specified
        if vendor:
            image_dir = self.assets_dir / vendor
        else:
            image_dir = self.assets_dir
        image_dir.mkdir(parents=True, exist_ok=True)

        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Skip unsupported formats
                    if image_ext.lower() not in Settings.IMAGE_FORMATS:
                        continue

                    # Generate filename: {pdf_name}_p{page}_i{index}.{ext}
                    image_name = f"{filename_stem}_p{page_num}_i{img_index}.{image_ext}"
                    image_path = image_dir / image_name

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    extracted_images.append({
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "image_path": str(image_path),
                    })

                except Exception as e:
                    # Skip problematic images
                    print(f"Warning: Could not extract image {img_index} from page {page_num}: {e}")
                    continue

        doc.close()
        return extracted_images

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks for vector storage.

        Args:
            text: Raw text to split.

        Returns:
            List of text chunks.
        """
        return self.text_splitter.split_text(text)

    def process_pdf(
        self,
        pdf_path: Path,
        vendor: str = "",
        extract_images: bool = True,
    ) -> list[dict]:
        """
        Full processing pipeline for a single PDF.

        Args:
            pdf_path: Path to PDF file.
            vendor: Vendor name (hanwha, axis, bosch).
            extract_images: Whether to extract images.

        Returns:
            List of chunk dicts ready for embedding, with structure:
            {
                "content": str,
                "metadata": {
                    "source_file": str,
                    "vendor": str,
                    "page_num": int,
                    "chunk_index": int,
                    "image_refs": list[str],
                }
            }
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name

        # Extract pages
        pages = self.extract_text(pdf_path)

        # Extract images and create page -> images mapping
        page_images = {}
        if extract_images:
            images = self.extract_images(pdf_path, vendor)
            for img in images:
                page = img["page_num"]
                if page not in page_images:
                    page_images[page] = []
                page_images[page].append(img["image_path"])

        # Process each page
        chunks = []
        global_chunk_index = 0

        for page_data in pages:
            page_num = page_data["page_num"]
            page_text = page_data["text"]

            # Get image refs for this page
            image_refs = page_images.get(page_num, [])

            # Chunk the page text
            page_chunks = self.chunk_text(page_text)

            for chunk_text in page_chunks:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source_file": filename,
                        "vendor": vendor,
                        "page_num": page_num,
                        "chunk_index": global_chunk_index,
                        "image_refs": image_refs,
                    },
                })
                global_chunk_index += 1

        return chunks


if __name__ == "__main__":
    # Quick test
    parser = PDFParser()
    print(f"PDFParser initialized")
    print(f"  Chunk size: {parser.chunk_size}")
    print(f"  Chunk overlap: {parser.chunk_overlap}")
    print(f"  Assets dir: {parser.assets_dir}")
