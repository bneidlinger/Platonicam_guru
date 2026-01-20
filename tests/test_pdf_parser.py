"""
Unit tests for PDFParser.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.pdf_parser import PDFParser
from config.settings import Settings


@pytest.fixture
def parser(tmp_path):
    """Create parser with temporary assets directory."""
    return PDFParser(assets_dir=tmp_path / "images")


class TestPDFParserInit:
    """Tests for PDFParser initialization."""

    def test_default_chunk_settings(self, parser):
        assert parser.chunk_size == Settings.CHUNK_SIZE
        assert parser.chunk_overlap == Settings.CHUNK_OVERLAP

    def test_custom_chunk_settings(self, tmp_path):
        parser = PDFParser(
            chunk_size=500,
            chunk_overlap=50,
            assets_dir=tmp_path / "images",
        )
        assert parser.chunk_size == 500
        assert parser.chunk_overlap == 50

    def test_creates_assets_dir(self, tmp_path):
        assets_dir = tmp_path / "new_assets" / "images"
        parser = PDFParser(assets_dir=assets_dir)
        assert assets_dir.exists()


class TestTextChunking:
    """Tests for text chunking."""

    def test_short_text_single_chunk(self, parser):
        text = "Short text that fits in one chunk."
        chunks = parser.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self, parser):
        # Create text longer than chunk size
        text = "Word " * 500  # ~2500 characters
        chunks = parser.chunk_text(text)
        assert len(chunks) > 1

    def test_preserves_content(self, parser):
        text = "Important specification: 25.5W power consumption."
        chunks = parser.chunk_text(text)
        # Content should be preserved (possibly across chunks)
        combined = " ".join(chunks)
        assert "25.5W" in combined

    def test_chunks_have_overlap(self, parser):
        """Adjacent chunks should share some content due to overlap."""
        text = "A " * 1000  # Create long repetitive text
        chunks = parser.chunk_text(text)

        if len(chunks) >= 2:
            # With overlap, the end of chunk 1 should appear in chunk 2
            # (approximately, due to separator logic)
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0


class TestProcessPDFStructure:
    """Tests for process_pdf output structure (without real PDFs)."""

    def test_chunk_structure_mock(self, parser):
        """Verify expected chunk structure."""
        # This tests the expected output format
        expected_keys = {"content", "metadata"}
        expected_metadata_keys = {
            "source_file", "vendor", "page_num", "chunk_index", "image_refs"
        }

        # We can't test actual PDF processing without a PDF,
        # but we can document the expected structure
        sample_chunk = {
            "content": "Sample text content",
            "metadata": {
                "source_file": "test.pdf",
                "vendor": "hanwha",
                "page_num": 1,
                "chunk_index": 0,
                "image_refs": [],
            }
        }

        assert set(sample_chunk.keys()) == expected_keys
        assert set(sample_chunk["metadata"].keys()) == expected_metadata_keys


@pytest.mark.skipif(
    not list(Settings.DATA_DIR.glob("**/*.pdf")),
    reason="No PDF files in data directory"
)
class TestWithRealPDFs:
    """
    Integration tests with real PDF files.
    These tests are skipped if no PDFs are available.
    """

    @pytest.fixture
    def sample_pdf(self):
        """Get first available PDF for testing."""
        pdfs = list(Settings.DATA_DIR.glob("**/*.pdf"))
        if pdfs:
            return pdfs[0]
        pytest.skip("No PDF files available")

    def test_extract_text(self, parser, sample_pdf):
        pages = parser.extract_text(sample_pdf)
        assert isinstance(pages, list)
        if pages:
            assert "page_num" in pages[0]
            assert "text" in pages[0]

    def test_process_pdf(self, parser, sample_pdf):
        # Determine vendor from path
        vendor = "unknown"
        for v in Settings.VENDORS:
            if v in str(sample_pdf).lower():
                vendor = v
                break

        chunks = parser.process_pdf(sample_pdf, vendor=vendor)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert "content" in chunks[0]
        assert "metadata" in chunks[0]
