"""
Integration tests for ChromaStore.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import ChromaStore


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary ChromaDB store for testing."""
    store = ChromaStore(
        persist_dir=tmp_path / "test_chroma",
        collection_name="test_collection",
    )
    yield store
    # Cleanup
    try:
        store.clear()
    except Exception:
        pass


@pytest.fixture
def sample_chunks():
    """Sample chunks with mock embeddings."""
    return [
        {
            "content": "XNV-8080R specifications. Max power 25.5W PoE++",
            "embedding": [0.1] * 768,  # Mock 768-dim embedding
            "metadata": {
                "source_file": "XNV-8080R_Datasheet.pdf",
                "vendor": "hanwha",
                "page_num": 1,
                "chunk_index": 0,
                "model_num": "XNV-8080R",
                "poe_wattage": 25.5,
                "poe_class": "4",
            },
        },
        {
            "content": "P3265-LVE outdoor dome camera. Power consumption 12.9W",
            "embedding": [0.2] * 768,
            "metadata": {
                "source_file": "P3265-LVE_Datasheet.pdf",
                "vendor": "axis",
                "page_num": 1,
                "chunk_index": 0,
                "model_num": "P3265-LVE",
                "poe_wattage": 12.9,
                "poe_class": "3",
            },
        },
        {
            "content": "Mounting bracket for XNV-8080R. Compatible with pendant mount.",
            "embedding": [0.15] * 768,
            "metadata": {
                "source_file": "XNV-8080R_Accessories.pdf",
                "vendor": "hanwha",
                "page_num": 3,
                "chunk_index": 0,
                "model_num": "XNV-8080R",
                "doc_type": "accessory",
            },
        },
    ]


class TestChromaStoreBasics:
    """Basic ChromaStore operations."""

    def test_init_creates_directory(self, tmp_path):
        persist_dir = tmp_path / "new_store"
        store = ChromaStore(persist_dir=persist_dir)
        # Access collection to trigger creation
        _ = store.collection
        assert persist_dir.exists()

    def test_empty_store_count(self, temp_store):
        assert temp_store.count() == 0

    def test_add_chunks(self, temp_store, sample_chunks):
        result = temp_store.add_chunks(sample_chunks, show_progress=False)
        assert result["added"] == 3
        assert result["skipped"] == 0
        assert temp_store.count() == 3

    def test_skip_existing_chunks(self, temp_store, sample_chunks):
        # Add chunks first time
        temp_store.add_chunks(sample_chunks, show_progress=False)

        # Add same chunks again
        result = temp_store.add_chunks(sample_chunks, skip_existing=True, show_progress=False)
        assert result["added"] == 0
        assert result["skipped"] == 3
        assert temp_store.count() == 3


class TestChromaStoreSearch:
    """Search functionality tests."""

    def test_basic_search(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        # Search with mock query embedding
        query_embedding = [0.1] * 768  # Similar to first chunk
        results = temp_store.search(query_embedding, n_results=2)

        assert len(results) == 2
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_search_by_vendor(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        query_embedding = [0.1] * 768
        results = temp_store.search_by_vendor(query_embedding, vendor="hanwha")

        assert len(results) >= 1
        for result in results:
            assert result["metadata"].get("vendor") == "hanwha"

    def test_search_by_model(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        query_embedding = [0.1] * 768
        results = temp_store.search_by_model(query_embedding, model_num="XNV-8080R")

        assert len(results) >= 1
        for result in results:
            assert result["metadata"].get("model_num") == "XNV-8080R"

    def test_search_accessories(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        query_embedding = [0.15] * 768
        results = temp_store.search_accessories(query_embedding)

        # Should find the accessory document
        assert len(results) >= 1
        assert any(r["metadata"].get("doc_type") == "accessory" for r in results)


class TestChromaStorePOE:
    """POE calculation tests - metadata for computation."""

    def test_get_poe_wattage(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        wattage = temp_store.get_poe_wattage("XNV-8080R")
        assert wattage == 25.5

    def test_get_poe_wattage_not_found(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        wattage = temp_store.get_poe_wattage("UNKNOWN-MODEL")
        assert wattage is None

    def test_calculate_poe_budget(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        budget = temp_store.calculate_poe_budget(["XNV-8080R", "P3265-LVE"])

        assert budget["total_watts"] == 25.5 + 12.9
        assert budget["by_model"]["XNV-8080R"] == 25.5
        assert budget["by_model"]["P3265-LVE"] == 12.9
        assert len(budget["missing"]) == 0

    def test_calculate_poe_budget_with_missing(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        budget = temp_store.calculate_poe_budget(["XNV-8080R", "UNKNOWN-MODEL"])

        assert budget["total_watts"] == 25.5
        assert "UNKNOWN-MODEL" in budget["missing"]


class TestChromaStoreStats:
    """Statistics and management tests."""

    def test_get_stats(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)

        stats = temp_store.get_stats()

        assert stats["total_chunks"] == 3
        assert "hanwha" in stats["by_vendor"]
        assert "axis" in stats["by_vendor"]
        assert stats["by_vendor"]["hanwha"] == 2
        assert stats["by_vendor"]["axis"] == 1

    def test_delete_by_source(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)
        assert temp_store.count() == 3

        deleted = temp_store.delete_by_source("XNV-8080R_Datasheet.pdf")

        assert deleted == 1
        assert temp_store.count() == 2

    def test_clear(self, temp_store, sample_chunks):
        temp_store.add_chunks(sample_chunks, show_progress=False)
        assert temp_store.count() == 3

        cleared = temp_store.clear()

        assert cleared == 3
        assert temp_store.count() == 0


class TestMetadataCleaning:
    """Test metadata cleaning for ChromaDB compatibility."""

    def test_list_metadata_converted(self, temp_store):
        chunks = [{
            "content": "Test content",
            "embedding": [0.1] * 768,
            "metadata": {
                "source_file": "test.pdf",
                "image_refs": ["/path/to/img1.png", "/path/to/img2.png"],
            },
        }]

        temp_store.add_chunks(chunks, show_progress=False)

        # Retrieve and check metadata
        results = temp_store.collection.get(include=["metadatas"])
        meta = results["metadatas"][0]

        # List should be converted to comma-separated string
        assert "image_refs" in meta
        assert "/path/to/img1.png" in meta["image_refs"]
        assert "," in meta["image_refs"]
