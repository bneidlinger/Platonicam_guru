"""
Unit tests for MetadataExtractor.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.metadata_extractor import MetadataExtractor


@pytest.fixture
def extractor():
    return MetadataExtractor()


class TestModelNumberExtraction:
    """Tests for model number pattern matching."""

    def test_hanwha_model_xnv(self, extractor):
        text = "The XNV-8080R is an outdoor vandal dome."
        result = extractor.extract_model_numbers(text)
        assert "XNV-8080R" in result

    def test_hanwha_model_xnp(self, extractor):
        text = "XNP-6400RW PTZ camera specifications"
        result = extractor.extract_model_numbers(text)
        assert "XNP-6400RW" in result

    def test_axis_model(self, extractor):
        text = "The Axis P3265-LVE features advanced analytics."
        result = extractor.extract_model_numbers(text)
        assert "P3265-LVE" in result

    def test_bosch_model(self, extractor):
        text = "Bosch NBE-3502-AL network bullet camera"
        result = extractor.extract_model_numbers(text)
        assert "NBE-3502-AL" in result

    def test_multiple_models(self, extractor):
        text = "Compatible models: XNV-8080R, XNV-8082R, XNV-8081Z"
        result = extractor.extract_model_numbers(text)
        assert len(result) >= 3

    def test_no_model(self, extractor):
        text = "General camera specifications and features."
        result = extractor.extract_model_numbers(text)
        assert len(result) == 0


class TestPoEWattageExtraction:
    """Tests for PoE power consumption extraction."""

    def test_simple_wattage(self, extractor):
        text = "Max Power Consumption: 25.5W"
        result = extractor.extract_poe_wattage(text)
        assert result == 25.5

    def test_wattage_with_space(self, extractor):
        text = "Power: 12.9 W typical"
        result = extractor.extract_poe_wattage(text)
        assert result == 12.9

    def test_integer_wattage(self, extractor):
        text = "Consumption: 8W"
        result = extractor.extract_poe_wattage(text)
        assert result == 8.0

    def test_wattage_spelled_out(self, extractor):
        text = "Uses 15 Watts maximum"
        result = extractor.extract_poe_wattage(text)
        assert result == 15.0

    def test_excludes_wdr(self, extractor):
        """Should not match WDR (Wide Dynamic Range)."""
        text = "Features 120dB WDR for challenging lighting"
        result = extractor.extract_poe_wattage(text)
        assert result is None

    def test_multiple_returns_max(self, extractor):
        text = "Typical: 10W, Maximum: 25.5W"
        result = extractor.extract_poe_wattage(text)
        assert result == 25.5

    def test_no_wattage(self, extractor):
        text = "High resolution camera with advanced features"
        result = extractor.extract_poe_wattage(text)
        assert result is None


class TestPoEClassExtraction:
    """Tests for PoE class extraction."""

    def test_class_3(self, extractor):
        text = "PoE Class 3 compliant"
        result = extractor.extract_poe_class(text)
        assert result == "3"

    def test_class_4(self, extractor):
        text = "IEEE 802.3bt Class 4"
        result = extractor.extract_poe_class(text)
        assert result == "4"

    def test_class_without_poe(self, extractor):
        text = "Power Class 2 device"
        result = extractor.extract_poe_class(text)
        assert result == "2"

    def test_no_class(self, extractor):
        text = "Standard power supply required"
        result = extractor.extract_poe_class(text)
        assert result is None


class TestBrandExtraction:
    """Tests for brand name extraction."""

    def test_hanwha(self, extractor):
        text = "Hanwha Techwin camera system"
        result = extractor.extract_brand(text)
        assert result == "Hanwha"

    def test_wisenet_normalizes(self, extractor):
        """Wisenet should normalize to Hanwha."""
        text = "Wisenet X Series camera"
        result = extractor.extract_brand(text)
        assert result == "Hanwha"

    def test_axis(self, extractor):
        text = "Axis Communications network camera"
        result = extractor.extract_brand(text)
        assert result == "Axis"

    def test_bosch(self, extractor):
        text = "Bosch Security Systems"
        result = extractor.extract_brand(text)
        assert result == "Bosch"

    def test_no_brand(self, extractor):
        text = "Generic camera specifications"
        result = extractor.extract_brand(text)
        assert result is None


class TestResolutionExtraction:
    """Tests for resolution extraction."""

    def test_4k(self, extractor):
        text = "4K UHD resolution"
        result = extractor.extract_resolution(text)
        assert result == "4K"

    def test_megapixel(self, extractor):
        text = "5 Megapixel sensor"
        result = extractor.extract_resolution(text)
        assert result == "5MP"

    def test_mp_format(self, extractor):
        text = "2MP camera"
        result = extractor.extract_resolution(text)
        assert result == "2MP"


class TestIPRatingExtraction:
    """Tests for IP rating extraction."""

    def test_ip66(self, extractor):
        text = "Protection rating: IP66"
        result = extractor.extract_ip_rating(text)
        assert result == "IP66"

    def test_ip67(self, extractor):
        text = "Weatherproof IP67 rated"
        result = extractor.extract_ip_rating(text)
        assert result == "IP67"


class TestDocTypeClassification:
    """Tests for document type classification."""

    def test_datasheet(self, extractor):
        text = "Technical specifications and features for the camera system."
        result = extractor.classify_doc_type(text, "XNV-8080R_Datasheet.pdf")
        assert result == "datasheet"

    def test_installation(self, extractor):
        text = "Installation guide. Mounting instructions and setup procedures."
        result = extractor.classify_doc_type(text, "install_guide.pdf")
        assert result == "installation"

    def test_accessory(self, extractor):
        text = "Compatible mounting bracket and pendant accessories."
        result = extractor.classify_doc_type(text, "accessories.pdf")
        assert result == "accessory"


class TestExtractAll:
    """Tests for complete extraction."""

    def test_full_extraction(self, extractor):
        text = """
        XNV-8080R Specifications
        The Wisenet X series XNV-8080R is an outdoor vandal dome camera.
        Max Power Consumption: 25.5W (PoE++ Class 4)
        Resolution: 4K (8MP)
        Protection: IP66
        """
        result = extractor.extract_all(text, "XNV-8080R_Datasheet.pdf")

        assert result.get("model_num") == "XNV-8080R"
        assert result.get("poe_wattage") == 25.5
        assert result.get("poe_class") == "4"
        assert result.get("brand") == "Hanwha"
        assert result.get("ip_rating") == "IP66"
        assert result.get("doc_type") == "datasheet"


class TestEnrichChunk:
    """Tests for chunk enrichment."""

    def test_enrich_preserves_existing(self, extractor):
        chunk = {
            "content": "XNV-8080R with 25.5W power consumption",
            "metadata": {
                "source_file": "test.pdf",
                "vendor": "hanwha",
                "page_num": 1,
            }
        }

        result = extractor.enrich_chunk(chunk)

        # Existing metadata preserved
        assert result["metadata"]["source_file"] == "test.pdf"
        assert result["metadata"]["vendor"] == "hanwha"
        assert result["metadata"]["page_num"] == 1

        # New metadata added
        assert result["metadata"]["model_num"] == "XNV-8080R"
        assert result["metadata"]["poe_wattage"] == 25.5
