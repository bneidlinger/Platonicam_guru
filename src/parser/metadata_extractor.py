"""
Metadata Extractor module for extracting structured data from text using regex patterns.

Extracts surveillance equipment specific fields:
- Model numbers (XNV-8080R, P3265-LVE, etc.)
- PoE wattage values
- PoE class (0-4)
- Brand names
- Document type classification
"""
import re
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings


class MetadataExtractor:
    """
    Extracts structured metadata from text chunks using regex patterns.

    Design Principle: These extracted values are used for computation and filtering,
    not for LLM generation. POE budgets sum these values directly.
    """

    # Regex patterns for surveillance equipment data
    PATTERNS = {
        # Model numbers: XNV-8080R, XNP-6400RW, P3265-LVE, NBE-3502, etc.
        "model_num": r"\b([A-Z]{1,4}[-]?[A-Z0-9]{3,10}(?:[-][A-Z0-9]+)?)\b",

        # PoE wattage: 12.9W, 25.5 W, 8W (excludes WDR false positives)
        "poe_wattage": r"(\d{1,2}\.?\d?)\s?W(?:atts?)?(?!\s*D)",

        # PoE class: Class 3, Class 4, PoE Class 2
        "poe_class": r"(?:PoE\s*)?Class\s*([0-8])",

        # Brand names
        "brand": r"\b(Hanwha|Wisenet|Axis|Bosch|Honeywell|Avigilon|Hikvision|Dahua)\b",

        # Resolution: 4K, 5MP, 2MP, 8 Megapixel
        "resolution": r"\b(\d+)\s*(?:MP|Megapixel|K)\b",

        # IP rating: IP66, IP67
        "ip_rating": r"\b(IP[0-9]{2})\b",

        # Operating temperature: -40°C to +60°C, -30° ~ 60°
        "temp_range": r"(-?\d+)\s*°?\s*[C°]?\s*(?:to|~|-)\s*\+?(-?\d+)\s*°?\s*C?",
    }

    # Keywords for document type classification
    DOC_TYPE_KEYWORDS = {
        "datasheet": ["specifications", "features", "datasheet", "spec sheet", "technical data"],
        "installation": ["installation", "mounting", "setup", "quick start", "install guide"],
        "accessory": ["accessory", "mount", "bracket", "pendant", "corner", "pole"],
        "manual": ["user manual", "operation manual", "user guide", "operating instructions"],
        "guide": ["configuration", "network setup", "admin guide", "programming"],
    }

    def __init__(self):
        # Pre-compile patterns for performance
        self._compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in self.PATTERNS.items()
        }

    def extract_model_numbers(self, text: str) -> list[str]:
        """
        Extract all model numbers from text.

        Returns deduplicated list of model numbers found.
        """
        matches = self._compiled_patterns["model_num"].findall(text)
        # Deduplicate while preserving order, uppercase for consistency
        seen = set()
        result = []
        for m in matches:
            upper = m.upper()
            if upper not in seen and len(upper) >= 4:  # Filter out short false positives
                seen.add(upper)
                result.append(upper)
        return result

    def extract_poe_wattage(self, text: str) -> Optional[float]:
        """
        Extract PoE power consumption value.

        Returns the first (typically max) wattage found, or None.
        """
        matches = self._compiled_patterns["poe_wattage"].findall(text)
        if matches:
            try:
                # Return highest value (usually max power consumption)
                return max(float(m) for m in matches)
            except ValueError:
                return None
        return None

    def extract_poe_class(self, text: str) -> Optional[str]:
        """
        Extract PoE class (0-8).

        Returns class as string, or None.
        """
        match = self._compiled_patterns["poe_class"].search(text)
        if match:
            return match.group(1)
        return None

    def extract_brand(self, text: str) -> Optional[str]:
        """
        Extract brand name from text.

        Returns standardized brand name, or None.
        """
        match = self._compiled_patterns["brand"].search(text)
        if match:
            brand = match.group(1).title()
            # Normalize Wisenet -> Hanwha
            if brand.lower() == "wisenet":
                return "Hanwha"
            return brand
        return None

    def extract_resolution(self, text: str) -> Optional[str]:
        """
        Extract camera resolution.

        Returns resolution string (e.g., "4K", "5MP"), or None.
        """
        match = self._compiled_patterns["resolution"].search(text)
        if match:
            value = match.group(1)
            # Determine format based on original match
            if "K" in text[match.start():match.end()].upper():
                return f"{value}K"
            return f"{value}MP"
        return None

    def extract_ip_rating(self, text: str) -> Optional[str]:
        """
        Extract IP protection rating.
        """
        match = self._compiled_patterns["ip_rating"].search(text)
        if match:
            return match.group(1).upper()
        return None

    def classify_doc_type(self, text: str, filename: str = "") -> str:
        """
        Classify document type based on content and filename.

        Returns one of: datasheet, installation, accessory, manual, guide, unknown
        """
        search_text = (text + " " + filename).lower()

        # Score each doc type by keyword matches
        scores = {}
        for doc_type, keywords in self.DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in search_text)
            if score > 0:
                scores[doc_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def extract_all(self, text: str, filename: str = "") -> dict:
        """
        Extract all metadata fields from a text chunk.

        Args:
            text: The text content to analyze.
            filename: Original filename for doc type classification.

        Returns:
            Dict with all extracted metadata fields.
            Only includes fields where values were found.
        """
        metadata = {}

        # Model numbers (can have multiple)
        model_nums = self.extract_model_numbers(text)
        if model_nums:
            metadata["model_num"] = model_nums[0]  # Primary model
            if len(model_nums) > 1:
                metadata["related_models"] = model_nums[1:]

        # PoE wattage (numeric for computation)
        poe_wattage = self.extract_poe_wattage(text)
        if poe_wattage is not None:
            metadata["poe_wattage"] = poe_wattage

        # PoE class
        poe_class = self.extract_poe_class(text)
        if poe_class:
            metadata["poe_class"] = poe_class

        # Brand
        brand = self.extract_brand(text)
        if brand:
            metadata["brand"] = brand

        # Resolution
        resolution = self.extract_resolution(text)
        if resolution:
            metadata["resolution"] = resolution

        # IP rating
        ip_rating = self.extract_ip_rating(text)
        if ip_rating:
            metadata["ip_rating"] = ip_rating

        # Document type
        doc_type = self.classify_doc_type(text, filename)
        if doc_type != "unknown":
            metadata["doc_type"] = doc_type

        return metadata

    def enrich_chunk(self, chunk: dict) -> dict:
        """
        Enrich a chunk dict with extracted metadata.

        Args:
            chunk: Dict with 'content' and 'metadata' keys.

        Returns:
            Same chunk with additional metadata fields merged in.
        """
        content = chunk.get("content", "")
        filename = chunk.get("metadata", {}).get("source_file", "")

        extracted = self.extract_all(content, filename)

        # Merge extracted metadata with existing metadata
        chunk["metadata"] = {**chunk.get("metadata", {}), **extracted}

        return chunk


if __name__ == "__main__":
    # Test extraction with sample text
    extractor = MetadataExtractor()

    sample_text = """
    XNV-8080R Specifications
    The Wisenet X series XNV-8080R is an outdoor vandal dome camera.
    Max Power Consumption: 25.5W (PoE++ Class 4)
    Operating Temperature: -50°C to +55°C
    Protection: IP66, IK10
    Resolution: 4K (8MP)
    """

    print("Sample text extraction:")
    print("-" * 40)
    result = extractor.extract_all(sample_text, "XNV-8080R_Datasheet.pdf")
    for key, value in result.items():
        print(f"  {key}: {value}")
