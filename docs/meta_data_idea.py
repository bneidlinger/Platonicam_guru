import os
import re
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SurveillanceMetadataParser:
    def __init__(self, input_dir, asset_dir="./assets/images"):
        self.input_dir = input_dir
        self.asset_dir = asset_dir
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        
        # Regex patterns for common surveillance data
        self.patterns = {
            "model_num": r"([A-Z]{1,4}-[A-Z0-9]{4,10})", # Matches XNP-6400R, P3265, etc.
            "poe_wattage": r"(\d{1,2}\.?\d?)\s?W(?!\s?DR)", # Matches 12.9W, 25W
            "poe_class": r"Class\s?([0-4])",               # Matches Class 3, Class 4
            "brand": r"(Hanwha|Axis|Bosch|Honeywell|Avigilon)"
        }

        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir)

    def extract_metadata(self, text):
        """Attempts to find structured data within a text chunk."""
        metadata = {}
        for key, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).upper()
        return metadata

    def process_library(self):
        processed_chunks = []
        for filename in os.listdir(self.input_dir):
            if not filename.endswith(".pdf"): continue
            
            doc = fitz.open(os.path.join(self.input_dir, filename))
            
            for page_index, page in enumerate(doc):
                text = page.get_text()
                
                # Extract image metadata (linking physical assets)
                image_list = page.get_images()
                img_refs = []
                for img_idx, img in enumerate(image_list):
                    # (Simplified image save logic from previous step)
                    img_refs.append(f"{filename}_p{page_index}_i{img_idx}.png")

                # Chunk and Tag
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    # Merge regex-found metadata with file metadata
                    extracted = self.extract_metadata(chunk)
                    meta = {
                        "source": filename,
                        "page": page_index + 1,
                        "images": img_refs,
                        **extracted # Injects 'model_num', 'poe_wattage', etc.
                    }
                    processed_chunks.append({"content": chunk, "metadata": meta})
                    
        return processed_chunks

# Example execution
# parser = SurveillanceMetadataParser("./manuals")
# data = parser.process_library()