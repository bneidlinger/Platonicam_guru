import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VendorDataParser:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        # Chunk size 1000 with 200 overlap helps maintain context for accessory tables
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_text_from_pdf(self, file_path):
        """Extracts text and basic metadata from a PDF."""
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    def process_library(self):
        """Iterates through folder, parses PDFs, and creates chunks."""
        processed_data = []
        
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".pdf"):
                print(f"Processing: {filename}...")
                file_path = os.path.join(self.input_dir, filename)
                
                try:
                    raw_text = self.extract_text_from_pdf(file_path)
                    chunks = self.text_splitter.split_text(raw_text)
                    
                    for i, chunk in enumerate(chunks):
                        processed_data.append({
                            "source": filename,
                            "chunk_id": i,
                            "content": chunk
                        })
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
                    
        return processed_data

if __name__ == "__main__":
    # Update this to your local PDF folder path
    LIBRARY_PATH = "./camera_manuals"
    
    if not os.path.exists(LIBRARY_PATH):
        os.makedirs(LIBRARY_PATH)
        print(f"Created {LIBRARY_PATH} folder. Add your PDFs there.")
    else:
        parser = VendorDataParser(LIBRARY_PATH)
        data = parser.process_library()
        print(f"\nSuccessfully parsed {len(data)} total chunks for the Vector DB.")
        # Next step would be sending 'data' to ChromaDB via Ollama