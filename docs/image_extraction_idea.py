import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VendorDataParser:
    def __init__(self, input_dir, asset_dir="./assets/images"):
        self.input_dir = input_dir
        self.asset_dir = asset_dir
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        if not os.path.exists(self.asset_dir):
            os.makedirs(self.asset_dir)

    def process_pdf(self, filename):
        file_path = os.path.join(self.input_dir, filename)
        doc = fitz.open(file_path)
        page_data = []

        for page_index, page in enumerate(doc):
            # 1. Extract Text
            text = page.get_text()
            
            # 2. Extract Images
            image_list = page.get_images(full=True)
            image_paths = []
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Create a unique filename for the image
                img_name = f"{filename.split('.')[0]}_p{page_index}_i{img_index}.{base_image['ext']}"
                img_path = os.path.join(self.asset_dir, img_name)
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(img_path)

            # Split text for this specific page
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                page_data.append({
                    "content": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_index,
                        "image_refs": image_paths # Link extracted images to this text
                    }
                })
        return page_data

# Usage remains similar, but now 'metadata' contains image links.