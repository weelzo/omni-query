import fitz  # PyMuPDF
import os
import re

def clean_text(text):
    """Clean extracted text by removing noise and formatting issues."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove isolated single characters (often noise from PDF extraction)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove very short lines (likely noise or headers/footers)
    if len(text.split()) < 3:
        return ''
    return text.strip()

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    image_data = []
    os.makedirs('assets', exist_ok=True)
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get text blocks with more structure
        blocks = page.get_text('blocks')
        
        for block in blocks:
            # block[4] contains the text content
            text = clean_text(block[4])
            if text:  # Only add non-empty blocks
                text_data.append({
                    'text': text,
                    'page': page_num + 1,  # Make pages 1-based for better UX
                    'bbox': block[:4]  # First 4 elements are coordinates
                })

        # Extract images
        images = page.get_images()
        for img in images:
            pix = fitz.Pixmap(doc, img[0])
            image_path = f"assets/page_{page_num}_image_{img[0]}.png"
            pix.save(image_path)
            image_data.append({"path": image_path, "page": page_num, "bbox": img[1:5]})
    return text_data, image_data