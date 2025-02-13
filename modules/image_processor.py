import os
from PIL import Image
import pytesseract
from typing import List, Dict

def process_images(image_paths: List[str]) -> List[Dict[str, str]]:
    """
    Process a list of image paths and return their descriptions and paths.
    
    Args:
        image_paths (List[str]): List of paths to image files
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing image descriptions and paths
    """
    processed_images = []
    
    for path in image_paths:
        if not os.path.exists(path):
            continue
            
        try:
            # Open and analyze the image
            with Image.open(path) as img:
                # Extract text from image using OCR (if any)
                text = pytesseract.image_to_string(img).strip()
                
                # Get basic image information
                width, height = img.size
                format_type = img.format
                
                # Create image description
                description = f"Image ({format_type}, {width}x{height}px)"
                if text:
                    description += f" - Contains text: {text[:100]}..."
                
                processed_images.append({
                    "path": path,
                    "description": description
                })
        except Exception as e:
            print(f"Error processing image {path}: {str(e)}")
            continue
    
    return processed_images
