from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

# Load models
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embeddings(texts):
    """
    Generate embeddings for a list of text strings.
    """
    return text_model.encode(texts)

def get_image_embeddings(image_paths):
    """
    Generate embeddings for a list of image file paths.
    Skips invalid paths and returns an empty array if no valid images are found.
    """
    embeddings = []
    for path in image_paths:
        try:
            # Attempt to open the image file
            image = Image.open(path)
            inputs = clip_processor(images=image, return_tensors="pt")
            emb = clip_model.get_image_features(**inputs).detach().numpy()
            embeddings.append(emb)
        except Exception as e:
            # Skip invalid paths or non-image files
            print(f"Skipping invalid image path: {path}. Error: {e}")
    return np.vstack(embeddings) if embeddings else np.array([])