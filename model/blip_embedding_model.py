from transformers import BlipProcessor, BlipModel
from PIL import Image
import numpy as np
import torch

class BLIPEmbeddingModel:
    def __init__(self, model_name: str):
        # Load model and processor from the local or remote path
        self.model = BlipModel.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get embedding for a single image
        """
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embedding = outputs.numpy().squeeze()
        return embedding / np.linalg.norm(embedding)
