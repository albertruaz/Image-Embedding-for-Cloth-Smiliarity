from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
class CLIPEmbeddingModel:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        outputs = outputs.detach().numpy().squeeze()
        return outputs

    def get_text_embedding(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.model.get_text_features(**inputs)
        return outputs 