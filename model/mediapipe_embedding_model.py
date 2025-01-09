import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import cv2

class MediaPipeEmbeddingModel:
    def __init__(self, model_name="embedder.tflite"):

        base_options = python.BaseOptions(model_asset_path=model_name)
        options = vision.ImageEmbedderOptions(
            base_options=base_options,
            l2_normalize=True
        )
        self.embedder = vision.ImageEmbedder.create_from_options(options)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        
        mp_image = mp.Image.create_from_file(image_path)
        embedding_result = self.embedder.embed(mp_image)
        embedding = embedding_result.embeddings[0].embedding
        return embedding