import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import cv2
import requests
from io import BytesIO

class MediaPipeEmbeddingModel:
    def __init__(self, model_name="embedder.tflite"):

        base_options = python.BaseOptions(model_asset_path="./model/" + model_name)
        options = vision.ImageEmbedderOptions(
            base_options=base_options,
            l2_normalize=True
        )
        self.embedder = vision.ImageEmbedder.create_from_options(options)

    def get_image_embedding(self, image_path: str, mode : str = "image_url") -> np.ndarray:
        if mode == "image_path":
            mp_image = mp.Image.create_from_file(image_path)
        elif mode == "image_url":
            response = requests.get(image_path)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {image_path}")
            
            # BytesIO로 이미지 변환 후 PIL 이미지로 열기
            image_data = BytesIO(response.content)
            pil_image = Image.open(image_data).convert("RGB")
            
            # PIL 이미지를 MediaPipe 이미지로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_image))
            
        embedding_result = self.embedder.embed(mp_image)
        embedding = embedding_result.embeddings[0].embedding
        return embedding

    def embed_batch(self, image_urls: dict) -> dict:
        """
        여러 이미지의 임베딩을 한번에 처리하는 메서드
        
        :param image_urls: Dict[product_id, image_url]
        :return: Dict[product_id, embedding]
        """
        embeddings = {}
        for i, (product_id, image_url) in enumerate(image_urls.items()):
            try:
                # print("this time:", i)
                embedding = self.get_image_embedding(image_url, mode="image_url")
                embeddings[product_id] = embedding
            except Exception as e:
                print(f"Error processing image for product {product_id}: {str(e)}")
                continue
        
        return embeddings