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
        self.session = requests.Session()

    def get_image_embedding(
        self, 
        image_url: str,
        resize: tuple = None
    ) -> np.ndarray:
        
        resized_image_url = f"{image_url}?width={resize[0]}&height={resize[1]}"
        response = self.session.get(resized_image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {image_url}")
        
        image_data = BytesIO(response.content)
        pil_image = Image.open(image_data).convert("RGB")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=np.array(pil_image)
        )

        embedding_result = self.embedder.embed(mp_image)
        embedding = embedding_result.embeddings[0].embedding
        return embedding

    def embed_batch(self, product_datas: list, resize: tuple = (224, 224)) -> list:
        """
        여러 이미지의 임베딩을 한 번에 처리하는 메서드
        
        :param product_datas: List[(product_id, image_url), ...] 형태의 튜플 리스트
        :param resize: (width, height)를 지정하면 모든 이미지를 해당 크기로 리사이즈 후 임베딩
        :return: Dict[product_id, embedding]
        """
        embeddings = []
        for product_id, image_url in product_datas:
            try:
                embedding = self.get_image_embedding(image_url,resize=resize)
                embeddings.append({
                    "product_id": product_id,
                    "embedding": embedding.tolist()  # NumPy 배열을 리스트로 변환
                })
            except Exception as e:
                print(f"Error processing image for product {product_id}: {str(e)}")
                continue
        
        return embeddings
