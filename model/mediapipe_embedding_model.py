import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
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

    def resize_with_padding(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """
        이미지를 비율을 유지하며 리사이즈하고 패딩을 추가하는 메서드
        
        :param image: PIL Image 객체
        :param target_size: (width, height) 목표 크기
        :return: 리사이즈된 PIL Image 객체
        """
        original_width, original_height = image.size
        target_width, target_height = target_size

        # 원본 이미지 비율 유지
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 이미지 Resize
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 패딩 추가
        padded_image = ImageOps.pad(resized_image, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
        return padded_image

    def get_image_resize(self, image_url: str, resize: tuple = (224, 224)) -> Image.Image:
        """
        URL에서 이미지를 다운로드하고 리사이즈하는 메서드
        
        :param image_url: 이미지 URL
        :param resize: (width, height) 리사이즈 크기
        :return: 처리된 PIL Image 객체
        """
        response = self.session.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        resized_image = self.resize_with_padding(image, resize)
        
        final_buffer = BytesIO()
        resized_image.save(final_buffer, format='JPEG')
        return resized_image

    def get_image_embedding(self, image_url: str, resize: tuple = None) -> np.ndarray:
        """
        이미지 URL에서 임베딩을 생성하는 메서드
        
        :param image_url: 이미지 URL
        :param resize: (width, height) 리사이즈 크기
        :return: 이미지 임베딩 벡터
        """
        pil_image = self.get_image_resize(image_url, resize)
        
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
