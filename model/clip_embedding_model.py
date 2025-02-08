import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

class GramMatrixModel:
    def __init__(self, model_name="vgg16", output_dim=1024):
        """
        VGG 모델을 로드하고, 특정 레이어에서 Feature Map을 추출하는 클래스.
        """
        self.model = self._load_vgg_model(model_name)
        self.model.eval()
        self.preprocess = self._get_preprocessing_transform()
        self.output_dim = output_dim  # 최종 벡터 차원 (예: 1024)

    def _load_vgg_model(self, model_name):
        """
        VGG 모델 로드 및 Feature Extractor로 변환 (깊은 계층만 선택)
        """
        if model_name == "vgg16":
            vgg = models.vgg16(pretrained=True).features
        else:
            raise ValueError("지원되지 않는 모델입니다. 현재 vgg16만 지원됨.")

        selected_layers = ['22']  # ✅ 깊은 계층(rel4_3)만 사용
        extracted_layers = {name: layer for name, layer in vgg._modules.items() if name in selected_layers}
        return nn.Sequential(extracted_layers)

    def _get_preprocessing_transform(self):
        """
        이미지를 VGG 입력 형식에 맞게 전처리하는 함수
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _compute_gram_matrix(self, feature_map):
        """
        Feature Map에서 Gram Matrix 계산
        """
        b, c, h, w = feature_map.size()
        features = feature_map.view(c, h * w)  # (C, H*W)
        gram = torch.mm(features, features.t())  # (C, C) Gram Matrix 생성
        return gram / (c * h * w)  # 정규화

    def get_gram_matrix_vector(self, image_path: str) -> np.ndarray:
        """
        Gram Matrix를 계산하고, PCA를 적용하여 1024차원 벡터로 변환
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            feature_maps = self.model(image_tensor)
        
        gram_matrices = [self._compute_gram_matrix(fmap) for fmap in feature_maps]

        # Gram Matrix를 벡터로 변환 (Flatten)
        gram_vectors = np.concatenate([g.flatten().cpu().numpy() for g in gram_matrices])

        # ✅ PCA 적용하여 1024 차원으로 압축
        gram_vectors = self._reduce_dimensionality(gram_vectors)
        
        return gram_vectors  # 최종 1024차원 벡터 반환

    def _reduce_dimensionality(self, gram_vector):
        """
        Gram Matrix 벡터를 PCA로 1024차원으로 압축
        """
        pca = PCA(n_components=self.output_dim)
        gram_vector = gram_vector.reshape(1, -1)  # (1, D) 형태로 변환
        reduced_vector = pca.fit_transform(gram_vector)
        return reduced_vector.flatten()  # 1D 벡터 반환

    def compare_images(self, image1_path: str, image2_path: str) -> float:
        """
        두 이미지의 Gram Matrix 기반 스타일 유사도 계산 (1024차원 벡터 비교)
        """
        gram1 = self.get_gram_matrix_vector(image1_path)
        gram2 = self.get_gram_matrix_vector(image2_path)

        # 두 벡터 간 L2 거리 계산
        similarity = np.linalg.norm(gram1 - gram2)
        return similarity  # 값이 작을수록 스타일이 비슷함


# # 📌 사용 예시
# extractor = GramMatrixModel(output_dim=1024)

# # 1024차원 스타일 벡터 추출
# style_vector = extractor.get_gram_matrix_vector("cloth1.jpg")
# print(f"1024차원 스타일 벡터 크기: {style_vector.shape}")

# # 두 이미지 스타일 비교
# similarity = extractor.compare_images("cloth1.jpg", "cloth2.jpg")
# print(f"스타일 유사도 점수: {similarity}")  # 값이 작을수록 스타일이 유사함
