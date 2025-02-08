import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. 이미지 전처리 함수 (VGG 모델 입력 크기 맞추기)
def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 2. Feature Map에서 Gram Matrix 계산 함수
def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()  # (batch, channels, height, width)
    features = feature_map.view(c, h * w)  # (C, H*W) 형태로 변환
    gram = torch.mm(features, features.t())  # (C, C) Gram Matrix 생성
    return gram / (c * h * w)  # 정규화

# 3. VGG16 모델에서 특정 계층의 Feature Map 추출
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features  # VGG16의 Feature 추출 부분만 사용
        self.selected_layers = ['3', '8', '15', '22']  # Conv Layer 선택 (relu1_2, relu2_2, relu3_3, relu4_3)
        self.model = nn.Sequential(*list(vgg.children())[:23])  # Conv4_3까지 사용 (style 정보 추출에 적절)

    def forward(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features  # 선택한 레이어들의 feature map 반환

# 4. 주어진 이미지에서 Gram Matrix 추출
def extract_gram_matrices(image_path):
    model = VGGFeatureExtractor().eval()  # 모델 초기화 (추론 모드)
    image_tensor = preprocess_image(image_path)  # 이미지 전처리
    with torch.no_grad():  # 그래디언트 계산 비활성화
        feature_maps = model(image_tensor)
    
    gram_matrices = [gram_matrix(fmap) for fmap in feature_maps]  # 각 계층의 Gram Matrix 계산
    return gram_matrices

# 5. Gram Matrix 기반 유사도 계산 (두 이미지 비교)
def gram_similarity(gram1, gram2):
    similarity = []
    for g1, g2 in zip(gram1, gram2):
        sim = torch.norm(g1 - g2, p='fro')  # Frobenius norm 사용 (행렬 차이 측정)
        similarity.append(sim.item())
    return np.mean(similarity)  # 평균 유사도 반환 (값이 작을수록 스타일이 비슷함)

# 6. 두 이미지 비교 예제
image1_path = "cloth1.jpg"
image2_path = "cloth2.jpg"

gram1 = extract_gram_matrices(image1_path)
gram2 = extract_gram_matrices(image2_path)

similarity_score = gram_similarity(gram1, gram2)
print(f"스타일 유사도 점수: {similarity_score}")  # 낮을수록 스타일이 비슷함
