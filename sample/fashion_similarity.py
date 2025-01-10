from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 1. 모델과 프로세서 로드
model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. 이미지와 텍스트 준비
image = Image.open("data/2.png")  # 사용할 이미지 경로
colors_list = ["red","green","blue","black","gray"]  # 텍스트 리스트
types_list = ["아우터","상의","바지","원피스","패션소품","가방","스커트","셋업","신발"]
styles_list = ["캐주얼","스트릿","러블리","브랜드","비즈니즈캐주얼","아메카지","스포티","빈티지"]



# 3. 입력 데이터 전처리
colors = processor(text=colors_list, images=image, return_tensors="pt", padding=True)
types = processor(text=types_list, images=image, return_tensors="pt", padding=True)
styles = processor(text=styles_list, images=image, return_tensors="pt", padding=True)
 
# 4. 모델 추론
outputs = model(**colors)
logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도
probs = logits_per_image.softmax(dim=1)  # 확률 계산
max_index = torch.argmax(probs, dim=1).item()
print(colors_list[max_index])


outputs = model(**types)
logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도
probs = logits_per_image.softmax(dim=1)  # 확률 계산
max_index = torch.argmax(probs, dim=1).item()
print(types_list[max_index])

outputs = model(**styles)
logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도
probs = logits_per_image.softmax(dim=1)  # 확률 계산
max_index = torch.argmax(probs, dim=1).item()
print(styles_list[max_index])