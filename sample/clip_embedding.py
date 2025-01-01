from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 및 텍스트 데이터 준비
image = Image.open("data/2.png")  # 옷 이미지
text = ["red shirt", "blue jeans", "jacket"]  # 텍스트 라벨

colors_list = ["red","green","blue","black","gray"]  # 텍스트 리스트
types_list = ["아우터","상의","바지","원피스","패션소품","가방","스커트","셋업","신발"]
styles_list = ["캐주얼","스트릿","러블리","브랜드","비즈니즈캐주얼","아메카지","스포티","빈티지"]



# 이미지와 텍스트를 전처리
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# 임베딩 계산
outputs = model(**inputs)
image_embedding = outputs.image_embeds  # 이미지 임베딩
text_embedding = outputs.text_embeds    # 텍스트 임베딩

# 유사도 계산 (코사인 유사도)
import torch
similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding)
print(similarity)
