import os
import numpy as np
from typing import List, Tuple, Set
from scipy.spatial.distance import cosine
import json

class TestEmbedding:
    def __init__(self, model_name: str, data_dir: str, model_class: object, folders: List[str]):
        self.model = model_class(model_name=model_name)
        self.data_dir = data_dir
        self.folders = folders
        self.embeddings = self.get_embeddings()
        

    def get_embeddings(self) -> dict:
        embeddings = {}
        # 지정된 폴더들에 대해서만 임베딩 계산
        for folder in self.folders:
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                # 폴더 내 모든 이미지 파일 찾기
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                folder_embeddings = []
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    embedding = self.model.get_image_embedding(image_path)
                    embedding = embedding.detach().numpy().squeeze()
                    embedding = embedding / np.linalg.norm(embedding)
                    folder_embeddings.append(embedding)
                
                if folder_embeddings:  # 빈 폴더가 아닌 경우만 저장
                    embeddings[folder] = folder_embeddings
        
        return embeddings

    def calculate_accuracy(self, similar_pairs: List[Tuple[str, str, float]], 
                           different_pairs: List[Tuple[str, str, float]], 
                           self_similarity_folders: Set[str] = None,
                           similarity_threshold: float = 0.5) -> float:
        # def sigmoid(x, center=similarity_threshold, steepness=10):
        #     return 1 / (1 + np.exp(-steepness * (x - center)))
        
        total_score = 0
        total_weight = 0

        # Check self-similarity within folders
        for folder in self_similarity_folders:
            if folder not in self.embeddings:
                raise ValueError(f"Folder {folder} not found in embeddings")
            
            # Assuming self.embeddings[folder] is a list of embeddings for all images in the folder
            folder_embeddings = self.embeddings[folder]
            num_images = len(folder_embeddings)
            if num_images < 2:
                continue  # Skip if less than two images

            for i in range(num_images):
                for j in range(i + 1, num_images):
                    similarity = cosine(folder_embeddings[i], folder_embeddings[j])
                    total_score += similarity
                    total_weight += 1
                    print("Self Similarity ",folder,": ",similarity)
        
        # Check similar pairs - higher similarity should give score closer to 1
        for folder1, folder2, weight in similar_pairs:
            if folder1 not in self.embeddings or folder2 not in self.embeddings:
                raise ValueError(f"Folders {folder1} or {folder2} not found in embeddings")
            
            # Calculate similarity between all image pairs in the folders
            for emb1 in self.embeddings[folder1]:
                for emb2 in self.embeddings[folder2]:
                    similarity = cosine(emb1, emb2)
                    total_score += similarity * weight
                    total_weight += weight
                    print("Similar Similarity ", folder1, folder2, ": ",similarity)
                
        # Check different pairs - lower similarity should give score closer to 1
        for folder1, folder2, weight in different_pairs:
            if folder1 not in self.embeddings or folder2 not in self.embeddings:
                raise ValueError(f"Folders {folder1} or {folder2} not found in embeddings")
            for emb1 in self.embeddings[folder1]:
                for emb2 in self.embeddings[folder2]:
                    similarity = -1 * cosine(emb1, emb2)
                    total_score += similarity * weight
                    total_weight += weight
                    print("Different Similarity ", folder1, folder2, ": ",similarity)

        return total_score / total_weight if total_weight > 0 else 0.0

def main():
    # Read environment variables for paths
    embedding_model_file = os.getenv('EMBEDDING_MODEL_FILE', 'clip_embedding_model')
    model_name = os.getenv('MODEL_NAME', 'openai/clip-vit-base-patch32')
    data_dir = os.getenv('DATA_DIR', './data')
    # Import the specified embedding model
    if embedding_model_file == 'blip_embedding_model':
        from model.blip_embedding_model import BLIPEmbeddingModel
        model_class = BLIPEmbeddingModel
    elif embedding_model_file == 'clip_embedding_model':
        from model.clip_embedding_model import CLIPEmbeddingModel
        model_class = CLIPEmbeddingModel
    else:
        raise ValueError(f"Unknown embedding model file: {embedding_model_file}")
    

    # Define similar and different pairs from environment variables
    similar_pairs = eval(os.getenv('SIMILAR_PAIRS', '[]'))
    different_pairs = eval(os.getenv('DIFFERENT_PAIRS', '[]'))
    other_folders = eval(os.getenv('OTHER_FOLDERS', '[]'))
    all_folders = set(other_folders)  # Ensure it's a set
    all_folders.update(folder for pair in similar_pairs + different_pairs for folder in pair[:2])

    # Example usage
    test = TestEmbedding(model_name=model_name, data_dir=data_dir, folders=all_folders)

    # Calculate accuracy
    accuracy = test.calculate_accuracy(similar_pairs, different_pairs, all_folders)
    print("MODEL_NAME: ",model_name)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()