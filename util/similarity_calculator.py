import numpy as np
from scipy.spatial.distance import cosine

class SimilarityCalculator:
    """
    embeddings dict를 넣어주면 모든 pairwise cosine 거리를 구하고,
    이를 1 - distance(= similarity)로 변환한 뒤 
    각 product별 유사상품 리스트를 정렬해서 반환하는 클래스.
    """

    def calculate_similarity(self, embeddings: dict) -> dict:
        """
        :param embeddings: { product_id: np.array([...]) }
        :return: { 
                    product_id: [ (other_product_id, distance), ..., ],
                    ...}
        낮은 distance 값이 더 유사한 상품을 의미
        """
        result = {}
        product_ids = list(embeddings.keys())
        n = len(product_ids)

        for i in range(n):
            pid_i = product_ids[i]
            emb_i = embeddings[pid_i]
            distances = []

            for j in range(i + 1, n):
                pid_j = product_ids[j]
                emb_j = embeddings[pid_j]

                # cosine distance 직접 사용 (1 - similarity 연산 제거)
                distance = cosine(emb_i, emb_j)
                
                distances.append((pid_j, distance))
                
                if pid_j not in result:
                    result[pid_j] = []
                result[pid_j].append((pid_i, distance))

            # distance가 작은 순으로 정렬 (가장 유사한 것이 앞으로)
            distances.sort(key=lambda x: x[1])
            result[pid_i] = distances

        # 각 product_id에 대해 distance 기준 정렬
        for pid in result:
            result[pid].sort(key=lambda x: x[1])

        return result

    def get_similar_and_dissimilar_products(self, similarities: dict, top_k: int = 10) -> tuple[dict, dict]:
        """
        각 상품별로 가장 유사한 상품 top_k개와 가장 다른 상품 top_k개의 id 리스트를 반환
        
        :param similarities: calculate_similarity()의 결과
        :param top_k: 반환할 상품 개수
        :return: (similar_products, dissimilar_products)
                각각 { product_id: [similar_product_ids], ... }
        """
        similar_products = {}
        dissimilar_products = {}

        for pid, similarity_list in similarities.items():
            # 유사한 상품 top_k (이미 정렬되어 있으므로 앞에서부터)
            similar_products[pid] = [sim[0] for sim in similarity_list[:top_k]]
            
            # 가장 다른 상품 top_k (뒤에서부터)
            dissimilar_products[pid] = [sim[0] for sim in similarity_list[-top_k:]]

        return similar_products, dissimilar_products
