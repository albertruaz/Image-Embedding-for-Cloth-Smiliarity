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
                    product_id: [ (other_product_id, similarity), ..., ],
                    ...}
        """
        result = {}
        product_ids = list(embeddings.keys())

        for i in range(len(product_ids)):
            pid_i = product_ids[i]
            emb_i = embeddings[pid_i]

            similarities = []
            for j in range(len(product_ids)):
                if i == j:
                    continue

                pid_j = product_ids[j]
                emb_j = embeddings[pid_j]

                # cosine distance 계산
                dist = cosine(emb_i, emb_j)
                # 유사도 = 1 - distance
                similarity_score = 1 - dist

                similarities.append((pid_j, similarity_score))

            # 유사도가 높은 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            result[pid_i] = similarities

        return result
