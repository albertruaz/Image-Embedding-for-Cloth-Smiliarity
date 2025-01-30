from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
import os

def main():
    # 0) 임베딩할 Product의 id 리스트로 변환
    product_ids = os.getenv("PRODUCT_IDS", "").split(",")
    if not product_ids or product_ids == [""]:
        return
    product_ids = [int(pid) for pid in product_ids if pid.isdigit()]

    # 1) MySQL DB에서 특정 조건의 상품정보를 가져옴
    mysql_db = DBConnector()
    try:
        id_list_str = ",".join(map(str, product_ids))
        where_condition = f"id IN ({id_list_str})"
        product_datas = mysql_db.get_product_data(where_condition, 5000) 
    finally:
        mysql_db.close()
    
    # 2) 이미지 임베딩 생성
    model = MediaPipeEmbeddingModel(model_name="embedder.tflite")
    product_datas_with_embedding = model.embed_batch(product_datas, (224, 224))

    # 3) PGVector DBConnector를 통해 임베딩 저장
    vector_db = VectorDBConnector()
    try:
        vector_db.upsert_embeddings(product_datas_with_embedding)
    finally:
        vector_db.close()

if __name__ == "__main__":
    main()
