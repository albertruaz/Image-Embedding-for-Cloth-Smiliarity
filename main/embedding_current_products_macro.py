from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
import time

def main():
    
    date = '2025-01-29%'
    
    # 1) MySQL DB에서 특정 조건의 상품정보를 가져옴
    mysql_db = DBConnector()
    try:
        where_condition = f"created_at LIKE '{date}%' and status = 'SALE'"
        product_datas = mysql_db.get_product_data(where_condition, 200) 
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
