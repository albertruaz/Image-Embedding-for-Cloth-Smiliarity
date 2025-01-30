from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
import time
import argparse

def main():
    # 0) 임베딩할 Product가 생성된 날짜 지정해주기
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='Date in YYYY-MM-DD format')
    args = parser.parse_args()
    date = args.date
    
    # 1) MySQL DB에서 특정 조건의 상품정보를 가져옴
    mysql_db = DBConnector()
    try:
        where_condition = f"created_at LIKE '{date}%'"
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
