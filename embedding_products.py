from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel

def main():
    # 1) MySQL DB에서 특정 조건의 상품정보를 가져옴
    mysql_db = DBConnector()
    try:
        where_condition = "created_at LIKE '2025-01-11%'"
        product_datas = mysql_db.fetch_product_data(where_condition) 
    finally:
        mysql_db.close()

    # 2) 이미지 임베딩 생성
    model = MediaPipeEmbeddingModel(model_name="embedder.tflite")
    image_embeddings = model.embed_batch(product_datas, (224, 224))
    
    # 3) PGVector DBConnector를 통해 임베딩 저장
    pg_db = VectorDBConnector()
    try:
        # pg_db.create_vector_table(dimension=1024)
        pg_db.upsert_embeddings(image_embeddings)
        # 예시: 특정 product_id에 대해 유사한 상품 10개 확인
        # sample_pid = list(image_embeddings.keys())[0]  # 임의로 첫 번째 product_id
        # similar_products = pg_db.get_similar_products(sample_pid, top_k=10)
        # print(f"Product {sample_pid}와 유사한 Top 10:", similar_products)
    finally:
        pg_db.close()

if __name__ == "__main__":
    main()
