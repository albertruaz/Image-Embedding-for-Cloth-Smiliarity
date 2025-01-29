from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
import time

def main():
    for batch_no in range(5+1):
        start_time = time.time()
        date = '2024-05'

        # 1) MySQL DB에서 특정 조건의 상품정보를 가져옴
        mysql_db = DBConnector()
        try:
            where_condition = f"created_at LIKE '{date}%'"
            product_datas = mysql_db.fetch_product_data(where_condition, 500, batch_no) 
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
        
        print("Finish Embedding of date: ", date, " and no: ", batch_no)
        print(len(product_datas)," toke ", time.time() - start_time, "time")

if __name__ == "__main__":
    main()
