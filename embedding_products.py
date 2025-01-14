from db.db_connector import DBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel

def main():
    db_connector = DBConnector()
    try:
        where_condition = "created_at LIKE '2025-01-11%'"
        product_datas = db_connector.fetch_product_data(where_condition) # [(id, main_image), ...]
    finally:
        db_connector.close()


    model = MediaPipeEmbeddingModel(model_name="embedder.tflite")
    image_embeddings = model.embed_batch(product_datas,(224, 224))
    
    # 3) 임베딩 결과를 다시 DB에 저장
    # db_connector.connect()
    # try:
    #     db_connector.save_embeddings(image_embeddings)
    # finally:
    #     db_connector.close()

if __name__ == "__main__":
    main()
