from db.db_connector import DBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
from util.similarity_calculator import SimilarityCalculator

def main():

    db_connector = DBConnector()
    product_datas = db_connector.fetch_product_data(
        date_prefix="2025-01-09 09%"  # 필요에 따라 다른 날짜/형식 지정
    )
    product_images = db_connector.fetch_product_images(product_datas)
    
    print(product_datas)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(product_images)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # embedding_model = MediaPipeEmbeddingModel(model_name="embedder.tflite")

    # image_embeddings = {}
    # for (product_id, image_path) in product_datas:
    #     embedding = embedding_model.get_image_embedding(image_path)
    #     image_embeddings[product_id] = embedding

    # # 5) 유사도 계산
    # sim_calc = SimilarityCalculator()
    # similar_items = sim_calc.calculate_similarity(image_embeddings)

    # # 6) DB에 임베딩 정보와 유사상품 정보 저장
    # db_connector.save_embeddings_and_similarities(image_embeddings, similar_items)

    # print("작업 완료! DB에 임베딩과 유사도 리스트가 저장되었습니다.")

if __name__ == "__main__":
    main()
