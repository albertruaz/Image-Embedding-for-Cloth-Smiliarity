from db.db_connector import DBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
from util.similarity_calculator import SimilarityCalculator

def main():
    save_to_db = False
    
    db_connector = DBConnector()
    try:
        product_datas = db_connector.fetch_product_data()
        product_images = db_connector.fetch_product_images(product_datas)
    finally:
        db_connector.close()

    embedding_model = MediaPipeEmbeddingModel(model_name="embedder.tflite")
    image_embeddings = embedding_model.embed_batch(product_images)
    sim_calc = SimilarityCalculator()
    similarity_pairs = sim_calc.calculate_similarity(image_embeddings)
    similar_products, dissimilar_products = sim_calc.get_similar_and_dissimilar_products(similarity_pairs)
    
    print(similarity_pairs)

    # 유사도 계산
    calculator = SimilarityCalculator()
    similarities = calculator.calculate_similarity(image_embeddings)
    similar_products, dissimilar_products = calculator.get_similar_and_dissimilar_products(similarities)
    
    # ID를 링크로 변환
    similar_links = convert_ids_to_links(similar_products, product_datas)
    dissimilar_links = convert_ids_to_links(dissimilar_products, product_datas)
    
    save_similarity_results(similar_links, dissimilar_links, "similarity_results")


    # 6) DB에 임베딩 정보와 유사상품 정보 저장
    if save_to_db:
        try:
            db_connector.connect()
            db_connector.save_embeddings_and_similarities(image_embeddings, similar_products)
        finally:
            db_connector.close()

    
if __name__ == "__main__":
    main()
