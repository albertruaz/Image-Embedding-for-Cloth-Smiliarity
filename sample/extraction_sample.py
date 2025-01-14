from db.db_connector import DBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
from util.similarity_calculator import SimilarityCalculator

def convert_ids_to_links(similar_products: dict, product_data: list) -> dict:
    """
    상품 ID 기반의 유사도 딕셔너리를 링크 기반으로 변환
    
    :param similar_products: Dict[product_id, List[similar_product_ids]]
    :param product_data: List of tuples (id, main_image, link)
    :return: Dict[product_link, List[similar_product_links]]
    """
    # id to link 매핑 생성
    id_to_link = {str(pid): link for pid, _, link in product_data}
    
    similar_links = {}
    for pid, similar_ids in similar_products.items():
        # 현재 상품의 링크
        product_link = id_to_link.get(str(pid))
        if not product_link:
            continue
            
        # 유사 상품들의 링크
        similar_links[product_link] = [
            id_to_link.get(str(similar_id))
            for similar_id in similar_ids
            if id_to_link.get(str(similar_id))
        ]
    
    return similar_links

def save_similarity_results(similar_links: dict, dissimilar_links: dict, filename: str) -> None:
    """
    유사도 결과를 가독성 좋게 파일로 저장
    
    :param similar_links: Dict[product_link, List[similar_product_links]]
    :param dissimilar_links: Dict[product_link, List[dissimilar_product_links]]
    :param filename: 저장할 파일 이름
    """
    import os
    from datetime import datetime
    
    # save 폴더가 없으면 생성
    save_dir = "save"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 파일명에 날짜 추가
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(save_dir, f"{filename}_{date_str}.txt")
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write("=== 유사 상품 목록 ===\n\n")
        for product_link, similar_product_links in similar_links.items():
            f.write(f"상품: {product_link}\n")
            f.write("유사한 상품들:\n")
            for idx, link in enumerate(similar_product_links, 1):
                f.write(f"{idx}. {link}\n")
            f.write("\n" + "="*50 + "\n\n")
            
        f.write("\n\n=== 비유사 상품 목록 ===\n\n")
        for product_link, dissimilar_product_links in dissimilar_links.items():
            f.write(f"상품: {product_link}\n")
            f.write("다른 스타일의 상품들:\n")
            for idx, link in enumerate(dissimilar_product_links, 1):
                f.write(f"{idx}. {link}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"결과가 저장되었습니다: {full_path}")

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
