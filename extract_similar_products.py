from db.vector_db_connector import VectorDBConnector
from db.db_connector import DBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel

def main():
    pg_db = VectorDBConnector()
    mysql_db = DBConnector()

    try:
        # product_id 목록 가져오기
        product_ids = pg_db.fetch_product_ids()

        # 각 product_id에 대해 유사 상품 추출
        for product_id in product_ids:
            similar_products = pg_db.get_similar_products(product_id, top_k=10)

            print(f"Product ID: {product_id}")
            print("Similar Products with Links:")

            for similar_id, distance in similar_products:
                similar_product_links = mysql_db.findLinksById(similar_id)
                if similar_product_links:
                    for link in similar_product_links:
                        print(f"- Link: {link} (Distance: {distance})")
                else:
                    print(f"- Product ID: {similar_id} (Distance: {distance})")
            print("\n")

    finally:
        # 연결 종료
        pg_db.close()
        mysql_db.close()

if __name__ == "__main__":
    main()