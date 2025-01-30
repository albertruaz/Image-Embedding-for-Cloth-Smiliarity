from db.vector_db_connector import VectorDBConnector
from db.db_connector import DBConnector
import os


def main():
    
    check = True
    extract_num = 10
    condition = "created_at like '2025-01-01%'"
    
    mysql_db = DBConnector()
    pg_db = VectorDBConnector()
    try:
        product_ids = mysql_db.get_product_ids_by_condition(condition)
        similar_products = pg_db.get_similar_products(product_ids, top_k=extract_num)
        # mysql_db.update_similar_products(similar_products)

        if check == True:
            for product_id in product_ids:
                print(f"Product ID: {product_id}")
                print("Similar Products with Links:")
                for similar_id in similar_products.get(product_id, []): 
                    similar_product_links = mysql_db.find_links_by_id(similar_id)
                    if similar_product_links:
                        for link in similar_product_links:
                            print(f"- Link: {link}")
                print("\n")
    finally:
        # 연결 종료
        pg_db.close()
        mysql_db.close()

if __name__ == "__main__":
    main()