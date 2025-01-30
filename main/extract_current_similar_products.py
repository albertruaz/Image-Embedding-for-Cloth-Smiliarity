from db.vector_db_connector import VectorDBConnector
from db.db_connector import DBConnector
import os

def main():
    extract_num = 100
    
    product_ids = os.getenv("PRODUCT_IDS", "").split(",")
    if not product_ids or product_ids == [""]:
        return
    product_ids = [int(pid) for pid in product_ids if pid.isdigit()]
    
    mysql_db = DBConnector()
    pg_db = VectorDBConnector()
    try:
        similar_products = pg_db.get_similar_products(product_ids, top_k=extract_num)
        mysql_db.update_similar_products(similar_products)
    finally:
        # 연결 종료
        pg_db.close()
        mysql_db.close()

if __name__ == "__main__":
    main()