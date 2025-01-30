from db.vector_db_connector import VectorDBConnector
from db.db_connector import DBConnector
import os


def main():
    
    check = True
    # check = False
    extract_num = 20
    condition = "similar_ids like '[]' and status = 'SALE'"
    

    mysql_db = DBConnector()
    pg_db = VectorDBConnector()
    try:
        print("Current Condition is...")
        print(condition)
        print("\n")
        product_ids = mysql_db.get_product_ids_by_condition(condition)
        print("check1\n")
        print(product_ids)

        similar_products = pg_db.get_similar_products(product_ids, top_k=extract_num)
        print("check2\n")
        print(similar_products)
        mysql_db.update_similar_products(similar_products)
        print("check3\n")
        if check == True:
            print("\n")
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
        pg_db.close()
        mysql_db.close()

if __name__ == "__main__":
    main()