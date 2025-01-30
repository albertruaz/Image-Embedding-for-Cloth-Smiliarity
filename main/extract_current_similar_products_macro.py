from db.vector_db_connector import VectorDBConnector
from db.db_connector import DBConnector
import os


def main():
    
    # check = True
    check = False
    extract_num = 20
    # condition = "created_at BETWEEN '2025-01-10 00:00:00' AND '2025-01-15 00:00:00' and status = 'SALE'"
    conditions = [
        "created_at BETWEEN '2025-01-15 00:00:00' AND '2025-01-20 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-01-20 00:00:00' AND '2025-01-25 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-01-25 00:00:00' AND '2025-01-30 00:00:00' and status = 'SALE'",

        "created_at BETWEEN '2025-01-30 00:00:00' AND '2025-02-01 00:00:00' and status = 'SALE'",

        "created_at BETWEEN '2025-02-01 00:00:00' AND '2025-02-05 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-02-05 00:00:00' AND '2025-02-10 00:00:00' and status = 'SALE'",

        "created_at BETWEEN '2025-02-10 00:00:00' AND '2025-02-15 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-02-15 00:00:00' AND '2025-02-20 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-02-20 00:00:00' AND '2025-02-25 00:00:00' and status = 'SALE'",
        "created_at BETWEEN '2025-02-25 00:00:00' AND '2025-02-30 00:00:00' and status = 'SALE'"
    ]
    new_conditions = [
        "similar_ids like '[]' and status = 'SALE'"
    ]

    for condition in new_conditions:
        mysql_db = DBConnector()
        pg_db = VectorDBConnector()
        try:
            print("Current Condition is...")
            print(condition)
            print("\n")
            
            product_ids = mysql_db.get_product_ids_by_condition(condition)
            print("check1\n")
            similar_products = pg_db.get_similar_products(product_ids, top_k=extract_num)
            print("check2\n")
            mysql_db.update_similar_products(similar_products)
            print("check3\n")
            # if check == True:
            #     print(product_ids)
            #     print("\n")
            #     print(similar_products)
            #     print("\n")
            #     for product_id in product_ids:
            #         print(f"Product ID: {product_id}")
            #         print("Similar Products with Links:")
            #         for similar_id in similar_products.get(product_id, []): 
            #             similar_product_links = mysql_db.find_links_by_id(similar_id)
            #             if similar_product_links:
            #                 for link in similar_product_links:
            #                     print(f"- Link: {link}")
            #         print("\n")
        finally:
            pg_db.close()
            mysql_db.close()

if __name__ == "__main__":
    main()