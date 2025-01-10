from sshtunnel import SSHTunnelForwarder
import pymysql
import os
from typing import Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class S3UrlFinder:
    def __init__(self):
        self.cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        self.protocol = "https"
        
    def locate(self, file_name: str) -> str: 
        if not file_name: 
            return None
        return f"{self.protocol}://{self.cloudfront_domain}/{file_name}"

class DBConnector:
    def __init__(self):
        self.ssh_config = {
            'ssh_host': os.getenv('SSH_HOST'),
            'ssh_username': os.getenv('SSH_USERNAME'),
            'ssh_pkey_path': os.getenv('SSH_PKEY_PATH'),
            'db_host': os.getenv('DB_HOST'),
            'db_port': int(os.getenv('DB_PORT', 3306))
        }
        
        self.db_config = {
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'db': os.getenv('DB_NAME'),
            'charset': 'utf8mb4'
        }
        
        self.s3_url_finder = S3UrlFinder()

    def connect(self) -> Tuple[SSHTunnelForwarder, pymysql.connections.Connection]:
        """SSH 터널과 DB 연결을 설정합니다."""
        tunnel = SSHTunnelForwarder(
            (self.ssh_config['ssh_host'], 22),
            ssh_username=self.ssh_config['ssh_username'],
            ssh_pkey=self.ssh_config['ssh_pkey_path'],
            remote_bind_address=(self.ssh_config['db_host'], self.ssh_config['db_port'])
        )
        tunnel.start()

        connection = pymysql.connect(
            host='127.0.0.1',
            port=tunnel.local_bind_port,
            **self.db_config
        )
        
        return tunnel, connection

    def fetch_product_data(self, date_prefix: str = '2025-01-09 09%') -> Dict[int, str]:
        """특정 날짜의 제품 이미지 URL을 조회합니다."""
        tunnel, connection = self.connect()
        
        try:
            with connection.cursor() as cursor:
                sql = """
                    SELECT id, main_image
                    FROM product
                    WHERE created_at LIKE %s
                """
                cursor.execute(sql, (date_prefix,))
                products = cursor.fetchall()
            return products 
            
        finally:
            connection.close()
            tunnel.close()

    def fetch_product_images(self, product_data: Dict[int, str]) -> Dict[int, str]:
        
        product_urls = {
            product_id: self.s3_url_finder.locate(main_image)
            for product_id, main_image in product_data
        }
        return product_urls
    
    def save_embeddings(self, product_id: int, embeddings: np.ndarray) -> None:
        """제품의 임베딩을 저장합니다."""
        tunnel, connection = self.connect()
        
        try:
            with connection.cursor() as cursor:
                sql = """
                    UPDATE product 
                    SET embedding = %s
                    WHERE id = %s
                """
                # numpy 배열을 바이너리로 변환
                embedding_binary = embeddings.tobytes()
                cursor.execute(sql, (embedding_binary, product_id))
            
            connection.commit()
            
        finally:
            connection.close()
            tunnel.close()

    def get_embeddings(self, product_id: int) -> np.ndarray:
        """제품의 임베딩을 조회합니다."""
        tunnel, connection = self.connect()
        
        try:
            with connection.cursor() as cursor:
                sql = """
                    SELECT embedding 
                    FROM product 
                    WHERE id = %s
                """
                cursor.execute(sql, (product_id,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    # 바이너리를 numpy 배열로 변환
                    embedding_binary = result[0]
                    return np.frombuffer(embedding_binary, dtype=np.float32)
                return None
                
        finally:
            connection.close()
            tunnel.close()


