from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from typing import List, Dict
import os
import json

from dotenv import load_dotenv
load_dotenv()
class SingletonMeta(type):
    """
    Singleton을 위한 메타클래스.
    __call__을 오버라이딩해서, 이미 인스턴스가 존재하면 새로 만들지 않고
    기존 인스턴스를 반환하도록 함.
    """
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class DBConnector(metaclass=SingletonMeta):
    def __init__(self):
        """
        .env에서 SSH 및 DB 정보 읽어와서
        객체 생성 시 자동으로 SSH 터널, DB 커넥션 풀 설정
        """
        # 이미 _instance가 존재할 경우, __init__이 다시 호출될 수 있으므로
        # 필요한 경우, 재호출 방어 로직이 필요할 수 있음
        # 여기서는 간단히 'engine이 이미 있으면 초기화 로직을 생략'하는 식으로 처리 가능
        if hasattr(self, 'engine') and self.engine is not None:
            return  # 이미 초기화된 경우는 재실행 방지

        self.ssh_host = os.getenv('SSH_HOST')
        self.ssh_username = os.getenv('SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('SSH_PKEY_PATH')
        self.remote_bind_address = (
            os.getenv('DB_HOST'),
            int(os.getenv('DB_PORT', 3306))
        )

        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_name = os.getenv('DB_NAME')

        # 커넥션 풀 설정값
        self.pool_size = int(os.getenv('DB_POOL_SIZE', 10))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', 20))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', 3600))

        self.tunnel = None
        self.engine = None
        self.Session = None

        # 객체 생성 시 자동 연결
        self.connect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def connect(self):
        """SSH 터널 및 SQLAlchemy 세션 연결"""
        # 이미 연결되어 있다면 재연결을 피하기 위해 검사
        if self.tunnel is not None and self.tunnel.is_active:
            return

        self.tunnel = SSHTunnelForwarder(
            (self.ssh_host, 22),
            ssh_username=self.ssh_username,
            ssh_pkey=self.ssh_pkey_path,
            remote_bind_address=self.remote_bind_address
        )
        self.tunnel.start()

        db_url = f"mysql+pymysql://{self.db_user}:{self.db_password}@127.0.0.1:{self.tunnel.local_bind_port}/{self.db_name}"
        self.engine = create_engine(
            db_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle
        )
        self.Session = sessionmaker(bind=self.engine)

    def close(self):
        """세션/엔진/터널 종료"""
        if self.Session:
            self.Session.close_all()
            self.Session = None
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
        self.tunnel = None
        self.engine = None
        
        DBConnector._instance = None

    def get_s3_url(self, file_name: str) -> str:
        """S3(또는 CloudFront) 경로 생성"""
        cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        protocol = "https"
        if not file_name or not cloudfront_domain:
            return None
        return f"{protocol}://{cloudfront_domain}/{file_name}"

    def get_product_data(self, where_condition: str = "1!=1", limit: int = 500, batch_no: int = 0) -> list:
        offset = batch_no * limit

        session = self.Session()
        try:
            sql = text(f"""
                SELECT 
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM product
                WHERE 
                    -- status LIKE "SALE" AND
                {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            result = session.execute(sql)

            products = []
            for row in result.fetchall():
                products.append((
                    row[0],  # id
                    self.get_s3_url(row[1]) if row[1] else None,  # main_image -> S3 URL
                    row[2],  # status
                    row[3],  # primary_category_id
                    row[4],  # secondary_category_id
                ))
            return products
        finally:
            session.close()
    
    def get_product_ids_by_condition(self, where_condition: str = "1!=1") -> list:
        session = self.Session()
        try:
            sql = text(f"""
                SELECT 
                    id
                FROM product
                WHERE
                {where_condition}
            """)
            result = session.execute(sql).fetchall()
            return [row[0] for row in result]
        finally:
            session.close()

    def find_links_by_id(self, product_id: str) -> list:
        session = self.Session()
        try:
            sql = text("""
                SELECT link
                FROM product
                WHERE id = :product_id
            """)
            results = session.execute(sql, {"product_id": product_id}).fetchall()
            return [row[0] for row in results if row[0]]
        finally:
            session.close()

    def update_similar_products(self, product_similar_products: Dict[str, List[str]]):
        if not product_similar_products:
            return []
        session = self.Session()
        try:
            sql = text("""
                UPDATE product
                SET similar_ids = :similar
                WHERE id = :product_id
            """)
            
            data = [
                {
                    "product_id": product_id, 
                    # JSON 배열로 저장할 것이므로 리스트를 직렬화
                    "similar": json.dumps(similar_list, ensure_ascii=False)
                }
                for product_id, similar_list in product_similar_products.items()
            ]
            
            session.execute(sql, data)  # executemany 방식으로 한 번에 여러 UPDATE
            session.commit()
        finally:
            session.close()
