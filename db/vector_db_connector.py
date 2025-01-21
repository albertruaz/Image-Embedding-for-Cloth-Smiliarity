# file: vector_db_connector.py

import os
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

class VectorDBConnector:
    """
    PostgreSQL + PGVector 전용 DBConnector
    """
    def __init__(self):
        self.ssh_host = os.getenv('PG_SSH_HOST')           # SSH가 필요하다면
        self.ssh_username = os.getenv('PG_SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('PG_SSH_PKEY_PATH')
        self.pg_host = os.getenv('PG_HOST')               # PostgreSQL 호스트
        self.pg_port = int(os.getenv('PG_PORT', 5432))    # PostgreSQL 포트
        self.pg_user = os.getenv('PG_USER')
        self.pg_password = os.getenv('PG_PASSWORD')
        self.pg_dbname = os.getenv('PG_DB_NAME')

        # 커넥션 풀 설정값
        self.pool_size = int(os.getenv('PG_POOL_SIZE', 5))
        self.max_overflow = int(os.getenv('PG_MAX_OVERFLOW', 10))
        self.pool_timeout = int(os.getenv('PG_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('PG_POOL_RECYCLE', 3600))

        self.tunnel = None
        self.engine = None
        self.Session = None

        # 커넥션 초기화
        self.connect()

    def connect(self):
        # SSH 터널이 필요한 경우
        # (SSH 터널이 없다면 바로 포트/호스트로 연결)
        if self.ssh_host and self.ssh_username and self.ssh_pkey_path:
            self.tunnel = SSHTunnelForwarder(
                (self.ssh_host, 22),
                ssh_username=self.ssh_username,
                ssh_pkey=self.ssh_pkey_path,
                remote_bind_address=(self.pg_host, self.pg_port)
            )
            self.tunnel.start()
            local_port = self.tunnel.local_bind_port
            db_host = '127.0.0.1'
            db_port = local_port
        else:
            # SSH 터널 없이 직접 연결
            db_host = self.pg_host
            db_port = self.pg_port

        db_url = f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{db_host}:{db_port}/{self.pg_dbname}"

        self.engine = create_engine(
            db_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle
        )
        self.Session = sessionmaker(bind=self.engine)

    def close(self):
        if self.Session:
            self.Session.close_all()
            self.Session = None
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
        self.tunnel = None
        self.engine = None

    def create_vector_table(self, dimension: int = 1024):
        """
        PGVector 확장 활성화 및 product_embedding 테이블 생성
        """
        session = self.Session()
        try:
            # 1) PGVector 확장 설치
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()

            # 2) 테이블 생성
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS product_embedding (
                product_id TEXT PRIMARY KEY,
                embedding VECTOR({dimension})
            );
            """
            session.execute(text(create_table_sql))
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    def fetch_product_ids(self):
        """
        product_embedding 테이블에서 모든 product_id를 가져오는 함수
        """
        session = self.Session()
        try:
            query = text("SELECT product_id FROM product_embedding")
            product_ids = session.execute(query).fetchall()
            return [row[0] for row in product_ids]  # 리스트 형태로 반환
        finally:
            session.close()
            
    def upsert_embeddings(self, embeddings: list):
        session = self.Session()
        try:
            for item in embeddings:
                product_id = item["product_id"]
                vector_vals = item["embedding"]

                # 벡터 데이터를 PostgreSQL VECTOR 형식으로 변환
                if not all(isinstance(val, (int, float)) for val in vector_vals):
                    raise ValueError(f"Vector contains non-numeric values: {vector_vals}")
                vector_str = "[" + ",".join(map(str, vector_vals)) + "]"

                # SQL 쿼리 실행
                sql = text("""
                    INSERT INTO product_embedding (product_id, embedding)
                    VALUES (:pid, :vec)
                    ON CONFLICT (product_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                """)

                # 벡터 데이터를 문자열로 전달
                session.execute(sql, {"pid": product_id, "vec": vector_str})
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_similar_products(self, product_id: str, top_k: int = 10) -> list:
        """
        예시로 Euclidean distance 사용 (<->)
        Cosine distance를 사용하려면 (<#>) 또는 다른 문법 사용
        """
        session = self.Session()
        try:
            # 1) 대상 product_id의 벡터 가져오기
            query_vec_sql = text("SELECT embedding FROM product_embedding WHERE product_id=:pid")
            res = session.execute(query_vec_sql, {"pid": product_id}).fetchone()
            if not res:
                return []

            target_vec = res[0]

            # 2) 유사도 계산 (Euclidean distance)
            #    자신 제외, distance ASC로 정렬
            sim_sql = text("""
                SELECT product_id, (embedding <#> :tvec) AS distance
                FROM product_embedding
                WHERE product_id != :pid
                ORDER BY embedding <#> :tvec
                LIMIT :top_k
            """)
            rows = session.execute(sim_sql, {"tvec": target_vec, "pid": product_id, "top_k": top_k}).fetchall()

            # [(product_id, distance), ...] 형태로 반환
            similar_list = [(r[0], r[1]) for r in rows]
            return similar_list
        finally:
            session.close()
