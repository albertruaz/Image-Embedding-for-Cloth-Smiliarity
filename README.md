# Image-Embedding-for-Cloth-Smiliarity

## 개요

유사한 이미지의 상품 추천을 위한, 이미지 임베딩 구현

## 사용 설명

### 1. Product가 SALE로 바뀌는 시점에 파이썬 프로세스를 통한 임베딩

- (SALE 변경 전에는 카테고리가 수작업으로 변경될 수 있기에)
- 환경 설정

  - uv python 환경 세팅
  - 파이썬 git clone (deploy branch)
  - ENV 파일 설정

- 실행 예시 (conda 실행 후, id 리스트 매개변수로 넘겨주며 파이썬 실행)

```java
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class PythonProcessService {
    public void runPythonScript(List<Integer> idList) throws IOException {
        // ID 리스트를 ","로 변환 (환경 변수로 넘길 형태)
        String joinedIds = idList.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));

        String[] command = {
            "/bin/bash", "-c",
            "source ~/.bashrc && source /path/to/recommendation_env/bin/activate && python /path/to/main/embedding_current_products_daily.py"
        }; // 여기도 경로에 맞게 수정해줘야할듯

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().put("PRODUCT_IDS", joinedIds);
        processBuilder.start();
    }
}
```

### 2. 유사도 추출

- 단순히 vectorDB에서 sql문으로 추출 가능
- vectorDB 접속 정보는 .ENV 참고
- 예시:

```sql
-- 아직 안팔렸고, 같은 카테고리 상품에 대해, top_k개의 유사한 상품 추출 SQL문
WITH ranked AS (
    SELECT
        p2.id AS similar_id,
        (p1.image_vector <#> p2.image_vector) AS distance,
        ROW_NUMBER() OVER (
            ORDER BY (p1.image_vector <#> p2.image_vector)
        ) AS rn
    FROM product p1
    JOIN product p2 ON
        p1.id != p2.id
        AND p1.primary_category_id = p2.primary_category_id
        AND p1.secondary_category_id = p2.secondary_category_id
    WHERE
        p1.id = :pid  -- 특정 ID
        AND p2.status = 'SALE'
)
SELECT similar_id, distance
FROM ranked
WHERE rn <= :top_k
ORDER BY distance;
```

### 3. SALE 정보 업데이트

- (SALE 정보를 가져온 상태여야, SQL문에서 join을 쓰지 않아도 돼서, Table에 칼럼을 추가함)
- (product의 status가 수정되는 상황에 vector db의 status도 같이 반영해야함)
