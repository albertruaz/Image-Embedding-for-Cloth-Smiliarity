# Image-Embedding-for-Cloth-Smiliarity

## 개요

유사한 이미지의 상품 추천을 위한, 이미지 임베딩 구현

## 구성

### 1. embedding_current_products.py

- 상품을 등록시 실행
- ID를 기반으로 해당 상품 이미지의 임베딩 백터 DB에 저장

### 2. extract_current_similar_products.py

- 등록된 상품의 유사 상품 리스트업

### 3. fetch_product_daily.py

- 새벽에 프로세스로 실행
- 팔린 상품들 업데이트
- 유사 상품에 팔린 상품이 일정수를 넘어가면 다시 추출

## 백엔드 사용 예시

```java
// embedding by id lists (by gpt)

public class PythonProcessService {
    public void runPythonScript(List<Integer> idList) throws IOException {
        // ID 리스트를 ","로 변환 (환경 변수로 넘길 형태)
        String joinedIds = String.join(",", idList.stream().map(String::valueOf).toList());

        // ProcessBuilder로 Python 실행 (환경 변수 설정)
        ProcessBuilder processBuilder = new ProcessBuilder("python", "/path/to/main/embedding_current_products.py");
        processBuilder.environment().put("PRODUCT_IDS", joinedIds);

        processBuilder.start();
    }
}
```
