import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

def main():
    """
    저장된 FAISS 벡터스토어를 로드하여 유사도 검색을 테스트합니다.
    """
    # --- 1. 환경 설정 ---
    
    # .env 파일에서 환경 변수 로드 (AZURE_OPENAI_API_KEY 등)
    load_dotenv()

    # 벡터 DB 폴더 경로와 검색할 쿼리 지정
    DB_PATH = "faiss_db"
    query = "SQL"

    # --- 2. 벡터 DB 로드 ---

    # DB 폴더가 존재하는지 확인
    if not os.path.exists(DB_PATH):
        print(f"오류: 벡터 DB 폴더 '{DB_PATH}'를 찾을 수 없습니다.")
        print("DB 생성 스크립트를 먼저 실행해주세요.")
        return

    try:
        # DB 생성 시 사용했던 것과 "동일한" 임베딩 모델을 준비합니다.
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

        print(f"'{DB_PATH}' 폴더에서 벡터스토어를 로드합니다...")
        loaded_vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 로컬 DB 로드 시 필요
        )
        print("로드 완료.")

    except Exception as e:
        print(f"벡터스토어 로드 중 오류 발생: {e}")
        return

    # --- 3. 유사도 검색 실행 ---

    print(f"\n--- \"{query}\"(으)로 유사도 검색을 시작합니다. ---")
    
    # k: 검색 결과 개수
    results = loaded_vectorstore.similarity_search(query, k=3)

    if not results:
        print(" -> 유사한 내용을 찾을 수 없습니다.")
    else:
        for i, doc in enumerate(results):
            print(f"\n[결과 {i+1}]")
            
            # 메타데이터 추출
            source_file = doc.metadata.get('source', 'N/A')
            page_num = doc.metadata.get('page') # get()은 키가 없으면 None을 반환
            row_num = doc.metadata.get('row')

            # 조건에 따라 출처 정보를 다르게 표시
            if page_num is not None:
                # 페이지 번호가 있는 경우 (주로 PDF)
                print(f"  - 출처: {source_file}, 페이지: {page_num + 1}")
            elif row_num is not None:
                # 행 번호가 있는 경우 (주로 CSV)
                print(f"  - 출처: {source_file}, 행: {row_num}")
            else:
                # 둘 다 없는 경우 (주로 HWP)
                print(f"  - 출처: {source_file}")

            print("  - 내용:", doc.page_content)

if __name__ == "__main__":
    main()