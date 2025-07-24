import os
from dotenv import load_dotenv

# 기존에 만든 두 개의 핵심 로직을 임포트합니다.
from agentA import run_analyzer
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env 파일 로드
load_dotenv()

def search_internal_db(query: str, k: int = 5) -> list[str]:
    """
    로컬 FAISS DB에서 관련 문서를 검색하여 텍스트 목록으로 반환합니다.
    (RAG_test.py의 핵심 로직)
    """
    print(f"\n--- 내부 DB에서 '{query}'(으)로 유사도 검색 시작 ---")
    db_path = "faiss_db"
    if not os.path.exists(db_path):
        print(" -> FAISS DB를 찾을 수 없습니다. build_faiss_db.py를 먼저 실행하세요.")
        return []

    try:
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        results = vectorstore.similarity_search(query, k=k)
        
        # 검색된 문서의 내용만 추출하여 리스트로 반환
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f" -> 내부 DB 검색 중 오류 발생: {e}")
        return []

def generate_final_questions(external_report: str, internal_snippets: list[str], company_name: str, job_role: str) -> str:
    """
    두 종류의 정보를 바탕으로 최종 면접 질문을 생성합니다.
    """
    print("\n--- 최종 면접 질문 생성 시작 ---")
    
    # 최종 생성을 위한 LLM 초기화
    deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    llm = AzureChatOpenAI(
        azure_deployment=deployment_name,
        temperature=0.7, # 창의적인 질문 생성을 위해 온도를 약간 높임
        max_tokens=2000
    )

    # 두 종류의 컨텍스트를 모두 포함하는 프롬프트
    prompt_template = """
    당신은 [{company_name}]의 [{job_role}] 직무 채용을 위한 최고 수준의 기술 면접관입니다.
    당신의 임무는 아래에 제공된 **두 종류의 심층 분석 자료**를 모두 활용하여, 지원자의 기술적 깊이, 문제 해결 능력, 그리고 회사 문화 적합성을 종합적으로 평가할 수 있는 날카로운 면접 질문 10개를 생성하는 것입니다.

    ---
    **[자료 1: 외부 웹 리서치 기반 기업/시장 분석 보고서]**
    {external_report}
    ---
    **[자료 2: 내부 직무기술서 및 Q&A 데이터베이스 검색 결과]**
    {internal_snippets}
    ---

    **지시사항:**
    1.  두 자료의 내용을 넘나들며 정보를 조합하여 질문을 만드세요.
    2.  단순한 기술 지식 질문을 넘어, 실제 회사 상황과 비즈니스에 대한 이해도를 묻는 질문을 포함하세요.
    3.  지원자의 경험을 구체적으로 이끌어낼 수 있는 행동 기반 질문(BEI)을 포함하세요.
    4.  질문은 "1.", "2.", ... 와 같이 번호가 매겨진 목록 형식으로만 답변해주세요.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | llm
    
    # 내부 자료(snippets)를 보기 좋게 문자열로 변환
    formatted_snippets = "\n\n".join(f"- {snippet}" for snippet in internal_snippets)
    
    response = chain.invoke({
        "company_name": company_name,
        "job_role": job_role,
        "external_report": external_report,
        "internal_snippets": formatted_snippets
    })
    
    return response.content

def main():
    """
    전체 면접 질문 생성 파이프라인을 실행하는 메인 함수.
    """
    # --- 1. 분석 대상 정보 입력 ---
    COMPANY_NAME = "토스"
    JOB_ROLE = "웹 개발자"
    JOB_URL = "https://toss.im/career/job-detail?job_id=6034265003"

    # --- 2. 외부 정보 수집 (웹 리서처 실행) ---
    external_analysis_report = run_analyzer(
        company_name=COMPANY_NAME,
        job_role=JOB_ROLE,
        url=JOB_URL
    )
    print("\n" + "="*50)
    print(" [1/3] 외부 정보 수집 완료")
    print("="*50)

    # --- 3. 내부 정보 수집 (내부 자료 전문가 실행) ---
    # 내부 DB 검색을 위한 쿼리 생성
    internal_query = f"{COMPANY_NAME} {JOB_ROLE} 직무 기술 및 면접 질문"
    internal_search_results = search_internal_db(internal_query, k=5)
    print("\n" + "="*50)
    print(" [2/3] 내부 정보 검색 완료")
    print("="*50)

    # --- 4. 정보 종합 및 최종 질문 생성 (면접관 AI 실행) ---
    final_interview_questions = generate_final_questions(
        external_report=external_analysis_report,
        internal_snippets=internal_search_results,
        company_name=COMPANY_NAME,
        job_role=JOB_ROLE
    )
    print("\n" + "="*50)
    print(" [3/3] 최종 면접 질문 생성 완료")
    print("="*50)
    print(final_interview_questions)


if __name__ == "__main__":
    main()
