import os
import json
import re
import traceback
from dotenv import load_dotenv
from openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
# --- 환경 변수 로드 ---
# load_dotenv()
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --- 클라이언트 초기화 ---
client = AzureChatOpenAI(
)

# --- RAG: 벡터 DB 로딩 ---
def load_retriever(db_path="faiss_db"):
    if os.path.exists(db_path):
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={'k': 5})
    return None

retriever = load_retriever()
web_search = DuckDuckGoSearchRun(region='kr-kr')

# --- 분석 에이전트 (업그레이드 프롬프트 포함) ---
def analyze_answer_with_agent(question, answer, company_analysis="", personal_info="", chat_history=""):
    """
    개인 데이터 + 기업 분석 기반 맞춤형 면접 답변 평가 에이전트 (고도화 버전)
    """
    # 1. 내부 DB 검색
    context_from_db = ""
    if retriever:
        docs = retriever.invoke(answer)
        context_from_db = "\n\n".join([d.page_content for d in docs])

    # 2. 웹 검색
    web_context = web_search.run(f"{question} {company_analysis[:50]}")

    # 3. 프롬프트 구성
    prompt = f"""
    당신은 기업 인사담당자이자 커리어 전문 코치입니다.
    지원자의 답변을 아래 기준에 따라 심층 평가하고 전략적 피드백을 제공합니다.

    평가 기준:
    - 관련성: 질문과 직접적으로 관련된 내용인가?
    - 논리성: 일관되고 구체적인 흐름인가?
    - 진정성: 진심과 의도가 설득력 있게 느껴지는가?
    - 직무적합성: 해당 직무와 회사 인재상에 잘 부합하는가?

    출력은 아래 JSON 형식으로 작성하세요:
    {{
      "관련성": {{"점수": X, "이유": "설명"}},
      "논리성": {{"점수": X, "이유": "설명"}},
      "진정성": {{"점수": X, "이유": "설명"}},
      "직무적합성": {{"점수": X, "이유": "설명"}},
      "전략적코멘트": "이 질문 유형에 대한 전략적 대응 팁",
      "개선피드백": "답변 개선을 위한 구체적 조언",
      "모범답안": "지원자의 배경과 회사 인재상을 반영한 STAR 기법 기반 모범답안 예시",
      "참고자료": ["웹검색 또는 내부 DB 기반의 참고 정보 요약"]
    }}

    [질문]
    {question}

    [답변]
    {answer}

    [회사 및 직무 분석]
    {company_analysis or "없음"}

    [지원자 정보 요약]
    {personal_info or "없음"}

    [이전 대화 요약]
    {chat_history or "없음"}

    [내부 DB 검색 결과]
    {context_from_db or "없음"}

    [웹 검색 결과]
    {web_context or "없음"}
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "너는 HR 전문가이자 커리어 코치야. JSON 형식으로만 출력해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r"\{[\s\S]*\}", content)
        content_clean = json_match.group() if json_match else content
        return json.loads(content_clean)
    except json.JSONDecodeError:
        return {"raw_output": content}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# --- 단독 실행 테스트용 ---
if __name__ == "__main__":
    question = "우리 회사에 지원한 이유는 무엇인가요?"
    answer = "저는 이 회사가 성장 가능성이 높고 제 역량을 발휘할 수 있는 기회가 많다고 생각했습니다."
    company_info = "이 회사는 혁신 기술을 주도하고, 도전과 협업을 인재상으로 강조하는 글로벌 IT 기업입니다."
    personal_info = "저는 AI 프로젝트에서 팀장을 맡아 실제 서비스로 연결되는 시스템을 개발한 경험이 있습니다."
    result = analyze_answer_with_agent(question, answer, company_info, personal_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))
