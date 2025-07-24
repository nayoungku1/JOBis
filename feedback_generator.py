from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# LLM 로딩 함수
def load_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return AzureChatOpenAI(
        model="gpt-4o",  # 또는 "gpt-4o-mini"
        temperature=0.7,
    )

# 프롬프트 템플릿
feedback_prompt = ChatPromptTemplate.from_template("""
너는 모의면접 평가 전문가야. 아래는 답변 분석 agent가 제공한 분석 결과야 (in json format).

배점 결과 및 기준별 평가:
{parsed_answer}

이 정보를 바탕으로 아래 내용을 출력해줘:

1. 부족한 부분
2. 보완 사항
3. (선택) 잘한 점

각 항목을 Markdown 형식으로 출력해줘. 예시:

## 부족한 부분
- 직무 연관성 강조 부족
- 회사 맞춤 표현 미흡

## 보완 사항
- 직무 기술서 기반으로 경험 연결
- 회사명과 미션 언급으로 관심 표현

## 잘한 점
- 핵심 프로젝트 경험 언급
""")

# 체인 구성
llm = load_llm()
chain: Runnable = feedback_prompt | llm

# 최종 호출 함수
def generate_feedback(parsed_answer: str, score_justification: str) -> str:
    result = chain.invoke({
        "parsed_answer": parsed_answer,
        "score_justification": score_justification
    })
    return result.content