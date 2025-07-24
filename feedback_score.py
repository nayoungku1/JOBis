import os
import json
import traceback
from dotenv import load_dotenv

# [개선점 1] LangChain의 구성 요소를 직접 활용합니다.
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict

# --- 환경 변수 및 클라이언트 초기화 ---
load_dotenv()

# --- [개선점 1] JSON 출력 형식을 Pydantic 모델로 명확하게 정의 ---
class EvaluationScore(BaseModel):
    점수: int = Field(description="1-5점 척도의 점수")
    이유: str = Field(description="점수를 부여한 구체적인 이유")

class Feedback(BaseModel):
    관련성: EvaluationScore
    논리성: EvaluationScore
    진정성: EvaluationScore
    직무적합성: EvaluationScore
    전략적코멘트: str = Field(description="이 질문 유형에 대한 전략적 대응 팁")
    개선피드백: str = Field(description="답변 개선을 위한 구체적인 조언")
    모범답안: str = Field(description="지원자의 배경과 회사 인재상을 반영한 STAR 기법 기반 모범답안 예시")
    참고자료: List[str] = Field(description="웹검색 또는 내부 DB 기반의 참고 정보 요약")

# --- [개선점 4] 전체 로직을 클래스로 캡슐화 ---
class FeedbackAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0.3,
            max_tokens=1500
        )
        self.retriever = self._load_retriever()
        self.web_search = DuckDuckGoSearchRun(region='kr-kr')
        self.parser = JsonOutputParser(pydantic_object=Feedback)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser

    def _load_retriever(self, db_path="faiss_db"):
        if os.path.exists(db_path):
            try:
                embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                return vectorstore.as_retriever(search_kwargs={'k': 3})
            except Exception as e:
                print(f"Warning: Failed to load FAISS DB. RAG will be disabled. Error: {e}")
        return None

    def _create_prompt(self):
        # [개선점 3] 프롬프트를 강화하여 agentA의 심층 분석 결과를 활용하도록 지시
        prompt_template = """
        당신은 기업 인사담당자이자 커리어 전문 코치입니다.
        지원자의 답변을 아래의 모든 정보를 종합하여 심층적으로 평가하고, 전략적 피드백을 제공해야 합니다.
        특히, [회사 및 시장 분석] 정보를 활용하여 지원자의 답변이 비즈니스 맥락을 얼마나 잘 이해하고 있는지 평가하세요.

        **평가 기준:**
        - 관련성: 질문의 핵심 의도에 직접적으로 관련된 내용인가?
        - 논리성: STAR 기법(Situation, Task, Action, Result)에 기반하여 일관되고 구체적인 흐름을 가졌는가?
        - 직무적합성: [회사 및 시장 분석]과 [지원자 정보]를 비교했을 때, 지원자의 경험이 해당 직무와 회사 인재상에 얼마나 부합하는가?
        - 전략적 사고: 답변에 회사의 현재 상황, 경쟁사, 산업 동향에 대한 이해가 녹아있는가?

        {format_instructions}

        ---
        **[분석 대상 정보]**

        [면접 질문]
        {question}

        [지원자 답변]
        {answer}

        ---
        **[참고 컨텍스트]**

        [회사 및 시장 분석 (agentA 결과)]
        {company_analysis}

        [지원자 정보 요약 (personal_info 결과)]
        {personal_info}

        [내부 DB 검색 결과 (질문 기반)]
        {context_from_db}

        [실시간 웹 검색 결과 (질문 기반)]
        {web_context}
        """
        return ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def analyze(self, question: str, answer: str, company_analysis: str = "", personal_info: str = "") -> Dict:
        """
        모든 정보를 종합하여 지원자의 답변을 분석하고 JSON 형식의 피드백을 반환합니다.
        """
        try:
            # [개선점 2] 질문을 기반으로 RAG 및 웹 검색 수행
            context_from_db = ""
            if self.retriever:
                docs = self.retriever.invoke(question)
                context_from_db = "\n\n".join([d.page_content for d in docs])

            web_context = self.web_search.run(f"{company_analysis[:50]} {question}")

            # LCEL 체인 실행
            result = self.chain.invoke({
                "question": question,
                "answer": answer,
                "company_analysis": company_analysis or "제공되지 않음",
                "personal_info": personal_info or "제공되지 않음",
                "context_from_db": context_from_db or "관련 정보 없음",
                "web_context": web_context or "관련 정보 없음",
            })
            return result
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

# --- 단독 실행 테스트용 ---
if __name__ == "__main__":
    # 1. 피드백 에이전트 생성
    feedback_agent = FeedbackAgent()

    # 2. 분석에 필요한 정보 준비
    question = "우리 회사에 지원한 이유는 무엇이며, 입사 후 어떻게 기여하고 싶나요?"
    answer = "저는 귀사의 성장 가능성과 혁신적인 문화에 매료되었습니다. 특히 최근 발표하신 AI 기반 데이터 분석 플랫폼에 깊은 인상을 받았습니다. 저는 이전 프로젝트에서 대용량 데이터를 처리하고 예측 모델을 개발하여 서비스 이탈률을 15% 감소시킨 경험이 있습니다. 이러한 저의 데이터 분석 및 모델링 역량을 활용하여 귀사의 플랫폼을 고도화하고 새로운 가치를 창출하는 데 기여하고 싶습니다."
    
    # agentA가 생성했을 법한 가상의 보고서
    company_info = """
    ## 기업 분석: 글로벌 IT 기업 '퓨처테크'
    - **비전**: AI를 통해 인류의 삶을 혁신한다.
    - **인재상**: 끊임없이 학습하고, 동료와 적극적으로 협업하며, 실패를 두려워하지 않는 도전적인 인재.
    - **최신 동향**: 최근 AI 데이터 분석 플랫폼 '인사이트-X'를 출시했으나, 경쟁사 '데이터-코어'의 유사 서비스 대비 시장 점유율 확보에 어려움을 겪고 있음. 속도와 정확성 개선이 시급한 과제.
    """
    
    # personal_info 모듈이 생성했을 법한 가상의 요약
    personal_info = """
    ## 지원자 정보 요약
    - **경험**: AI 프로젝트 팀장 (3년), 대용량 데이터 처리 및 예측 모델 개발 주도.
    - **기술**: Python, TensorFlow, Scikit-learn, AWS Sagemaker.
    - **성과**: 개발한 모델을 통해 서비스 이탈률 15% 감소.
    """

    # 3. 분석 실행
    result = feedback_agent.analyze(question, answer, company_info, personal_info)

    # 4. 결과 출력
    print(json.dumps(result, ensure_ascii=False, indent=2))
