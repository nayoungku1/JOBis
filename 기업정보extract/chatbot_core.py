import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional

# .env 파일 로드
load_dotenv()

# --- 1. 메모리 구조 정의 ---
# Pydantic 모델을 사용하여 각 메모리 유형을 명확하게 정의합니다.
class MemoryHub(BaseModel):
    """챗봇의 모든 메모리를 구조적으로 관리하는 클래스"""
    # 1. 개인 신상 정보 (자주 바뀌지 않음)
    personal_info_summary: Optional[str] = Field(
        default=None, description="사용자의 이력서, 자소서 등을 요약한 내용"
    )
    # 2. 기업 및 직무 분석 정보 (분석 시마다 바뀜)
    company_analysis_report: Optional[str] = Field(
        default=None, description="agentA가 생성한 기업/직무/시장 분석 보고서"
    )
    # 3. 생성된 면접 질문 (면접 시뮬레이션 시 사용)
    interview_questions: List[str] = Field(
        default_factory=list, description="AI가 생성한 면접 질문 목록"
    )
    # 4. 실시간 대화 기록 (계속 누적됨)
    chat_history: List[dict] = Field(
        default_factory=list, description="사용자와 AI 간의 대화 기록"
    )

# --- 2. 챗봇 핵심 로직 클래스 ---

class ChatbotCore:
    # [변경점] __init__ 메서드가 Streamlit의 session_state 대신 MemoryHub 객체를 직접 받도록 수정
    def __init__(self, memory: MemoryHub):
        self.memory = memory

        # LLM 초기화
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=2000
        )
        
        # 벡터 DB 검색을 위한 Retriever 초기화
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self):
        """FAISS DB를 로드하여 Retriever를 준비합니다."""
        db_path = "faiss_db"
        if os.path.exists(db_path):
            try:
                embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                return vectorstore.as_retriever(search_kwargs={'k': 5})
            except Exception as e:
                # [변경점] st.error 대신 print를 사용하여 UI와 로직을 분리
                print(f"오류: FAISS DB 로드 실패: {e}")
                return None
        return None

    def add_company_analysis(self, report: str):
        """기업 분석 보고서를 메모리에 저장합니다."""
        self.memory.company_analysis_report = report
        # 보고서를 바탕으로 면접 질문을 생성하여 저장
        self._generate_interview_questions()

    def add_personal_info(self, summary: str):
        """사용자 개인 정보 요약을 메모리에 저장합니다."""
        self.memory.personal_info_summary = summary

    def _generate_interview_questions(self):
        """기업 분석 보고서와 개인 정보를 바탕으로 면접 질문을 생성합니다."""
        if not self.memory.company_analysis_report:
            return

        print("--- 면접 질문 생성 시작 ---")
        system_prompt = "당신은 최고 수준의 기술 면접관입니다. 주어진 기업 분석 보고서와 지원자 정보를 바탕으로, 지원자의 역량을 검증할 수 있는 날카로운 면접 질문 10개를 생성해주세요. 질문만 목록 형식으로 답변해주세요."
        
        human_content = f"""
        [기업 분석 보고서]
        {self.memory.company_analysis_report}

        [지원자 정보 요약]
        {self.memory.personal_info_summary or "제공되지 않음"}
        """
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        response = self.llm.invoke(messages)
        
        # 생성된 질문을 파싱하여 리스트로 저장
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and (q.strip()[0].isdigit() or q.strip()[0] == '-')]
        self.memory.interview_questions = questions
        print(f"생성된 면접 질문: {questions}")


    def get_response(self, user_input: str) -> str:
        """사용자 입력에 대한 AI의 최종 응답을 생성합니다."""
        self.memory.chat_history.append({"role": "user", "content": user_input})

        # 1. 내부 DB에서 관련 정보 검색 (RAG)
        context_from_db = ""
        if self.retriever:
            relevant_docs = self.retriever.invoke(user_input)
            context_from_db = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 2. 상황에 맞는 프롬프트 구성
        system_prompt = """
        당신은 AI 면접 코치입니다. 당신의 임무는 아래에 제공된 모든 정보를 종합하여, 사용자의 질문이나 답변에 대해 상세하고 도움이 되는 피드백을 제공하는 것입니다.
        - 사용자의 이전 대화 내용을 기억하고, 대화의 흐름을 유지하세요.
        - 사용자가 면접 질문에 답변하는 경우, [기업 분석 보고서], [지원자 정보], [내부 DB 검색 결과]를 총동원하여 답변을 평가하고 구체적인 피드백을 주세요.
        """
        
        # LangChain 메시지 형식으로 변환
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for msg in self.memory.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 마지막 HumanMessage에 컨텍스트 추가
        final_user_message = f"""
        [사용자 현재 메시지]: {user_input}

        ---
        [참고 자료]
        
        [기업 분석 보고서]: {self.memory.company_analysis_report or "아직 분석되지 않음"}
        
        [생성된 면접 질문 목록]: {self.memory.interview_questions or "아직 생성되지 않음"}

        [지원자 정보 요약]: {self.memory.personal_info_summary or "제공되지 않음"}
        
        [내부 DB 검색 결과]: {context_from_db or "관련 정보 없음"}
        """
        messages[-1] = HumanMessage(content=final_user_message) # 마지막 메시지를 컨텍스트가 포함된 메시지로 교체

        # 3. LLM 호출
        ai_response = self.llm.invoke(messages).content
        self.memory.chat_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
