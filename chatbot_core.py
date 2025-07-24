import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# --- 1. 세분화된 메모리 구조 정의 ---

class PersonalContext(BaseModel):
    """사용자의 개인 정보를 저장하는 모델"""
    uploaded_files: List[str] = Field(
        default_factory=list, description="사용자가 업로드한 개인 파일 목록 (예: 이력서, 자소서)"
    )
    summary: Optional[str] = Field(
        default=None, description="업로드된 개인 파일을 요약한 내용"
    )

class CompanyContext(BaseModel):
    """분석 대상 기업 정보를 저장하는 모델"""
    uploaded_files: List[str] = Field(
        default_factory=list, description="사용자가 업로드한 특정 기업 관련 파일 목록"
    )
    analysis_report: Optional[str] = Field(
        default=None, description="agentA가 생성한 기업/직무/시장 분석 보고서"
    )

class InterviewSession(BaseModel):
    """면접 시뮬레이션 관련 정보를 저장하는 모델"""
    generated_questions: List[str] = Field(
        default_factory=list, description="AI가 생성한 면접 질문 목록"
    )
    chat_history: List[dict] = Field(
        default_factory=list, description="사용자와 AI 간의 대화 기록"
    )

class MemoryHub(BaseModel):
    """챗봇의 모든 메모리를 구조적으로 관리하는 최상위 클래스"""
    # 1. 개인 관련 업로드 파일 (이름 목록)
    # 2. 개인 관련 summary
    personal_context: PersonalContext = Field(default_factory=PersonalContext)
    
    # 3. 회사 관련 업로드 파일 (이름 목록)
    # 4. 회사 관련 report
    company_context: CompanyContext = Field(default_factory=CompanyContext)
    
    # 5. 생성된 질문 목록
    # 6. 대화 내역
    interview_session: InterviewSession = Field(default_factory=InterviewSession)


# --- 2. 챗봇 핵심 로직 클래스 (업데이트된 메모리 구조 사용) ---

class ChatbotCore:
    def __init__(self, memory: MemoryHub):
        self.memory = memory
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.7, max_tokens=2000
        )
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self):
        db_path = "faiss_db"
        if os.path.exists(db_path):
            try:
                embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                return vectorstore.as_retriever(search_kwargs={'k': 5})
            except Exception as e:
                print(f"오류: FAISS DB 로드 실패: {e}")
                return None
        return None

    def add_company_analysis(self, report: str):
        """기업 분석 보고서를 메모리에 저장합니다."""
        self.memory.company_context.analysis_report = report
        self._generate_interview_questions()

    def add_personal_info(self, summary: str, filenames: List[str]):
        """사용자 개인 정보 요약 및 파일 목록을 메모리에 저장합니다."""
        self.memory.personal_context.summary = summary
        self.memory.personal_context.uploaded_files = filenames

    def _generate_interview_questions(self):
        """기업 분석 보고서와 개인 정보를 바탕으로 면접 질문을 생성합니다."""
        if not self.memory.company_context.analysis_report: return
        print("--- 면접 질문 생성 시작 ---")
        system_prompt = "당신은 최고 수준의 기술 면접관입니다. 주어진 기업 분석 보고서와 지원자 정보를 바탕으로, 지원자의 역량을 검증할 수 있는 날카로운 면접 질문 10개를 생성해주세요. 질문만 번호가 매겨진 목록 형식으로 답변해주세요."
        
        human_content = f"""
        [기업 분석 보고서]
        {self.memory.company_context.analysis_report}

        [지원자 정보 요약]
        {self.memory.personal_context.summary or "제공되지 않음"}
        """
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        response = self.llm.invoke(messages)
        
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and (q.strip()[0].isdigit() or q.strip()[0] == '-')]
        self.memory.interview_session.generated_questions = questions
        print(f"생성된 면접 질문: {questions}")

    def get_response(self, user_input: str) -> str:
        """사용자 입력에 대한 AI의 최종 응답을 생성합니다."""
        self.memory.interview_session.chat_history.append({"role": "user", "content": user_input})

        context_from_db = ""
        if self.retriever:
            print(f"--- RAG 검색 수행: '{user_input}' ---")
            relevant_docs = self.retriever.invoke(user_input)
            context_from_db = "\n\n".join([f"[출처: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in relevant_docs])
            print("--- RAG 검색 완료 ---")

        system_prompt = """
        당신은 AI 면접 코치입니다. 당신의 임무는 아래에 제공된 모든 정보를 종합하여, 사용자의 질문이나 답변에 대해 상세하고 도움이 되는 피드백을 제공하는 것입니다.
        - 사용자의 이전 대화 내용을 기억하고, 대화의 흐름을 유지하세요.
        - 사용자가 면접 질문에 답변하는 경우, [기업 분석 보고서], [지원자 정보], [내부 DB 검색 결과]를 총동원하여 답변을 평가하고 구체적인 피드백을 주세요.
        - 답변은 항상 한글로, 친절하고 전문적인 말투를 사용하세요.
        """
        
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for msg in self.memory.interview_session.chat_history:
            messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        
        final_user_message = f"""
        [사용자 현재 메시지]: {user_input}
        ---
        [참고 자료]
        
        [기업 분석 보고서]: {self.memory.company_context.analysis_report or "아직 분석되지 않음"}
        
        [생성된 면접 질문 목록]: {self.memory.interview_session.generated_questions or "아직 생성되지 않음"}

        [지원자 정보 요약]: {self.memory.personal_context.summary or "제공되지 않음"}
        
        [내부 DB 검색 결과]: {context_from_db or "관련 정보 없음"}
        """
        messages[-1] = HumanMessage(content=final_user_message)

        ai_response = self.llm.invoke(messages).content
        self.memory.interview_session.chat_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
