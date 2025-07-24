import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# [추가] 개인 문서 로더 및 프롬프트 템플릿 임포트
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# --- 1. 세분화된 메모리 구조 정의 ---
class PersonalContext(BaseModel):
    uploaded_files: List[str] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)

class CompanyContext(BaseModel):
    analysis_report: Optional[str] = Field(default=None)

class InterviewSession(BaseModel):
    generated_questions: List[str] = Field(default_factory=list)
    chat_history: List[dict] = Field(default_factory=list)

class MemoryHub(BaseModel):
    personal_context: PersonalContext = Field(default_factory=PersonalContext)
    company_context: CompanyContext = Field(default_factory=CompanyContext)
    interview_session: InterviewSession = Field(default_factory=InterviewSession)


# --- 2. 챗봇 핵심 로직 클래스 ---
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

    def add_company_analysis(self, report: str):
        self.memory.company_context.analysis_report = report

    # --- [추가된 기능] 개인 문서 처리 및 요약 ---
    def process_personal_documents(self, uploaded_files: List[Any], job_description: str):
        if not uploaded_files:
            return
        
        print("--- 개인 문서 처리 및 요약 시작 ---")
        # 1. 파일 텍스트 로드
        combined_text = self._load_personal_docs_text(uploaded_files)
        
        # 2. LLM을 이용해 정보 추출 및 요약
        summary = self._extract_relevant_info(combined_text, job_description)
        
        # 3. 메모리에 저장
        self.memory.personal_context.summary = summary
        self.memory.personal_context.uploaded_files = [file.name for file in uploaded_files]
        print("--- 개인 문서 요약 완료 및 메모리 저장 ---")

    def _load_personal_docs_text(self, files: List[Any]) -> str:
        """업로드된 PDF/DOCX 파일들의 텍스트를 읽어 하나의 문자열로 합칩니다."""
        temp_dir = "temp_personal_docs"
        os.makedirs(temp_dir, exist_ok=True)
        
        all_texts = []
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            else:
                continue
            
            docs = loader.load()
            all_texts.append(" ".join([doc.page_content for doc in docs]))
        
        # 임시 파일 및 폴더 정리
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return "\n\n".join(all_texts)

    def _extract_relevant_info(self, doc_text: str, job_desc: str) -> str:
        """LLM을 사용해 문서 텍스트에서 직무와 관련된 정보를 추출/요약합니다."""
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in extracting relevant information from an individual's resume and CV.
            Your task is to extract information from the provided document text that is relevant to the given job description.
            Output the result in Markdown format with sections: Education, Experience, Skills, Certifications, Projects.
            Extract ONLY explicitly mentioned information. Do NOT infer or generate information.
            If a section has no relevant information, include the section header with no bullet points.

            Job Description: {job_description}
            Document Text: {document_text}
            """
        )
        chain = prompt | self.llm
        result = chain.invoke({"job_description": job_desc, "document_text": doc_text})
        return result.content if hasattr(result, 'content') else str(result)
    # --- [여기까지 추가된 기능] ---

    def generate_interview_questions(self):
        if not self.memory.company_context.analysis_report: return
        print("--- 개인 맞춤 면접 질문 생성 시작 ---")
        system_prompt = "당신은 최고 수준의 기술 면접관입니다. 주어진 [기업 분석 보고서]와 [지원자 정보 요약]을 모두 참고하여, 지원자의 경험과 회사의 요구사항을 연결하는 날카로운 면접 질문 10개를 생성해주세요. 질문만 목록 형식으로 답변해주세요."
        
        human_content = f"""
        [기업 분석 보고서]:
        {self.memory.company_context.analysis_report}

        [지원자 정보 요약]:
        {self.memory.personal_context.summary or "제공되지 않음"}
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        response = self.llm.invoke(messages)
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and (q.strip()[0].isdigit() or q.strip()[0] == '-')]
        self.memory.interview_session.generated_questions = questions
        print(f"생성된 면접 질문: {questions}")

    def get_response(self, user_input: str) -> str:
        self.memory.interview_session.chat_history.append({"role": "user", "content": user_input})
        context_from_db = ""
        if self.retriever:
            relevant_docs = self.retriever.invoke(user_input)
            context_from_db = "\n\n".join([f"[출처: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in relevant_docs])
        
        system_prompt = "당신은 AI 면접 코치입니다. 당신의 임무는 아래에 제공된 모든 정보를 종합하여, 사용자의 질문이나 답변에 대해 상세하고 도움이 되는 피드백을 제공하는 것입니다."
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for msg in self.memory.interview_session.chat_history:
            messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        
        final_user_message = f"""
        [사용자 현재 메시지]: {user_input}
        ---
        [참고 자료]
        [기업 분석 보고서]: {self.memory.company_context.analysis_report or "아직 분석되지 않음"}
        [지원자 정보 요약]: {self.memory.personal_context.summary or "제공되지 않음"}
        [내부 DB 검색 결과]: {context_from_db or "관련 정보 없음"}
        """
        messages[-1] = HumanMessage(content=final_user_message)
        ai_response = self.llm.invoke(messages).content
        self.memory.interview_session.chat_history.append({"role": "assistant", "content": ai_response})
        return ai_response
