import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

class MemoryHub(BaseModel):
    personal_info_summary: Optional[str] = Field(default=None)
    company_analysis_report: Optional[str] = Field(default=None)
    interview_questions: List[str] = Field(default_factory=list)
    chat_history: List[dict] = Field(default_factory=list)

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
        self.memory.company_analysis_report = report
        self._generate_interview_questions()

    def _generate_interview_questions(self):
        if not self.memory.company_analysis_report: return
        print("--- 면접 질문 생성 시작 ---")
        system_prompt = "당신은 최고 수준의 기술 면접관입니다. 주어진 기업 분석 보고서를 바탕으로, 지원자의 역량을 검증할 수 있는 날카로운 면접 질문 10개를 생성해주세요. 질문만 번호가 매겨진 목록 형식으로 답변해주세요."
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=self.memory.company_analysis_report)]
        response = self.llm.invoke(messages)
        questions = [q.strip() for q in response.content.split('\n') if q.strip() and (q.strip()[0].isdigit() or q.strip()[0] == '-')]
        self.memory.interview_questions = questions
        print(f"생성된 면접 질문: {questions}")

    def get_response(self, user_input: str) -> str:
        self.memory.chat_history.append({"role": "user", "content": user_input})
        context_from_db = ""
        if self.retriever:
            print(f"--- RAG 검색 수행: '{user_input}' ---")
            relevant_docs = self.retriever.invoke(user_input)
            context_from_db = "\n\n".join([f"[출처: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in relevant_docs])
            print("--- RAG 검색 완료 ---")
        system_prompt = """
        당신은 AI 면접 코치입니다. 당신의 임무는 아래에 제공된 모든 정보를 종합하여, 사용자의 질문이나 답변에 대해 상세하고 도움이 되는 피드백을 제공하는 것입니다.
        - 사용자의 이전 대화 내용을 기억하고, 대화의 흐름을 유지하세요.
        - 사용자가 면접 질문에 답변하는 경우, [기업 분석 보고서], [내부 DB 검색 결과] 등을 총동원하여 답변을 평가하고 구체적인 피드백을 주세요.
        - 답변은 항상 한글로, 친절하고 전문적인 말투를 사용하세요.
        """
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for msg in self.memory.chat_history:
            messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        final_user_message = f"""
        [사용자 현재 메시지]: {user_input}
        ---
        [참고 자료 1: 외부 웹 리서치 기반 기업 분석 보고서]: 
        {self.memory.company_analysis_report or "아직 분석되지 않음"}
        ---
        [참고 자료 2: 내부 문서 DB 검색 결과]: 
        {context_from_db or "관련 정보 없음"}
        """
        messages[-1] = HumanMessage(content=final_user_message)
        ai_response = self.llm.invoke(messages).content
        self.memory.chat_history.append({"role": "assistant", "content": ai_response})
        return ai_response
