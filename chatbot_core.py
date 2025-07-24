import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from feedback_score import FeedbackAgent

load_dotenv()

# --- 메모리 구조 정의 ---
class PersonalContext(BaseModel):
    uploaded_files: List[str] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)

class CompanyContext(BaseModel):
    analysis_report: Optional[str] = Field(default=None)

class InterviewSession(BaseModel):
    generated_questions: List[str] = Field(default_factory=list)
    chat_history: List[dict] = Field(default_factory=list)
    asked_questions: List[str] = Field(default_factory=list)
    current_question: Optional[str] = Field(default=None)
    interview_started: bool = Field(default=False)

class MemoryHub(BaseModel):
    personal_context: PersonalContext = Field(default_factory=PersonalContext)
    company_context: CompanyContext = Field(default_factory=CompanyContext)
    interview_session: InterviewSession = Field(default_factory=InterviewSession)

# --- 챗봇 핵심 로직 클래스 ---
class ChatbotCore:
    def __init__(self, memory: MemoryHub):
        self.memory = memory
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.7, max_tokens=2000
        )
        self.retriever = self._initialize_retriever()
        self.feedback_agent = FeedbackAgent()

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

    def process_personal_documents(self, uploaded_files: List[Any], job_description: str):
        if not uploaded_files:
            return
        print("--- 개인 문서 처리 및 요약 시작 ---")
        combined_text = self._load_personal_docs_text(uploaded_files)
        summary = self._extract_relevant_info(combined_text, job_description)
        self.memory.personal_context.summary = summary
        self.memory.personal_context.uploaded_files = [file.name for file in uploaded_files]
        print("--- 개인 문서 요약 완료 및 메모리 저장 ---")

    def _load_personal_docs_text(self, files: List[Any]) -> str:
        temp_dir = "temp_personal_docs"
        os.makedirs(temp_dir, exist_ok=True)
        all_texts = []
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            else:
                continue
            docs = loader.load()
            all_texts.append(" ".join([doc.page_content for doc in docs]))
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        return "\n\n".join(all_texts)

    def _extract_relevant_info(self, doc_text: str, job_desc: str) -> str:
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

    def generate_interview_questions(self):
        if not self.memory.company_context.analysis_report:
            return
        print("--- 개인 맞춤 면접 질문 생성 시작 ---")
        system_prompt = """
        당신은 최고 수준의 기술 면접관입니다. 주어진 [기업 분석 보고서]와 [지원자 정보 요약]을 모두 참고하여,
        지원자의 경험과 회사의 요구사항을 연결하는 날카로운 면접 질문 10개를 생성해주세요.
        질문은 지원자의 기술적 역량, 경험, 회사 인재상과 직무 적합성을 평가할 수 있도록 설계해야 합니다.
        질문만 목록 형식으로 답변해주세요.
        """
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

    def start_interview(self) -> str:
        self.memory.interview_session.interview_started = True
        if not self.memory.interview_session.generated_questions:
            response = "면접 질문을 먼저 생성해주세요."
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            return response
        available_questions = [q for q in self.memory.interview_session.generated_questions 
                            if q not in self.memory.interview_session.asked_questions]
        if not available_questions:
            response = "더 이상 새로운 질문이 없습니다. 다른 주제로 질문을 생성하시겠습니까?"
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            return response
        self.memory.interview_session.current_question = available_questions[0]
        self.memory.interview_session.asked_questions.append(available_questions[0])
        response = f"첫 번째 질문: {self.memory.interview_session.current_question}"
        self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
        return response

    def get_response(self, user_input: str) -> str:
        print(f"Processing user input: {user_input}")  # 디버깅 로그
        self.memory.interview_session.chat_history.append({"role": "user", "content": user_input})

        # 면접 시작 여부 확인
        if not self.memory.interview_session.interview_started:
            if user_input.lower() in ["시작할게", "시작", "start"]:
                return self.start_interview()
            response = "면접을 시작하시겠습니까? '시작할게'라고 입력해주세요."
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            print(f"Generated response: {response}")
            return response

        # 사용자 입력 처리
        normalized_input = user_input.strip().lower()
        if normalized_input in ["1", "다시 답변", "다시 답변하기"]:
            response = f"같은 질문: {self.memory.interview_session.current_question}"
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            print(f"Generated response: {response}")
            return response
        elif normalized_input in ["2", "심화 질문", "심화 질문 받기"]:
            response = self._generate_followup_question()
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            print(f"Generated response: {response}")
            return response
        elif normalized_input in ["3", "다른 질문", "다른 질문 받기"]:
            response = self._get_next_question()
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            print(f"Generated response: {response}")
            return response
        else:
            # 답변으로 간주하고 피드백 생성
            feedback = self._generate_feedback(self.memory.interview_session.current_question, user_input)
            options_prompt = (
                "\n\n다음 중 하나를 선택해주세요:\n"
                "1. 같은 질문에 대해 다시 답변하기\n"
                "2. 같은 주제에 대한 심화 질문 받기\n"
                "3. 다른 질문 받기"
            )
            response = f"{feedback}\n{options_prompt}"
            self.memory.interview_session.chat_history.append({"role": "assistant", "content": response})
            print(f"Generated response: {response}")
            return response

    def _generate_feedback(self, question: str, answer: str) -> str:
        print(f"Generating feedback for question: {question}, answer: {answer}")  # 디버깅 로그
        feedback_result = self.feedback_agent.analyze(
            question=question,
            answer=answer,
            company_analysis=self.memory.company_context.analysis_report or "제공되지 않음",
            personal_info=self.memory.personal_context.summary or "제공되지 않음"
        )
        if "error" in feedback_result:
            return f"피드백 생성 중 오류 발생: {feedback_result['error']}"
        
        feedback_text = (
            f"### 피드백\n"
            f"- **관련성**: {feedback_result['관련성']['점수']}/5 - {feedback_result['관련성']['이유']}\n\n"
            f"- **논리성**: {feedback_result['논리성']['점수']}/5 - {feedback_result['논리성']['이유']}\n\n"
            f"- **진정성**: {feedback_result['진정성']['점수']}/5 - {feedback_result['진정성']['이유']}\n\n"
            f"- **직무적합성**: {feedback_result['직무적합성']['점수']}/5 - {feedback_result['직무적합성']['이유']}\n\n"
            f"\n**전략적 코멘트**: {feedback_result['전략적코멘트']}\n"
            f"**개선 피드백**: {feedback_result['개선피드백']}\n"
            f"**모범 답안 예시**: {feedback_result['모범답안']}\n"
            f"**참고 자료**: {', '.join(feedback_result['참고자료']) or '없음'}"
        )
        return feedback_text

    def _generate_followup_question(self) -> str:
        print("Generating followup question...")  # 디버깅 로그
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 기술 면접관입니다. 주어진 질문과 사용자의 답변을 바탕으로,
            같은 주제에 대한 심화 질문을 하나 생성해주세요.
            질문은 지원자의 기술적 깊이, 문제 해결 능력, 또는 회사와의 적합성을 더 깊게 평가할 수 있도록 설계해야 합니다.
            기존에 사용된 질문과 중복되지 않도록 주의하세요.

            현재 질문: {current_question}
            사용자 답변: {user_answer}
            기존 질문들: {asked_questions}
            [기업 분석 보고서]: {company_analysis}
            [지원자 정보 요약]: {personal_info}
            """
        )
        chain = prompt | self.llm
        try:
            result = chain.invoke({
                "current_question": self.memory.interview_session.current_question,
                "user_answer": self.memory.interview_session.chat_history[-1]["content"],
                "asked_questions": ", ".join(self.memory.interview_session.asked_questions),
                "company_analysis": self.memory.company_context.analysis_report or "제공되지 않음",
                "personal_info": self.memory.personal_context.summary or "제공되지 않음"
            })
            new_question = result.content.strip()
            if new_question in self.memory.interview_session.asked_questions:
                return "심화 질문 생성 실패: 중복된 질문입니다. 다른 옵션을 선택해주세요."
            self.memory.interview_session.asked_questions.append(new_question)
            self.memory.interview_session.current_question = new_question
            return f"심화 질문: {new_question}"
        except Exception as e:
            print(f"Followup question generation error: {e}")
            return f"심화 질문 생성 중 오류 발생: {e}"

    def _get_next_question(self) -> str:
        print("Fetching next question...")  # 디버깅 로그
        available_questions = [q for q in self.memory.interview_session.generated_questions 
                            if q not in self.memory.interview_session.asked_questions]
        if not available_questions:
            return "더 이상 새로운 질문이 없습니다. 다른 주제로 질문을 생성하시겠습니까?"
        self.memory.interview_session.current_question = available_questions[0]
        self.memory.interview_session.asked_questions.append(available_questions[0])
        return f"다음 질문: {available_questions[0]}"