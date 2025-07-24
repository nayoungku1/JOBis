import streamlit as st
import os
from dotenv import load_dotenv
from chatbot_core import ChatbotCore, MemoryHub
from agentA import run_analyzer
from build_faiss_db import build_or_update_vector_db

load_dotenv()
st.set_page_config(page_title="AI 면접 코치", layout="wide")
st.title("AI 면접 코치 🤖")

# 세션 상태 초기화
if "memory_hub" not in st.session_state:
    st.session_state["memory_hub"] = MemoryHub(
        interview_session={"chat_history": [{"role": "assistant", "content": "안녕하세요! 먼저 사이드바에 정보를 입력하고 자료를 업로드 해주세요."}]}
    )
chatbot = ChatbotCore(memory=st.session_state.memory_hub)

with st.sidebar:
    st.header("1️⃣ 분석 대상 정보")
    company_name = st.text_input("회사명", key="company_name")
    job_role = st.text_input("희망 직무", key="job_role")
    job_url = st.text_input("채용 공고 URL (선택 사항)", key="job_url")

    st.divider()

    st.header("2️⃣ 개인 맞춤 자료")
    personal_files = st.file_uploader(
        "자기소개서, 이력서 등 업로드 (.pdf, .docx)",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        key="personal_files_uploader"
    )

    st.divider()

    if st.button("🚀 면접 준비 시작!", use_container_width=True, type="primary"):
        if not st.session_state.company_name or not st.session_state.job_role:
            st.warning("회사명과 희망 직무를 반드시 입력해주세요.")
        elif not personal_files:
            st.warning("개인 맞춤 분석을 위해 하나 이상의 개인 파일을 업로드해주세요.")
        else:
            with st.spinner("1/3 | 최신 기업 및 시장 정보를 분석 중입니다..."):
                try:
                    report = run_analyzer(
                        company_name=st.session_state.company_name,
                        job_role=st.session_state.job_role,
                        url=st.session_state.job_url
                    )
                    chatbot.add_company_analysis(report)
                except Exception as e:
                    st.error(f"기업 분석 중 오류 발생: {e}")
                    st.stop()
            
            with st.spinner("2/3 | 업로드된 개인 문서를 분석하고 있습니다..."):
                try:
                    job_description_from_report = report 
                    chatbot.process_personal_documents(personal_files, job_description_from_report)
                except Exception as e:
                    st.error(f"개인 문서 처리 중 오류 발생: {e}")
                    st.stop()

            with st.spinner("3/3 | 모든 정보를 종합하여 맞춤 면접 질문을 생성합니다..."):
                try:
                    chatbot.generate_interview_questions()
                    initial_message = (
                        f"✅ **'{st.session_state.company_name}'({st.session_state.job_role})** 직무에 대한 모든 준비가 완료되었습니다!\n\n"
                        "**생성된 맞춤 면접 질문:**\n"
                    )
                    # for i, q in enumerate(chatbot.memory.interview_session.generated_questions[:5]):
                    #     initial_message += f"- {q}\n"
                    initial_message += "\n면접을 시작하시겠습니까? '시작할게'라고 입력해주세요."
                    st.session_state.memory_hub.interview_session.chat_history = [
                        {"role": "assistant", "content": initial_message}
                    ]  # 초기화하여 중복 방지
                except Exception as e:
                    st.error(f"질문 생성 중 오류 발생: {e}")
                    st.stop()
            st.rerun()

    st.divider()
    st.header("📚 내부 DB 관리")
    db_files = st.file_uploader(
        "내부 지식 DB용 파일 업로드 (.pdf, .hwp, .csv, .zip 등)",
        accept_multiple_files=True,
        key="db_files_uploader"
    )
    if st.button("내부 DB 업데이트", use_container_width=True):
        if db_files:
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            with st.spinner('파일을 저장하고 벡터 DB를 업데이트합니다...'):
                for file in db_files:
                    with open(os.path.join(data_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())
                try:
                    build_or_update_vector_db()
                    st.success(f"{len(db_files)}개 파일로 DB 업데이트 완료!")
                except Exception as e:
                    st.error(f"DB 업데이트 중 오류 발생: {e}")
        else:
            st.warning("업로드할 파일이 없습니다.")

# --- 메인 채팅 인터페이스 ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.memory_hub.interview_session.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if user_input := st.chat_input("면접 질문에 답변하거나 자유롭게 질문해보세요..."):
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
    with st.spinner("답변을 생성하는 중입니다..."):
        try:
            # 중복 메시지 방지를 위해 chat_history를 직접 수정
            ai_response = chatbot.get_response(user_input)
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
        except Exception as e:
            with chat_container:
                with st.chat_message("assistant"):
                    st.error(f"응답 생성 중 오류 발생: {e}")
    st.rerun()