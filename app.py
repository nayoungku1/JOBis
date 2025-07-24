import streamlit as st
import os
from dotenv import load_dotenv

# [변경점] ChatbotCore와 MemoryHub를 모두 임포트합니다.
from chatbot_core import ChatbotCore, MemoryHub 
# from agentA import run_analyzer # 최종적으로 이 함수를 호출하게 됩니다.
from langchain.schema import HumanMessage, AIMessage
import time

# --- 1. 페이지 설정 및 초기화 ---
load_dotenv()
st.set_page_config(page_title="AI 면접 코치", layout="wide")
st.title("AI 면접 코치 🤖")

# --- [핵심 변경점] 세션 상태 초기화 로직 수정 ---
# 세션이 시작될 때, 메모리 허브를 먼저 생성합니다.
if "memory_hub" not in st.session_state:
    st.session_state["memory_hub"] = MemoryHub()

# 메모리 허브를 사용하여 챗봇 코어 인스턴스를 생성합니다.
# 이 인스턴스는 메모리 허브를 참조하므로, 세션 내내 동일한 메모리를 사용하게 됩니다.
chatbot = ChatbotCore(memory=st.session_state.memory_hub)


# --- 2. 사이드바 (입력 및 파일 업로드) ---
with st.sidebar:
    st.header("🎯 분석 대상 정보")
    company_name = st.text_input("회사명", key="company_name")
    job_role = st.text_input("희망 직무", key="job_role")
    job_url = st.text_input("채용 공고 URL (선택 사항)", key="job_url")

    if st.button("✨ 기업/직무 분석 시작", use_container_width=True, type="primary"):
        if not st.session_state.company_name or not st.session_state.job_role:
            st.warning("회사명과 희망 직무를 반드시 입력해주세요.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("웹 리서치 및 심층 분석을 진행 중입니다..."):
                    # agentA의 분석 결과를 ChatbotCore의 메모리에 저장
                    # report = run_analyzer(...) 
                    time.sleep(3)
                    report = f"## {st.session_state.company_name} 분석 보고서\n\n- 분석 완료"
                    
                    chatbot.add_company_analysis(report)
                    
                    initial_message = f"'{st.session_state.company_name}'에 대한 분석을 완료했습니다.\n\n"
                    if chatbot.memory.interview_questions:
                        initial_message += "**아래 생성된 면접 질문들을 바탕으로 면접 시뮬레이션을 시작해볼까요?**\n"
                        for i, q in enumerate(chatbot.memory.interview_questions):
                            initial_message += f"{i+1}. {q}\n"
                    
                    chatbot.memory.chat_history.append({"role": "assistant", "content": initial_message})
                    st.rerun()

    st.divider()

    st.header("📄 개인 맞춤 자료")
    uploaded_files = st.file_uploader("자기소개서, 이력서 등 업로드", type=['pdf', 'hwp', 'docx'], accept_multiple_files=True)
    
    if st.button("업로드된 파일로 DB 업데이트", use_container_width=True):
        if uploaded_files:
            with st.spinner('파일을 처리하고 DB를 업데이트합니다...'):
                # [향후 연동] 
                # 1. 업로드된 파일들을 'data' 폴더에 저장합니다.
                # 2. build_faiss_db.py의 build_or_update_vector_db()를 호출합니다.
                # 3. 업로드된 파일들의 내용을 요약하여 personal_info_summary에 저장합니다.
                #    summary = summarize_documents(uploaded_files)
                #    chatbot.add_personal_info(summary)
                time.sleep(2)
            st.success("개인 맞춤 자료 업데이트 완료!")
        else:
            st.warning("업로드할 파일이 없습니다.")

# --- 3. 메인 채팅 인터페이스 ---
# 챗봇의 메모리에 저장된 대화 기록을 사용
for msg in chatbot.memory.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("면접 질문에 답변하거나 자유롭게 질문해보세요..."):
    st.chat_message("user").write(user_input)

    with st.spinner("답변을 생성하는 중입니다..."):
        # 모든 로직을 ChatbotCore의 get_response에 위임
        ai_response = chatbot.get_response(user_input)
    
    st.rerun()
