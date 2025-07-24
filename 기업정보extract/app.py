import streamlit as st
import time
from feedback_generator import generate_feedback   # 반드시 feedback_generator.py 필요!


import re

def extract_expression_score(score_justification):
    if not score_justification:
        return ""
    m = re.search(r"표현력\s*:\s*(\d+)점", score_justification)
    if m:
        return m.group(1)
    return ""

# -------------------------
# 페이지 설정 및 타이틀
# -------------------------
st.set_page_config(page_title="📝 피드백 생성기", layout="centered")
st.title("🎤 모의 면접 피드백 시스템")

# -------------------------
# 사이드바 - 개인 맞춤 자료, 질문 생성
# -------------------------
with st.sidebar:
    st.header("📄 개인 자료")
    github_url = st.text_input("GitHub URL (선택)", key="github_url")
    linkedin_url = st.text_input("LinkedIn URL (선택)", key="linkedin_url")
    portfolio_url = st.text_input("Portfolio URL (선택)", key="portfolio_url")

    uploaded_files = st.file_uploader(
        "자기소개서, 이력서 등을 업로드하세요",
        type=['pdf', 'hwp', 'docx'],
        accept_multiple_files=True
    )

    if st.button("📂 개인 자료 분석 및 DB 업데이트"):
        if not uploaded_files and not (github_url or linkedin_url or portfolio_url):
            st.warning("업로드 파일 또는 URL 중 최소 하나는 입력해야 합니다.")
        else:
            with st.spinner("자료 분석 및 DB 구축 중..."):
                # TODO: 실제 파일/URL 분석 및 DB 구축 기능 구현
                time.sleep(2)
            st.success("개인 자료 분석 완료 및 DB 업데이트 성공!")

    st.markdown("---")
    st.header("🎯 질문 생성 Agent")
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""

    if st.button("📌 새 질문 생성"):
        with st.spinner("면접 질문 생성 중..."):
            # 실제로는 chatbot.generate_question()을 넣으세요.
            st.session_state["current_question"] = "우리 회사에 지원하게 된 동기를 말씀해 주세요."  # 예시

# -------------------------
# 기본 정보 입력
# -------------------------
st.subheader("📝 기본 정보 입력")
company_name = st.text_input("회사명", key="company_name_main")
job_role = st.text_input("희망 직무", key="job_role_main")

if st.button("✨ 분석 시작", type="primary"):
    if not company_name or not job_role:
        st.warning("회사명과 희망 직무는 반드시 입력해주세요.")
    else:
        st.session_state["company_name"] = company_name
        st.session_state["job_role"] = job_role
        with st.spinner("분석 중입니다..."):
            time.sleep(2)
        st.success("분석 완료! 아래 질문을 확인해보세요.")

# -------------------------
# 질문 출력
# -------------------------
st.subheader("🗣️ 면접 질문")
if st.session_state.get("current_question"):
    st.markdown(f"**{st.session_state['current_question']}**")
else:
    st.info("❗ 아직 생성된 면접 질문이 없습니다. 사이드바에서 질문을 생성하세요.")

# -------------------------
# 사용자 답변 입력 및 제출
# -------------------------
st.subheader("✍️ 사용자 답변")
user_answer = st.text_area("면접 질문에 대한 답변을 입력하세요", height=200)
answer_submitted = st.button("📤 답변 제출")

if answer_submitted:
    if not user_answer:
        st.warning("답변을 입력해주세요.")
    else:
        with st.spinner("답변 분석 중..."):
            st.session_state["parsed_answer"] = user_answer  # 예시는 답변 그대로 복사
            st.session_state["score_justification"] = """표현력: 4점 - 전달은 명확하나 다소 단조롭다.
직무 적합성: 2점 - 직무 연관성 언급 부족.
회사 맞춤성: 1점 - 회사에 대한 언급 없음."""
            time.sleep(1)
        st.success("답변이 제출되었습니다! 아래에서 피드백을 생성해보세요.")

# -------------------------
# 답변 해석 결과 및 피드백 생성 폼
# -------------------------
st.title("📝 모의면접 피드백 생성기")
st.markdown("답변 분석 Agent의 결과를 아래에 붙여넣고, 피드백을 생성하세요.")

with st.form("feedback_form"):
    parsed_answer = st.text_area(
        "🧠 사용자 답변 해석 결과",
        value=st.session_state.get("parsed_answer", ""),
        height=200
    )
    score_justification = st.text_area(
        "📊 배점 결과 및 이유",
        value=st.session_state.get("score_justification", ""),
        height=200
    )
    submitted = st.form_submit_button("📊 피드백 생성")

if submitted:
    if not parsed_answer or not score_justification:
        st.warning("두 입력 모두 작성해 주세요.")
    else:
        with st.spinner("피드백 생성 중..."):
            feedback = generate_feedback(parsed_answer, score_justification)
            time.sleep(1)
        st.markdown("### ✅ 생성된 피드백")
        st.markdown(feedback)
      