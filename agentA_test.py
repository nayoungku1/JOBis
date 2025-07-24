# agentA.py에서 마스터 함수를 임포트
from agentA import run_analyzer


# 원하는 입력값으로 함수를 직접 호출
# report = run_analyzer(
#     company_name="당근마켓", 
#     job_role="재무",
#     url="https://about.daangn.com/jobs/6497599003/" # 예시 URL
# )

report = run_analyzer(
    company_name="토스",
    job_role="웹 개발자",
    url="https://toss.im/career/job-detail?job_id=6034265003"
)

# 결과 출력
print(report)