import re
from feedback_generator import generate_feedback

def extract_expression_score(score_justification):
    if not score_justification:
        return ""
    m = re.search(r"표현력\s*:\s*(\d+)점", score_justification)
    if m:
        return m.group(1)
    return ""

def main():
    # 예시: 분석 결과를 받아왔다고 가정
    parsed_answer = """
    - 표현력: 4점 - 전달은 명확하나 다소 단조롭다.
    - 직무 적합성: 2점 - 직무 연관성 언급 부족.
    - 회사 맞춤성: 1점 - 회사에 대한 언급 없음.
    """

    feedback_markdown = generate_feedback(parsed_answer)
    print("\n[📋 피드백 결과]\n")
    print(feedback_markdown)

    # 표현력 점수만 추출!
    expr_score = extract_expression_score(parsed_answer)
    print(f"\n[표현력 점수] {expr_score}점")

if __name__ == "__main__":
    main()