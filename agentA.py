import os
import re
from typing import Optional
import time
from urllib.parse import urljoin
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import fitz

load_dotenv()

def initialize_llm_and_tools():
    """LLM과 도구들을 초기화하고 튜플 형태로 반환합니다."""
    deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME 환경 변수를 .env 파일에 설정해주세요.")
    llm = AzureChatOpenAI(azure_deployment=deployment_name, temperature=0.3, max_tokens=4000)
    search_tool = DuckDuckGoSearchRun(region='kr-kr')
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=4000))
    tools = [scrape_website_content, search_tool, wiki_tool]
    return llm, tools

@tool
def scrape_website_content(url: str) -> str:
    """
    주어진 URL의 웹사이트 콘텐츠와, 해당 페이지에 링크된 '의미있는' PDF 파일들의 텍스트를 함께 스크래핑합니다.
    '직무', '요강', '설명', '기술서' 등의 키워드가 포함된 PDF를 우선적으로 분석합니다.
    """
    print(f">>> Executing Smart Scraper for URL: {url}")
    scraped_data = []
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(3)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        main_text = soup.get_text(separator='\n', strip=True)
        scraped_data.append("--- 메인 페이지 내용 ---\n" + main_text)
        pdf_links = soup.find_all('a', href=lambda href: href and href.lower().endswith('.pdf'))
        print(f">>> Found {len(pdf_links)} PDF link(s).")
        meaningful_keywords = ['직무', '요강', '설명', '기술서', '소개서', '공고']
        for link in pdf_links:
            link_text = link.get_text(strip=True)
            if any(keyword in link_text for keyword in meaningful_keywords):
                pdf_url = urljoin(url, link['href'])
                print(f">>> 의미있는 PDF 발견: {link_text} ({pdf_url})")
                try:
                    pdf_response = requests.get(pdf_url, timeout=20)
                    pdf_response.raise_for_status()
                    pdf_doc = fitz.open(stream=pdf_response.content, filetype="pdf")
                    pdf_text = "".join(page.get_text() for page in pdf_doc)
                    scraped_data.append(f"\n\n--- 첨부 PDF 내용: {link_text} ---\n" + pdf_text)
                    print(f">>> PDF '{link_text}' 텍스트 추출 성공.")
                except Exception as e:
                    scraped_data.append(f"\n--- PDF '{link_text}' 처리 중 오류: {e} ---")
    except Exception as e:
        return f"웹사이트 스크래핑 중 오류 발생: {e}"
    finally:
        driver.quit()
    return "\n".join(scraped_data)[:20000]

class GptResearcherStyleAnalyzer:
    def __init__(self, llm_model, tool_list):
        self.llm = llm_model
        self.tools = tool_list
        self.agent_executor = self._create_agent()
    def _create_agent(self):
        prompt_template = """
        당신은 AI 정보 수집 및 시장 분석 전문 에이전트입니다. 당신의 임무는 아래에 명시된 '표준 행동 절차(SOP)'를 엄격히 따라서, 주어진 주제에 대한 심층 분석 보고서를 작성하는 것입니다. 모든 정보는 요약 없이 상세하게 수집하고, 출처를 명시해야 합니다.
        ---
        **[표준 행동 절차 (SOP)]**
        **Phase 1: 기업 및 직무 내부 정보 분석 (Internal Analysis)**
        * Step 1.1 (개요 파악): `wikipedia_query_run` 도구를 사용하여 `"{company_name}"`에 대한 기본 정보(연혁, 사업 분야 등)를 확보한다.
        * Step 1.2 (공식 정보 수집): `duckduckgo_search`로 `"{company_name} 공식 홈페이지"`를 검색해 URL을 찾고, `scrape_website_content`를 실행하여 회사의 비전, 미션, 인재상, 주요 서비스 등의 정보를 수집한다.
        * Step 1.3 (채용 공고 분석): 만약 `url`이 '제공되지 않음'이 아니라면, 해당 `url`에 대해 `scrape_website_content`를 실행하여 직무, 자격 요건, 첨부 PDF 내용 등 모든 상세 정보를 수집한다.
        * Step 1.4 (기술/직무 정보 심층 분석):
            * `duckduckgo_search`를 사용하여 `"{company_name} 기술 블로그"` 또는 `"{company_name} engineering blog"`를 검색하여 기술 블로그 URL을 찾고, `scrape_website_content`로 상세 내용을 수집한다.
            * `duckduckgo_search`를 사용하여 **`"{company_name} {job_role} 직무 기술서"`** 또는 **`"{company_name} {job_role} JD"`**를 검색하여, 일반적인 직무 기술서(Job Description) 정보를 찾아 `scrape_website_content`로 수집한다.
        **Phase 2: 시장 및 경쟁 환경 분석 (External Analysis)**
        * Step 2.1 (경쟁사 식별): `duckduckgo_search`를 사용하여 `"{company_name} 주요 경쟁사"`를 검색하여, 가장 중요한 경쟁사 2곳의 이름을 식별한다.
        * Step 2.2 (산업 동향 파악): `duckduckgo_search`를 사용하여 `"[회사의 주요 산업] 기술 트렌드 2025"` 또는 `"[{job_role}] 분야 최신 기술 동향"` 과 같은 검색어로 현재 및 미래의 산업 동향을 조사한다.
        * Step 2.3 (전문 보고서 탐색): `duckduckgo_search`를 사용하여 `"{company_name} 기업 분석 보고서"`, `"{company_name} 증권사 리포트"` 또는 `"{company_name} 시장 점유율"` 과 같은 키워드로 검색하여, 전문가들이 작성한 심층 분석 자료나 객관적인 데이터를 찾는다. 발견 시 `scrape_website_content`로 해당 내용을 수집한다.
        **Phase 3: 종합 보고서 생성 (Synthesis & Reporting)**
        * 수집한 모든 정보를 취합하여, 아래 "최종 결과물 형식"에 맞춰 상세 보고서를 작성한다. 요약은 최소화하고, 원본에 가까운 상세한 정보를 제공하는 데 집중한다.
        ---
        **사용자 입력 정보:**
        - 회사명: {company_name}
        - 희망 직무: {job_role}
        - 채용 공고 URL: {url}
        ---
        **최종 결과물 형식:**
        ### [회사명] 및 관련 산업 심층 분석 보고서 ([희망 직무] 관점)
        ## 1. 기업 분석 (Company Analysis)
        ### 1.1. 기업 개요 (Source: Wikipedia/Homepage)
        ```text
        (위키피디아, 공식 홈페이지 등에서 수집한 기업의 비전, 연혁, 주요 사업, 가치, 인재상 등 상세 정보)
        ```
        ### 1.2. 기술 및 개발 문화 (Source: Tech Blog/Job Posting)
        ```text
        (기술 블로그, 채용 공고 PDF 등에서 수집한 기술 스택, 인프라, 개발 문화, 협업 방식 관련 내용)
        ```
        ## 2. 채용 포지션 분석 (Position Analysis)
        ### 2.1. 공식 채용 공고 (Source: Job Posting URL)
        ```text
        (URL이 제공된 경우, 해당 공고 및 첨부 PDF에서 수집한 직무, 책임, 자격 요건, 우대 사항 등 모든 정보)
        ```
        ### 2.2. 일반적인 직무 기술서 (Source: Web Search)
        ```text
        (웹에서 검색한 해당 직무의 일반적인 역할, 책임, 필요 역량(JD)에 대한 정보)
        ```
        ## 3. 시장 및 산업 분석 (Market & Industry Analysis)
        ### 3.1. 주요 경쟁사 및 시장 내 위치 (Source: Web Search)
        - **경쟁사 목록**: [조사된 주요 경쟁사 2~3곳 나열]
        - **경쟁사별 특징 및 비교 분석**: 
        ```text
        (각 경쟁사에 대해 수집한 정보 및 타겟 기업과의 비교 분석 내용)
        ```
        ### 3.2. 관련 산업 최신 동향 및 전망 (Source: Web Search/Reports)
        ```text
        (수집한 산업 동향, 최신 기술, 시장 전망, 증권사 리포트 등 상세 정보)
        ```
        ### 참고 자료
        - [출처 1]: (정보 수집에 사용된 모든 URL 주소)
        - [출처 2]: (정보 수집에 사용된 모든 URL 주소)
        - ...
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "회사명: {company_name}\n희망 직무: {job_role}\n채용 공고 URL: {url}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=50, handle_parsing_errors=True)
    def process(self, company_name: str, url: Optional[str] = None, job_role: Optional[str] = None) -> str:
        print(f"▶ 분석 에이전트 실행 시작 (입력: 회사명={company_name}, 직무={job_role}, URL={url})")
        response = self.agent_executor.invoke({
            "company_name": company_name,
            "job_role": job_role or "지정되지 않음",
            "url": url or "제공되지 않음"
        })
        raw_output = response.get('output', "결과물을 생성하지 못했습니다.")
        print("\n--- 에이전트 최종 결과물 (Raw) ---")
        print(raw_output)
        report_match = re.search(r"###\s.*", raw_output, re.DOTALL)
        if report_match:
            return report_match.group(0).strip()
        return "최종 보고서 형식의 결과물을 찾을 수 없습니다."

def run_analyzer(company_name: str, job_role: Optional[str] = None, url: Optional[str] = None) -> str:
    """
    입력값만으로 에이전트의 모든 설정과 실행을 처리하고 최종 보고서를 반환하는 마스터 함수.
    """
    print("--- 분석 시스템 초기화 시작 ---")
    llm, tools = initialize_llm_and_tools()
    analyzer = GptResearcherStyleAnalyzer(llm_model=llm, tool_list=tools)
    print("--- 분석 시스템 초기화 완료 ---")
    report = analyzer.process(
        company_name=company_name,
        url=url,
        job_role=job_role
    )
    return report
