import argparse
from document_processor import load_documents
from info_extractor import extract_relevant_info
import os
import requests
from bs4 import BeautifulSoup

def crawl_url(url):
    """
    Crawl basic information from a URL (GitHub, portfolio, LinkedIn).
    Returns a string with extracted information or the URL itself if crawling fails.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # GitHub: 레포지토리 이름 및 설명 추출
        if 'github.com' in url:
            repos = soup.find_all('a', {'itemprop': 'name codeRepository'})
            repo_info = [f"GitHub Repository: {repo.text.strip()}" for repo in repos[:3]]  # 최대 3개
            return "\n".join(repo_info) if repo_info else f"GitHub: {url}"
        # LinkedIn: 제한적으로 URL만 포함 (스크래핑 제한)
        elif 'linkedin.com' in url:
            return f"LinkedIn: {url}"
        # 포트폴리오: 프로젝트 이름 또는 설명 추출
        else:
            title = soup.find('title')
            return f"Portfolio: {title.text.strip() if title else url}"
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return f"URL: {url}"

def main(file_paths, links):
    # 문서 로딩 (여러 PDF/DOCX 파일)
    try:
        document_text = load_documents(file_paths)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 링크 정보 크롤링
    link_info = []
    for link in links:
        link_info.append(crawl_url(link))
    link_info = "\n".join(link_info) if link_info else ""

    # 정보 추출
    try:
        result = extract_relevant_info(document_text, link_info)
        # Markdown 파일로 저장
        output_file = "output.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Output saved to {output_file}")
    except Exception as e:
        print(f"Error extracting information: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relevant info from resume/CV and links.")
    parser.add_argument("--file", required=True, action="append", help="Path to resume/CV file (PDF or DOCX), can be repeated")
    parser.add_argument("--links", action="append", default=[], help="URLs for GitHub, portfolio, or LinkedIn, can be repeated")
    args = parser.parse_args()

    main(args.file, args.links)