import argparse
from document_processor import load_documents
from info_extractor import extract_relevant_info
import os
import requests
from bs4 import BeautifulSoup
import json

def crawl_github(username, token=None):
    """
    Crawl public GitHub repositories for a user using GitHub API.
    Args:
        username (str): GitHub username extracted from URL.
        token (str): Optional GitHub personal access token.
    Returns:
        str: Formatted string of repository names and descriptions.
    """
    try:
        headers = {'Authorization': f'token {token}'} if token else {}
        response = requests.get(f"https://api.github.com/users/{username}/repos", headers=headers, timeout=5)
        response.raise_for_status()
        repos = response.json()
        repo_info = [f"GitHub Repository: {repo['name']} - {repo.get('description', 'No description')}" for repo in repos if not repo['private']]
        return "\n".join(repo_info[:5])  # 최대 5개 레포지토리
    except Exception as e:
        print(f"Error crawling GitHub {username}: {e}")
        return f"GitHub: https://github.com/{username}"

def crawl_linkedin(url):
    """
    Crawl public LinkedIn profile for basic info and posts (if available).
    Returns URL if private or scraping fails.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        ld_json = soup.find('script', {'type': 'application/ld+json'})
        if ld_json:
            profile_data = json.loads(ld_json.text)
            name = profile_data.get('name', 'Unknown')
            job_title = profile_data.get('jobTitle', 'No job title')
            return f"LinkedIn Profile: {name} - {job_title}"
        return f"LinkedIn: {url}"
    except Exception as e:
        print(f"Error crawling LinkedIn {url}: {e}")
        return f"LinkedIn: {url}"

def main(file_paths, links):
    # 문서 로딩 (여러 PDF/DOCX 파일)
    try:
        document_text = load_documents(file_paths)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 링크 정보 크롤링
    link_info = []
    github_token = os.getenv("GITHUB_TOKEN")
    for link in links:
        if 'github.com' in link:
            username = link.split('github.com/')[-1].split('/')[0]
            link_info.append(crawl_github(username, github_token))
        elif 'linkedin.com' in link:
            link_info.append(crawl_linkedin(link))
        else:
            link_info.append(f"Portfolio: {link}")
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