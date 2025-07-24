import argparse
from document_processor import load_documents
from info_extractor import extract_relevant_info
import os

def main(file_paths, job_file, links):
    # 문서 로딩 (여러 PDF/DOCX 파일)
    try:
        document_text = load_documents(file_paths)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 직무 공고 로딩 (Markdown)
    try:
        with open(job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
    except Exception as e:
        print(f"Error loading job description file: {e}")
        return

    # 링크 정보 추가
    link_info = "\n".join([f"Link: {link}" for link in links]) if links else ""

    # 정보 추출
    try:
        result = extract_relevant_info(document_text, job_description, link_info)
        # Markdown 파일로 저장
        output_file = "output.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Output saved to {output_file}")
    except Exception as e:
        print(f"Error extracting information: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relevant info from resume/CV and links for a job.")
    parser.add_argument("--file", required=True, action="append", help="Path to resume/CV file (PDF or DOCX), can be repeated")
    parser.add_argument("--job", required=True, help="Path to job description file (Markdown)")
    parser.add_argument("--links", action="append", default=[], help="URLs for GitHub, portfolio, or LinkedIn, can be repeated")
    args = parser.parse_args()

    main(args.file, args.job, args.links)