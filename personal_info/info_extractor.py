from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import json
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

def extract_relevant_info(document_text, link_info):
    """
    Extract relevant information from document text and links.
    Args:
        document_text (str): Text extracted from resume/CV files.
        link_info (str): Crawled information from GitHub, portfolio, or LinkedIn URLs.
    Returns:
        str: Markdown string containing relevant information.
    """
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    # LLM 설정
    try:
        llm = AzureChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM: {str(e)}")

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in extracting relevant information from individual's resumes, CVs, cover letters, and online profiles for job applications. 
        Your task is to extract information from the provided document text and link information that is EXPLICITLY mentioned, combining them into a single output with minimal duplication. 
        Under NO circumstances should you infer, assume, or generate information that is not explicitly stated in the document text or link information. 
        Output the result in Markdown format with the following sections: Education, Experience, Skills, Certifications, Participated Projects, Online Profiles, and Other Relevant Information. 
        For the Participated Projects section, include the relevant project's name and, if any details are provided, add a one-sentence summary. 
        Each section should use bullet points for entries. 
        If a section has no relevant information, include the section header with no bullet points.

        Document Text: {document_text}
        Link Information: {link_info}

        Output format:
        ```markdown
        # Extracted Resume Information

        ## Education
        - [Education entry 1]
        - [Education entry 2]

        ## Experience
        - [Experience entry 1]
        - [Experience entry 2]

        ## Skills
        - [Skill 1]
        - [Skill 2]

        ## Certifications
        - [Certification 1]
        - [Certification 2]

        ## Participated Projects
        - [Participated project 1]: [Participated project 1 summary]
        - [Participated project 2]: [Participated project 2 summary]

        ## Online Profiles
        - [GitHub/Portfolio/LinkedIn entry 1]
        - [GitHub/Portfolio/LinkedIn entry 2]

        ## Other Relevant Information
        - [Other relevant entry 1]
        - [Other relevant entry 2]
        ```

        Instructions:
        - Extract ONLY information explicitly stated in the resume, CV, cover letter, or link information.
        - Do NOT add any information not present in the document or link information.
        - Combine information from document text and link information, ensuring minimal duplication (e.g., if the same project appears in both, include it only once with the most detailed summary).
        - For links (GitHub, portfolio, LinkedIn), include only the URLs or explicitly mentioned details (e.g., project names from GitHub).
        - For Participated Projects, include only projects explicitly mentioned in the document or link information, with a one-sentence summary if details are provided.
        - If a section has no relevant information, include the section header with no bullet points.
        - Ensure the output is valid Markdown with no additional text, comments, or explanations outside the specified format.
        - Example:
          Document Text: "B.S. Computer Science, Python experience, Korean History Certificate, Project: AI Chatbot - Developed a chatbot using Python and NLP"
          Link Information: "GitHub Repository: AI Chatbot, LinkedIn: https://linkedin.com/in/user"
          Output:
          ```markdown
          # Extracted Resume Information

          ## Education
          - B.S. Computer Science

          ## Experience
          - Python experience

          ## Skills
          - Python

          ## Certifications
          - Korean History Certificate

          ## Participated Projects
          - AI Chatbot: Developed a chatbot using Python and NLP

          ## Online Profiles
          - LinkedIn: https://linkedin.com/in/user

          ## Other Relevant Information
          ```
        """
    )

    # 체인 설정
    chain = RunnableSequence(prompt | llm)

    # 체인 실행
    try:
        result = chain.invoke({
            "document_text": document_text,
            "link_info": link_info
        })
        result_text = result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        raise Exception(f"LLM invocation failed: {str(e)}")


    return result_text