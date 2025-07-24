import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document

# LangChain 모듈
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# [변경점] 별도로 분리한 파일 처리 함수들을 임포트합니다.
from file_processors import load_hwp_text_with_extractor, unzip_and_cleanup

load_dotenv()

# --- 메인 스크립트 로직 ---

def build_or_update_vector_db():
    """
    'data' 폴더의 문서를 기반으로 FAISS 벡터 DB를 생성하거나 업데이트합니다.
    """
    doc_dir = "data"
    db_path = "faiss_db"
    log_path = os.path.join(db_path, "processed_files.log")

    if not os.path.exists(doc_dir):
        print(f"오류: '{doc_dir}' 폴더를 찾을 수 없습니다.")
        return

    # --- 0. [호출] ZIP 파일 우선 처리 ---
    unzip_and_cleanup(doc_dir)

    # --- 1. 기존 DB 및 처리된 파일 목록 로드 ---
    vectorstore = None
    processed_files = set()
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(db_path):
        print(f"기존 벡터 DB를 '{db_path}'에서 로드합니다...")
        try:
            vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    processed_files = set(line.strip() for line in f)
            print(f"로드 완료. 총 {len(processed_files)}개의 파일이 이미 처리되었습니다.")
        except Exception as e:
            print(f"기존 DB 로드 실패: {e}. DB를 새로 생성합니다.")
            processed_files = set()
    else:
        print(f"기존 벡터 DB가 없습니다. '{db_path}'에 새로 생성합니다.")
        os.makedirs(db_path, exist_ok=True)

    # --- 2. 새로운 파일 식별 ---
    supported_extensions = ['.pdf', '.hwp', '.hwpx', '.csv']
    current_files = {f for f in os.listdir(doc_dir) if f.lower().endswith(tuple(supported_extensions))}
    new_files_to_process = sorted(list(current_files - processed_files))

    if not new_files_to_process:
        print("\n새롭게 추가된 파일이 없습니다. 프로세스를 종료합니다.")
        return

    print(f"\n총 {len(new_files_to_process)}개의 새로운 파일을 처리합니다: {new_files_to_process}")

    # --- 3. 새로운 파일만 처리 ---
    new_docs = []
    for file_basename in new_files_to_process:
        file_path = os.path.join(doc_dir, file_basename)
        try:
            ext = os.path.splitext(file_path)[1].lower()
            docs = []
            if ext == '.pdf':
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
            elif ext in ['.hwp', '.hwpx']:
                # [호출] 분리된 HWP 처리 함수 사용
                docs = load_hwp_text_with_extractor(file_path)
            elif ext == '.csv':
                encodings_to_try = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, FileNotFoundError):
                        continue
                if df is None: raise ValueError("지원되는 인코딩으로 파일을 읽을 수 없습니다.")
                for index, row in df.iterrows():
                    content = row.get("Question", " ".join(map(str, row.values)))
                    metadata = {"source": file_basename, "row": index + 1, **row.to_dict()}
                    docs.append(Document(page_content=content, metadata=metadata))
            
            new_docs.extend(docs)
            print(f"성공: '{file_basename}' ({len(docs)}개 문서)")
        except Exception as e:
            print(f"오류: '{file_basename}' 처리 중 오류 발생: {e}")

    if not new_docs:
        print("\n새로운 문서 내용이 없어 DB를 업데이트하지 않습니다.")
        return

    # --- 4. 문서 분할 및 DB에 추가 ---
    print("\n새로운 문서를 청크(chunk)로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    split_chunks = text_splitter.split_documents(new_docs)
    print(f"총 {len(split_chunks)}개의 새로운 청크로 분할되었습니다.")

    if vectorstore is None:
        print("\n새로운 벡터스토어를 생성합니다...")
        vectorstore = FAISS.from_documents(documents=split_chunks, embedding=embeddings)
    else:
        print("\n기존 벡터스토어에 새로운 문서를 추가합니다...")
        vectorstore.add_documents(split_chunks)

    # --- 5. DB 및 처리 목록 저장 ---
    vectorstore.save_local(db_path)
    with open(log_path, 'w', encoding='utf-8') as f:
        for filename in sorted(list(current_files)):
            f.write(filename + '\n')
            
    print(f"\n✅ 벡터스토어 업데이트 완료. 총 {len(current_files)}개의 파일이 처리되었습니다.")


if __name__ == "__main__":
    build_or_update_vector_db()
