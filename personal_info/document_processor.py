from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import os

def load_documents(file_paths):
    """
    Load multiple PDF or DOCX files and return combined text content.
    Args:
        file_paths (list): List of paths to document files.
    Returns:
        str: Combined text from all documents.
    """
    combined_text = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}. Use PDF or DOCX.")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        combined_text.append(text)
    
    return "\n".join(combined_text)