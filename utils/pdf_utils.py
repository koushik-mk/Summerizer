import tempfile
from langchain_community.document_loaders import PyPDFLoader

def extract_documents_from_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
