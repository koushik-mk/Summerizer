import streamlit as st
import time
from dotenv import load_dotenv
from utils.llm_utils import load_llm, get_prompt_template
from utils.pdf_utils import extract_documents_from_pdfs
from utils.vectorstore_utils import init_pinecone, process_and_store_docs
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

# Load environment variables
load_dotenv()

st.title("RAG App using Groq & Pinecone")

# Upload PDFs
uploaded_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        start_time = time.time()
        all_docs = extract_documents_from_pdfs(uploaded_files)
        index = init_pinecone()
        vectorstore = process_and_store_docs(all_docs, index)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success(f"Vector Store Ready in {time.time() - start_time:.2f} sec")
    else:
        st.warning("Please upload at least one PDF.")

query = st.text_input("Ask your question from the documents")

if query:
    if "retriever" in st.session_state:
        llm = load_llm()
        prompt = get_prompt_template()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)

        query_start_time = time.time()
        response = retrieval_chain.invoke({'input': query})
        query_time = time.time() - query_start_time

        st.write(f"Response Time: {query_time:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response.get("context", []):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("Please process documents first.")
