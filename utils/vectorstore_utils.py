import pinecone
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "test"

def init_pinecone():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=384)
    return pinecone.Index(index_name)

def process_and_store_docs(all_docs, index):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    texts = [doc.page_content for doc in split_docs]
    metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
    vectors = embed_model.encode(texts)

    index.upsert([
        (f"id-{i}", vec.tolist(), metadatas[i]) for i, vec in enumerate(vectors)
    ])

    return Pinecone(index, embed_model.encode, "text")
