from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_faiss_index(documents):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embedder)

def get_top_k_docs(db, query, k=3):
    return db.similarity_search(query, k=k)
