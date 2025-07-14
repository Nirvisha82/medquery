import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_prompt import build_rag_prompt, generate_answer
from rag_pipeline import get_top_k_docs


@st.cache_resource
def load_index():
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("index", embedder, allow_dangerous_deserialization=True)

db = load_index()

st.title("🧬 MedQueryBot - Clinical RAG Assistant")

query = st.text_input("Ask a clinical question (e.g., signs of sepsis in ICU patients):")

if query:
    with st.spinner("Retrieving and generating answer..."):
        top_docs = get_top_k_docs(db, query)
        prompt = build_rag_prompt(query, top_docs)
        answer = generate_answer(prompt)
    st.markdown("### 💡 Answer")
    st.write(answer)
