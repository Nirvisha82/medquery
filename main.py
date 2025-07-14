import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_prompt import build_rag_prompt, generate_answer
from rag_pipeline import get_top_k_docs
from pathlib import Path
import gdown


# https://drive.google.com/file/d/1EHd6lJS4Aag6pWIV4dgGBnvv_0ouqVxi/view?usp=sharing
# https://drive.google.com/file/d/1UsuDWi61R5PB9EoBkQiBaTUrEY65zCTb/view?usp=sharing


# @st.cache_resource
# def load_index():
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.load_local("index", embedder, allow_dangerous_deserialization=True)

@st.cache_resource
def load_index():
    # Map of filenames to Google Drive file IDs
    gdrive_ids = {
        "index.faiss": "1UsuDWi61R5PB9EoBkQiBaTUrEY65zCTb",
        "index.pkl": "1EHd6lJS4Aag6pWIV4dgGBnvv_0ouqVxi"
    }

    local_dir = Path("index")
    local_dir.mkdir(exist_ok=True)

    # Download each file only if not present locally
    for fname, file_id in gdrive_ids.items():
        fpath = local_dir / fname
        if not fpath.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(fpath), quiet=False)

    # Load the FAISS index
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(local_dir), embedder, allow_dangerous_deserialization=True)

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
