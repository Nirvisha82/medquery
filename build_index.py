import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading Embedding Model")
# Set up embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Loaded Embedding Model")

def load_all_pubmed_rct(paths, max_total=10000):
    documents = []
    count = 0
    for file_path in paths:
        with open(file_path, "r") as f:
            current_abstract = []
            for line in f:
                line = line.strip()
                if line == "":
                    if current_abstract:
                        text = " ".join(current_abstract)
                        documents.append(Document(page_content=text))
                        current_abstract = []
                        count += 1
                        if count >= max_total:
                            return documents
                else:
                    try:
                        section, sentence = line.split(" ", 1)
                        current_abstract.append(f"{section}: {sentence}")
                    except ValueError:
                        continue
    return documents

if __name__ == "__main__":
    data_paths = [
        os.path.join("data", "train.txt"),
        os.path.join("data", "dev.txt"),
        os.path.join("data", "test.txt")
    ]
    
    print("🔄 Loading abstracts...")
    docs = load_all_pubmed_rct(data_paths, max_total=20000)
    print(f"✅ Loaded {len(docs)} abstracts.")

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(docs, embedder)

    os.makedirs("index", exist_ok=True)
    db.save_local("index")
    print("✅ FAISS index saved to ./index")
