import pandas as pd
from langchain.schema import Document

def load_pubmed_abstracts_from_csv(file_path="pubmed_abstracts.csv", max_docs=100):
    df = pd.read_csv(file_path).dropna(subset=["title", "abstract"])
    docs = []
    for _, row in df.head(max_docs).iterrows():
        content = f"{row['title']}\n\n{row['abstract']}"
        metadata = {"journal": row.get("journal", ""), "date": row.get("pub_date", "")}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

if __name__=="__main__":
    x=load_pubmed_abstracts_from_csv()
    pass