import google.generativeai as genai

genai.configure(api_key="AIzaSyDK7tugJp0OfV159AbWIZmYAZXU2JogHAA")

model = genai.GenerativeModel("gemini-2.0-flash")

def build_rag_prompt(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"""You are a helpful medical assistant.
Use the following context from research articles to answer the question truthfully.
If unsure, say you don’t know.

Context:
{context}

Question: {question}
Answer:"""

def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text