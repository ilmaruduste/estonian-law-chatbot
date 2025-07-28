# rag_chain.py
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import requests
import gradio as gr

# --- Load FAISS index and documents ---
index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# --- Load embedding model (Legal-BERT or same as used for indexing) ---
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").eval().to("cuda")

def embed_query(query: str):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def retrieve_documents(query: str, k: int = 5):
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, k)
    return [docs[i] for i in indices[0]]

def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are an expert in Estonian law. Answer the user's question using only the provided context. You can directly quote the context to support your answer.

Context:
{context}

Question:
{query}

Answer in detail, but be concise. Also mention what document and specific § number contain(s) the answer the user is looking for, but only if you are absolutely sure. Think twice before you give an answer:"""
    return prompt

def call_ollama_mistral(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "mistral",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()

def rag_answer(query: str) -> str:
    chunks = retrieve_documents(query)
    prompt = build_prompt(query, chunks)
    return call_ollama_mistral(prompt)

def gradio_chat(query):
    if not query.strip():
        return "Please enter a question."
    return rag_answer(query)

iface = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a legal question..."),
    outputs=gr.Textbox(label="Answer"),
    title="Estonian Legal Assistant",
    description="Ask questions based on Estonian law documents. Powered by FAISS + Mistral + Ollama."
)

if __name__ == "__main__":
    iface.launch(share=True)