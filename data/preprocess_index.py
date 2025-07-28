import os
import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

DOCS_DIR = "data/riigiteataja_translated"
OUTPUT_INDEX = "data/embeddings/index.faiss"
OUTPUT_DOCS = "data/embeddings/docs.pkl"

# Load Legal-BERT
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import re

def extract_chunks_with_headers(text, filename, max_words=200):
    lines = text.splitlines()
    current_chapter = "Unknown Chapter"
    current_paragraph = "Unknown Paragraph"
    buffer = []
    chunks = []

    def flush_buffer():
        if buffer:
            words = " ".join(buffer).split()
            for i in range(0, len(words), max_words):
                chunk_words = words[i:i + max_words]
                chunk_text = " ".join(chunk_words)
                header = f"[{filename} - {current_chapter} - {current_paragraph} - chunk {len(chunks)+1}]\n"
                chunks.append(header + chunk_text)
            buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect chapter headers: e.g., ## XIII. Chapter COMMODITY
        chapter_match = re.match(r"^##\s+\*\*(.*?)\*\*", line)
        if chapter_match:
            flush_buffer()
            current_chapter = chapter_match.group(1).strip()
            continue

        # Detect paragraph headers: e.g., **§ 146. Some text...**
        paragraph_match = re.match(r"^\*\*§\s*(\d+[^\s]*)\.*", line)
        if paragraph_match:
            flush_buffer()
            current_paragraph = f"§ {paragraph_match.group(1)}"
            # Optional: capture rest of line as content
            line = re.sub(r"^\*\*§\s*\d+[^\s]*\.*\s*", "", line)
            if line:
                buffer.append(line)
            continue

        # Regular content line
        buffer.append(line)

    flush_buffer()
    return chunks


# --- 1. Split long text into chunks ---
def split_into_chunks(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# --- 2. Compute embeddings ---
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

# --- 3. Main pipeline ---
all_chunks = []
all_embeddings = []

logging.info("📄 Reading and chunking documents...")

filename_list = os.listdir(DOCS_DIR)
logging.info(f"Filenames: {filename_list}")

for filename in filename_list:
    if filename.endswith(".txt"):
        with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
            logging.info(f"Processing file: {filename}")
            text = f.read()

        chunks = extract_chunks_with_headers(text, filename)
        all_chunks.extend(chunks)


logging.info(f"🔢 Computing embeddings for {len(all_chunks)} chunks...")
for chunk in tqdm(all_chunks):
    embedding = embed_text(chunk)
    all_embeddings.append(embedding)

# --- 4. Store in FAISS index ---
logging.info("💾 Writing to FAISS index...")
dimension = all_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(all_embeddings))
faiss.write_index(index, OUTPUT_INDEX)

with open(OUTPUT_DOCS, "wb") as f:
    pickle.dump(all_chunks, f)

logging.info(f"✅ Done. Stored {len(all_chunks)} chunks in {OUTPUT_INDEX} and {OUTPUT_DOCS}")
