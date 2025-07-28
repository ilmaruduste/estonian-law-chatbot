# 🇪🇪 Riigi Teataja Legal Assistant (RAG)

This is a Retrieval-Augmented Generation (RAG) chatbot for answering questions about Estonian law using translated Riigi Teataja documents. It supports Estonian → English translation and uses open-source models to keep everything local and private.

## 📦 Project Structure

<pre><code>riigiteataja-juturobot/ 
├── data/ # Translated and original legal texts, FAISS index and embeddings 
├── translation/ # Local translation tools (Estonian ↔ English) 
├── rag_chain.py # Main chatbot logic 
├── utils/ # Utilities (e.g., chunking, formatting) 
└── README.md </code></pre>


## 🚀 Features

- ⚖️ Legal-BERT for semantic understanding of English-translated Estonian law  
- 🔍 FAISS vector search for fast document retrieval  
- 🌐 Estonian → English translation using a local model via [Ollama](https://ollama.com/)  
- 💬 Chat interface for asking questions like:  
  *“Millised on valitsuse ülesanded?”* → *“What are the roles of government?”*

## 🧠 Model Overview

| Component      | Model                               |
|----------------|-------------------------------------|
| Embedding      | `nlpaueb/legal-bert-base-uncased`   |
| Translation    | Local Estonian-English model via Ollama |
| Vector DB      | FAISS                               |
| RAG Framework  | Custom Python chain                 |

## 📥 Installation

```bash
git clone https://github.com/yourname/riigiteataja-juturobot.git
cd riigiteataja-juturobot
conda env create -f environment.yml  # or manually install requirements.txt
```

Make sure Ollama is running with a suitable translation model (e.g. mistral, nllb, or similar).

## 🛠 Usage
```bash
python rag_chain.py
```

Follow the prompt to enter legal questions in Estonian or English.

## 📝 Preprocessing & Embedding

1. Translate documents using:

```bash
python data/translate_riigiteataja.py
```

2. Embed the translated documents:
```bash
python data/embed_documents.py
```

## 🧪 Example

```vbnet
Enter your legal question (or type 'exit' to quit): How is the president elected?
Answer:
The President of Estonia is elected by the Riigikogu (Parliament), as specified in Section 79 of the Constitution of Estonia. If the Riigikogu fails to elect a president within three ballots, a fourth ballot is held with only two candidates who received the highest number of votes in the previous ballots. The candidate receiving an absolute majority (more than half) of the votes of all members of the Riigikogu wins the presidency. If no candidate receives an absolute majority after four ballots, a fifth ballot is held, and the candidate who receives the most votes becomes the President.
```

## 📚 Data Source

Public legal texts are from [Riigi Teataja](https://www.riigiteataja.ee/index.html).

## 📄 License

MIT License – use freely, but respect the legal source content.