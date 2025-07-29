import pickle

with open("data/embeddings/docs.pkl", "rb") as f:
    docs = pickle.load(f)

for i, doc in enumerate(docs):  # show all documents
    print(f"\n--- Document {i+1} ---\n{doc}\n")

with open("data/embeddings/docs.txt", "w", encoding = "utf-8") as f:
    for i, doc in enumerate(docs):
        f.write(f"\n--- Document {i+1} ---\n{doc}\n")