import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to vector database
client = chromadb.PersistentClient(path="vector_db")

cricket = client.get_collection("cricket")
olympics = client.get_collection("olympics")


def search(collection, question, k=3):
    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["documents"][0]


question = input("Ask a question: ")

print("\n===== Cricket Results =====\n")

for doc in search(cricket, question):
    print(doc)
    print("-" * 60)

print("\n===== Olympic Results =====\n")

for doc in search(olympics, question):
    print(doc)
    print("-" * 60)