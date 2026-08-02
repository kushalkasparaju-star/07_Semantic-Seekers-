import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
db = chromadb.PersistentClient(path="vector_db")

cricket_collection = db.get_collection("cricket")
olympics_collection = db.get_collection("olympics")


# -------------------------------
# LLM ROUTER
# -------------------------------
def route_query(question):

    prompt = f"""
You are an intelligent routing assistant.

Classify the user's question into ONLY ONE category.

cricket
olympics
both
unknown

Return ONLY one word.

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip().lower()


# -------------------------------
# VECTOR SEARCH
# -------------------------------
def search(collection, question, k=3):

    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["documents"][0]


# -------------------------------
# MAIN
# -------------------------------
while True:

    question = input("\nAsk Question : ")

    if question.lower() == "exit":
        break

    route = route_query(question)

    print("\nRoute :", route)

    if route == "cricket":

        docs = search(cricket_collection, question)

        print("\nRetrieved Cricket Documents\n")

        for doc in docs:
            print(doc)
            print("-" * 60)

    elif route == "olympics":

        docs = search(olympics_collection, question)

        print("\nRetrieved Olympic Documents\n")

        for doc in docs:
            print(doc)
            print("-" * 60)

    elif route == "both":

        print("\nRetrieved Cricket Documents\n")

        for doc in search(cricket_collection, question):
            print(doc)
            print("-" * 60)

        print("\nRetrieved Olympic Documents\n")

        for doc in search(olympics_collection, question):
            print(doc)
            print("-" * 60)

    else:

        print("\nQuestion is outside the provided datasets.")