import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load API
# -----------------------------
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Connect ChromaDB
# -----------------------------
db = chromadb.PersistentClient(path="vector_db")

cricket_collection = db.get_collection("cricket")
olympics_collection = db.get_collection("olympics")


# ==========================================
# ROUTER
# ==========================================

def route_query(question):

    prompt = f"""
You are a routing assistant.

Return ONLY ONE WORD.

cricket
olympics
both
unknown

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content.strip().lower()


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(collection, question, k=3):

    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["documents"][0]


# ==========================================
# GENERATION
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are a Sports Assistant.

Answer ONLY using the context below.

If the answer is not available in the context,
say

"I could not find the answer in the provided datasets."

Context:

{context}

Question:

{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content


# ==========================================
# MAIN
# ==========================================

while True:

    question = input("\nAsk Question : ")

    if question.lower()=="exit":
        break

    route = route_query(question)

    print("\nRoute :",route)

    if route=="unknown":

        print("\nQuestion outside provided datasets.\n")
        continue

    context=""

    if route=="cricket":

        docs=retrieve(
            cricket_collection,
            question
        )

        context="\n\n".join(docs)

    elif route=="olympics":

        docs=retrieve(
            olympics_collection,
            question
        )

        context="\n\n".join(docs)

    elif route=="both":

        cricket_docs=retrieve(
            cricket_collection,
            question
        )

        olympic_docs=retrieve(
            olympics_collection,
            question
        )

        context="\n\n".join(
            cricket_docs+olympic_docs
        )

    answer=generate_answer(
        question,
        context
    )

    print("\n========================")
    print("FINAL ANSWER")
    print("========================\n")

    print(answer)