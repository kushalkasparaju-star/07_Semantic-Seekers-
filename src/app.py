import os
import chromadb
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# Load Embedding Model
# =====================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# Connect to ChromaDB
# =====================================================

db = chromadb.PersistentClient(path="vector_db")

cricket_collection = db.get_collection("cricket")
olympics_collection = db.get_collection("olympics")

# =====================================================
# LLM ROUTER
# =====================================================

def route_query(question):

    prompt = f"""
You are an intelligent routing assistant.

Available datasets:

1. World_Cricketers.xlsx
2. Indian_Olympic_Players.xlsx

Choose ONLY ONE of these words.

cricket
olympics
both
unknown

Rules:

- Cricket questions → cricket
- Olympic questions → olympics
- Comparison questions → both
- Unrelated questions → unknown

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


# =====================================================
# VECTOR RETRIEVAL
# =====================================================

def retrieve(collection, question, k=3):

    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["documents"][0]


# =====================================================
# ANSWER GENERATION
# =====================================================

def generate(question, context):

    prompt = f"""
You are an expert Sports Assistant.

Answer ONLY using the information provided below.

Do NOT make up facts.

If the answer cannot be found in the context, reply:

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
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# =====================================================
# MAIN RAG FUNCTION
# =====================================================

def sports_router(question):

    if question.strip() == "":
        return (
            "",
            "",
            "Please enter a question."
        )

    route = route_query(question)

    if route == "unknown":

        return (
            "Unknown",
            "No relevant documents retrieved.",
            "Sorry! This application answers questions only from:\n\n"
            "• World_Cricketers.xlsx\n"
            "• Indian_Olympic_Players.xlsx"
        )

    retrieved_docs = []

    if route == "cricket":

        retrieved_docs = retrieve(
            cricket_collection,
            question
        )

    elif route == "olympics":

        retrieved_docs = retrieve(
            olympics_collection,
            question
        )

    elif route == "both":

        cricket_docs = retrieve(
            cricket_collection,
            question
        )

        olympic_docs = retrieve(
            olympics_collection,
            question
        )

        retrieved_docs = cricket_docs + olympic_docs

    context = "\n\n".join(retrieved_docs)

    answer = generate(
        question,
        context
    )

    return (
        route.capitalize(),
        context,
        answer
    )


# =====================================================
# GRADIO UI
# =====================================================

demo = gr.Interface(
    fn=sports_router,

    inputs=gr.Textbox(
        lines=2,
        placeholder="Example: Compare Virat Kohli and Neeraj Chopra"
    ),

    outputs=[
        gr.Textbox(
            label="Selected Route"
        ),

        gr.Textbox(
            label="Retrieved Context",
            lines=12
        ),

        gr.Textbox(
            label="Final Answer",
            lines=10
        )
    ],

    title="🏏 Agentic Sports Router",

    description="""
### RAG + LLM Query Routing + Multi-hop Retrieval

This application routes each question to the correct dataset (Cricket / Olympics), retrieves relevant information using ChromaDB, and generates grounded answers using Groq LLM.
"""
)

demo.launch()