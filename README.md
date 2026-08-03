# Agentic Sports Router

An AI-powered Retrieval-Augmented Generation (RAG) application that intelligently routes user queries to the appropriate sports dataset using a Large Language Model (Groq), retrieves relevant information from a local ChromaDB vector database, and generates accurate, grounded responses.

---

## Team Information

Team Name: Semantic Seekers

Team Number: 07

Problem Statement ID: PS-7 – Agentic Sports Router

---

## Team Members

- 24071A6694 - Kasparaju Saikushal
- 24071A66A8 - Manupati Srivathsava
- 24071A6669 - Yerram Sai Krishna

---


## Demo Video

Google Drive Link:

https://drive.google.com/<your-demo-video-link>

---

## GitHub Repository

https://github.com/kushalkasparaju-star/07_Semantic-Seekers-

---

## Problem Statement

An AI-powered Retrieval-Augmented Generation (RAG) application that intelligently routes user queries to the correct sports dataset using a Large Language Model (Groq), retrieves relevant information using ChromaDB, and generates grounded responses.


Datasets:

- World_Cricketers.xlsx
- Indian_Olympic_Players.xlsx

The application supports:

- LLM Query Routing
- Semantic Retrieval
- Multi-hop Retrieval
- Comparison Questions
- Out-of-Scope Detection
- Grounded Answer Generation

---

## Objective

The objective of this project is to build an intelligent Retrieval-Augmented Generation (RAG) system capable of:

- Understanding user queries
- Identifying the correct dataset
- Retrieving semantically relevant information
- Generating grounded answers
- Handling comparison and multi-hop questions
- Rejecting questions outside the provided datasets

This project demonstrates how Retrieval-Augmented Generation (RAG) combines semantic retrieval, LLM-based query routing, and grounded answer generation using only the provided datasets.

---

## Features

- LLM Query Routing using Groq
- Retrieval-Augmented Generation (RAG)
- ChromaDB Local Vector Database
- Sentence Transformer Embeddings
- Semantic Search
- Multi-hop Retrieval
- Comparison Questions
- Out-of-Scope Detection
- Interactive Gradio Web Interface

---

## Technologies Used

- Python
- Groq API
- ChromaDB
- Sentence Transformers
- Gradio
- Pandas
- OpenPyXL
- python-dotenv

---

## Datasets Used

This project uses only the datasets provided in the problem statement.

- World_Cricketers.xlsx
- Indian_Olympic_Players.xlsx

---

## Project Structure

```text
Agentic-Sports-Router
│
├── data
│   ├── World_Cricketers.xlsx
│   └── Indian_Olympic_Players.xlsx
│
├── src
│   ├── app.py
│   ├── rag_app.py
│   ├── llm_router.py
│   ├── search_engine.py
│   ├── create_vector_db.py
│   ├── prepare_documents.py
│   ├── retrieve.py
│   ├── read_data.py
│   └── test_setup.py
│
├── docs
├── requirements.txt
├── README.md
├── .gitignore
└── .env (not included)

```
 
## Environment Variables

This project requires a Groq API key for LLM-based query routing and answer generation.

Create a `.env` file in the project root and add the following:

```env
GROQ_API_KEY=your_groq_api_key
```

You can generate a Groq API key from:

https://console.groq.com/keys

**Note:** The actual API key is **not included** in this repository for security reasons.

---

## Local Vector Database

This project uses **ChromaDB** as a local vector database to store embeddings generated from the provided datasets.

The repository includes:

- Source datasets (`World_Cricketers.xlsx` and `Indian_Olympic_Players.xlsx`)
- Script to regenerate the vector database (`src/create_vector_db.py`)

To regenerate the vector database, run:

```bash
python src/create_vector_db.py
```

This will recreate the local ChromaDB vector database from the provided datasets for retrieval during application execution.

---

## Architecture

```text
                User
                 │
                 ▼
            Gradio Web UI
                 │
                 ▼
          Groq LLM Router
                 │
     ┌───────────┼───────────┐
     │           │           │
 Cricket     Olympics      Both
     │           │           │
     └───────────┼───────────┘
                 ▼
         ChromaDB Retrieval
                 │
                 ▼
        Retrieved Context
                 │
                 ▼
           Groq LLM
                 │
                 ▼
          Final Answer
```

## Workflow

```text

User Question
      │
      ▼
Groq LLM Router
      │
      ▼
Route Selection
(Cricket / Olympics / Both / Unknown)
      │
      ▼
Semantic Retrieval using ChromaDB
      │
      ▼
Relevant Context
      │
      ▼
Groq LLM
      │
      ▼
Grounded Answer

```
---

## Sample Questions

- Tell me about Virat Kohli
- Tell me about Neeraj Chopra
- Compare Virat Kohli and Neeraj Chopra
- Compare a Haryana Olympian and a Haryana cricketer
- Who is Elon Musk?