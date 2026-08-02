# 🏏 Agentic Sports Router

An AI-powered Retrieval-Augmented Generation (RAG) application that intelligently routes user queries to the correct sports dataset using a Large Language Model (Groq), retrieves relevant information using ChromaDB, and generates grounded responses.

---

## 📌 Problem Statement

Build a system that routes user questions to the correct dataset:

- World_Cricketers.xlsx
- Indian_Olympic_Players.xlsx

The system should also support:

- LLM Query Routing
- Multi-hop Retrieval
- Comparison Questions
- Out-of-Scope Handling

---

## 🚀 Features

- ✅ LLM Query Routing using Groq
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ ChromaDB Local Vector Database
- ✅ Sentence Transformer Embeddings
- ✅ Semantic Search
- ✅ Multi-hop Retrieval
- ✅ Comparison Questions
- ✅ Out-of-Scope Detection
- ✅ Interactive Gradio Web Interface

---

## 🛠️ Technologies Used

- Python
- Groq API
- ChromaDB
- Sentence Transformers
- Gradio
- Pandas
- OpenPyXL

---

## 📂 Project Structure

```
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
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Vector Database

```bash
python src/create_vector_db.py
```

### Run Application

```bash
python src/app.py
```

---

## 📸 Sample Questions

- Tell me about Virat Kohli
- Tell me about Neeraj Chopra
- Compare Virat Kohli and Neeraj Chopra
- Compare a Haryana Olympian and a Haryana cricketer
- Who is Elon Musk?

---

## 🧠 Architecture

```
User
   │
   ▼
Gradio UI
   │
   ▼
Groq LLM Router
   │
   ├── Cricket
   ├── Olympics
   ├── Both
   └── Unknown
   │
   ▼
ChromaDB Retrieval
   │
   ▼
Groq LLM Generation
   │
   ▼
Final Answer
```

---

## 🔮 Future Enhancements

- Support additional sports datasets
- Hybrid Search (Keyword + Semantic)
- Better Retrieval Ranking
- Cloud Deployment
- User Authentication

---

## 👨‍💻 Team

(Add your team member names here)

---

## 📄 License

Academic Project