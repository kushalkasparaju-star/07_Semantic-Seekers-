# Agentic Sports Router Architecture

```
                   User
                     │
                     ▼
              Gradio Web UI
                     │
                     ▼
             User Question
                     │
                     ▼
          Groq LLM Query Router
     (Cricket / Olympics / Both / Unknown)
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
 Cricket ChromaDB            Olympics ChromaDB
      │                             │
      └──────────────┬──────────────┘
                     ▼
          Retrieved Context
                     │
                     ▼
         Groq LLM Answer Generator
                     │
                     ▼
              Final Response
```

## Workflow

1. User enters a sports question.
2. Groq LLM classifies the query.
3. The router selects Cricket, Olympics, Both, or Unknown.
4. ChromaDB retrieves the most relevant documents.
5. Retrieved context is sent to Groq.
6. Groq generates a grounded answer.
7. The answer is displayed in the Gradio interface.