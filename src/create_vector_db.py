import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read datasets
cricket = pd.read_excel("data/World_Cricketers.xlsx")
olympics = pd.read_excel("data/Indian_Olympic_Players.xlsx")


def dataframe_to_documents(df):
    docs = []

    for _, row in df.iterrows():
        text = ""

        for column in df.columns:
            text += f"{column}: {row[column]}\n"

        docs.append(text)

    return docs


cricket_docs = dataframe_to_documents(cricket)
olympic_docs = dataframe_to_documents(olympics)

# Create vector database folder
client = chromadb.PersistentClient(path="vector_db")

# Create collections
cricket_collection = client.get_or_create_collection("cricket")
olympic_collection = client.get_or_create_collection("olympics")


# Store cricket embeddings
for i, doc in enumerate(cricket_docs):
    embedding = model.encode(doc).tolist()

    cricket_collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )


# Store olympic embeddings
for i, doc in enumerate(olympic_docs):
    embedding = model.encode(doc).tolist()

    olympic_collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )


print("Cricket players stored :", cricket_collection.count())
print("Olympic players stored :", olympic_collection.count())