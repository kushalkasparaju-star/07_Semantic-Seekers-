import pandas as pd

# Read datasets
cricket = pd.read_excel("data/World_Cricketers.xlsx")
olympics = pd.read_excel("data/Indian_Olympic_Players.xlsx")


def dataframe_to_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = ""

        for column in df.columns:
            text += f"{column}: {row[column]}\n"

        documents.append(text)

    return documents


cricket_docs = dataframe_to_documents(cricket)
olympic_docs = dataframe_to_documents(olympics)

print("Number of Cricket Documents :", len(cricket_docs))
print("Number of Olympic Documents :", len(olympic_docs))

print("\nFirst Cricket Document\n")
print(cricket_docs[0])

print("\n---------------------------------\n")

print("First Olympic Document\n")
print(olympic_docs[0])