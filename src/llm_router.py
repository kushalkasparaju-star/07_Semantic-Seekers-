import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def route_query(question):

    prompt = f"""
You are an intelligent routing assistant.

Your job is to classify the user's question into ONLY ONE of these categories.

cricket
olympics
both
unknown

Rules:

- If the question is only about cricket players -> cricket
- If the question is only about Olympic players -> olympics
- If the question needs both datasets -> both
- If the question is unrelated to the datasets -> unknown

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


while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    result = route_query(question)

    print("\nRoute =", result)