import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2") #384

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b));

documents = [
    "There are 24 paid leaves in a year",
    "apple is too sweet"
]

document_embeddings = model.encode(documents)

def retrieve(query_embedding):
    scores = []
    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(message, context):
    system_prompt = f"""Answer in one line only. Answer only based on the context provided. Do not provide the extra information in the response. Context: {context}"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.append({
        "role": "user",
        "content": message
    })

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content;

query = "How many vacation do i have?"

query_embedding = model.encode(query)

score, context = retrieve(query_embedding)

answer = ask_llm(query, context)

print("answer: ", answer)