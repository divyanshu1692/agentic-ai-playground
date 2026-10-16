from groq import Groq
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") #384 - Sentence transformer model used to generate embeddings.

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

documents = [ #documents are the context of the knowledge base.
    "There are 24 paid leaves in a year.",
    "Employees can work from home up to 2 days per week with manager approval.",
    "Standard working hours are 8 hours per day, excluding breaks.",
    "Employees must inform their manager about unexpected absences.",
    "Performance reviews are conducted every 6 months.",
    "Employees must maintain the confidentiality of company information.",
    "Business expense claims must be submitted within 30 days.",
    "Employees must follow the company's code of conduct.",
    "Overtime work requires prior approval from the reporting manager.",
    "Employees must use strong passwords and enable multi-factor authentication.",
    "New employees must complete onboarding within their first 2 weeks.",
    "Employees are entitled to 12 paid sick leaves per year.",
    "Company laptops and equipment must be used responsibly.",
    "Employees must give at least 30 days' notice before resignation, subject to their employment agreement.",
    "Annual salary reviews are conducted based on performance and company policy."
]

embeddings = model.encode(documents) #generate embeddings for the documents.

def cosine_similarity(a, b): #cosine similarity is a measure of how similar or close two vectors are.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b));

def retrieve(query_embedding): #retrieve the most relevant document from the knowledge base.
    scores = []
    for i, document in enumerate(embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(query, context): #ask the LLM to answer the question based on the context.
    system_prompt = f""" Answer in one line only. Do not hallucinate. Use the following context to answer the question: {context} """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content

query = "How many leaves do i get?"
query_embedding = model.encode(query)

score, context = retrieve(query_embedding)
# print(score)
# print(context)
response = ask_llm(query, context)
print(response)