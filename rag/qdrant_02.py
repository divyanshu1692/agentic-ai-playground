# multi tenant rag using qdrant
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("Connected to Vector DB!")

# create collection
COLLECTION_NAME = "knowledge_base"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print("Deleting Collection", COLLECTION_NAME)
    client.delete_collection(COLLECTION_NAME)

# create collection
client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print("Collection Created: ", COLLECTION_NAME)

# add indexing
client.create_payload_index(
    collection_name = COLLECTION_NAME,
    field_name = "category",
    field_schema = PayloadSchemaType.KEYWORD
)

# load knowledge file
with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Document Loaded: ", len(documents))

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready")

texts = [document['text'] for document in documents]

embeddings = model.encode(texts)
print("Embedding genrated: ", {len(embeddings)})

# qdrant points
points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id = i + 1,
        vector = embedding.tolist(),
        payload = documents[i]
    )

    points.append(point)

# upload to qdrant
client.upsert(
    collection_name = COLLECTION_NAME,
    points = points
)

# search qdrant with filter
def search_with_filter(query, query_filter=None, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True,
        query_filter = query_filter
    ).points

    return results

# create filters
employee_filter = Filter(
    must = [
        FieldCondition(
            key = "category",
            match = MatchValue(value = "Leave & Attendance")
        )
    ]
)

# # test search
# query = "How many vacation days do i get?"

# results = search_with_filter(query, employee_filter, top_k=3)

# print("\n Search Results:", results)

# for result in results:
#     print(f"Score: {result.score}")
#     print(f"payload {result.payload["text"]}")
#     print()

# connect to groq
groq_client = Groq(
    api_key = GROQ_API_KEY
)

# ask_llm
def ask_llm(query, context): #ask the LLM to answer the question based on the context.
    system_prompt = f""" Answer in one line only. Do not hallucinate. 
    Use the following context to answer the question: {context}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content

# complete rag pipline
# query = "How many vacation days do i get?"
query = input("Ask anything... ")
results = search_with_filter(query, employee_filter, top_k=3)

# extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)

print("Context", context)

answer = ask_llm(query, context)

print(f"\nfinal Answer --------------")
print(answer)