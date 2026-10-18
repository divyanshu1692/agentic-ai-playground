import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# connect to qdrant

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("Connected to Vector DB!")

# create collection
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384


# delete collection if already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print(f"Created Collection {COLLECTION_NAME}")
print(f"Vector Size: {EMBEDDING_SIZE}")

# load our knowledge
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} document")

# create embeddings
print("Loading embeddings model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model ready!")

embeddings = model.encode(documents)

print(f"Genrated {len(embeddings)} embeddings")
print(f"Embedding Size: {len(embeddings[0])}")

# create qdrant points

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id = i+1,
        vector=embedding.tolist(),
        payload={
            "text": documents[i]
        }
    )

    points.append(point)

# upload to qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points = points
)

print(f"Uploaded {len(points)} documents to qdrant")


# search qdrant
def search(query, top_k=3):
    # convert query into an embedding
    query_vector = model.encode(query).tolist()

    # search qdrant for similar vectors
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k, 
        with_payload = True
    ).points

    return results

# test search
query = "How many vacation days do i get?"

results = search(query, top_k=3)

print("\n Search Results:", results)

for result in results:
    print(f"Score: {result.score}")
    print(f"payload {result.payload["text"]}")
    print()

# connect to groq
groq_client = Groq(
    api_key=GROQ_API_KEY
)

# ask_llm
def ask_llm(query, context): #ask the LLM to answer the question based on the context.
    system_prompt = f""" Answer in one line only. Do not hallucinate. 
    Use the following context to answer the question: {context} 
    Note:
    Apart form the employee policy directly return that.. i can't answer these question..
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
query = input("Ask something to the groq...")
results = search(query, top_k=3)

# extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)

answer = ask_llm(query, context)

print(f"\nfinal Answer --------------")
print(answer)

