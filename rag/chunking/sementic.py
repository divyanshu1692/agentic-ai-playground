# Semantic chunking splits text when its meaning or topic changes, rather than after a fixed number of words or characters.


import re
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunking(text, threshold=0.45):
    # Step 1: Split the document into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]

    if not sentences:
        return []

    # Step 2: Generate an embedding for each sentence
    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current_chunk = [sentences[0]]

    # Step 3: Compare adjacent sentence embeddings
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i - 1], embeddings[i])

        # Step 4: Split when meaning changes enough
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


text = (
    "Python supports functions and loops. "
    "Python is useful for data processing. "
    "Embeddings convert text into numerical vectors. "
    "Vector databases store embeddings for similarity search."
)

chunks = semantic_chunking(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}\n")
