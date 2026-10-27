# In this technique, some text from the previous chunk is repeated in the next chunk.

def chunk_with_overlap(text, chunk_size=5, overlap=2):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


text = "Python is a programming language used for AI and machine learning."

print(chunk_with_overlap(text, 5, 2))