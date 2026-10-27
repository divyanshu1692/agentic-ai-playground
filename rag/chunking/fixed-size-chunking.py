# fixed size chunking - Splits text into chunks based on a fixed number of characters or words.
def fixed_size_chunking(text, chunk_size=20):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


text = "Python is a programming language used for AI and machine learning."

chunks = fixed_size_chunking(text, 20)

for chunk in chunks:
    print(chunk)