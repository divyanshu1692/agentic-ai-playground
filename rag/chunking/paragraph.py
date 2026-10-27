# Splits text using paragraph breaks.

def paragraph_chunking(text):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


text = """Python is a programming language.

It is widely used in AI and machine learning.

RAG retrieves relevant information before generating an answer."""

print(paragraph_chunking(text))