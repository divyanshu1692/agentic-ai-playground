# This method tries to split text using natural boundaries in a preferred order, such as paragraphs, then lines, then sentences, then words or characters.

# pip install langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is a programming language.
It is widely used in AI and machine learning.

RAG retrieves relevant documents.
The retrieved context helps an LLM answer questions.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n")