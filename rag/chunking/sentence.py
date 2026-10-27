# Splits text into complete sentences instead of cutting through words or sentences.
import re

def sentence_chunking(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence for sentence in sentences if sentence]


text = (
    "Python is easy to learn. "
    "It is widely used in AI. "
    "RAG combines retrieval with generation."
)

chunks = sentence_chunking(text)

for chunk in chunks:
    print(chunk)