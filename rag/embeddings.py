# Embeddings are vectors that represent the meaning of a text or in the simplest terms a vector of numbers that represent the meaning of a text or features of a text.
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") #384

def cosine_similarity(a, b): #cosine similarity is a measure of how similar or close two vectors are.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b));

# text = "Hello, how are you?"

# embedding = model.encode(text)
# print(embedding.shape)
# print(embedding)

text1 = "There are 24 paid leaves in a year"
text2 = "apple is too sweet"

v1 = model.encode(text1)
v2 = model.encode(text2)

print(cosine_similarity(v1, v2))