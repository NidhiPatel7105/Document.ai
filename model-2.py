# ✅ STEP 2: Import and parse an unstructured file (example: PDF)
from unstructured.partition.pdf import partition_pdf

# Replace 'example.pdf' with your actual file
elements = partition_pdf(filename="example.pdf")
texts = [el.text for el in elements if el.text and el.text.strip()]

# ✅ STEP 3: Function to embed using Ollama (you must run this locally)
import requests
import json

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )

    print("Status Code:", response.status_code)
    print("Raw Text:", response.text)

    data = response.json()
    if "embedding" not in data:
        raise ValueError(f"Missing 'embedding' key. Full response: {data}")
    
    return data["embedding"]

# WARNING: This may take time depending on model and hardware
embeddings = [get_embedding(text) for text in texts]

# ✅ STEP 4: Store embeddings in FAISS
import faiss
import numpy as np

# Convert to numpy array
embedding_vectors = np.array(embeddings).astype("float32")
dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_vectors)

# Map vector index to text
vector_to_text = {i: texts[i] for i in range(len(texts))}

# ✅ STEP 5: Search using a query
query = "What is the summary of this document?"
query_vector = np.array([get_embedding(query)]).astype("float32")
D, I = index.search(query_vector, k=3)

print("Top results:\n")
for i in I[0]:
    print(vector_to_text[i])
