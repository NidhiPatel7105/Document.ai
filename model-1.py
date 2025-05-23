# import os
# from PIL import Image
# import pytesseract
# import ollama

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# upload_path = "C:\\Users\\nidhi\\OneDrive\\Desktop\\Task-3\\tata.jpg"  # can be file or folder

# if os.path.isdir(upload_path):
#     filenames = [os.path.join(upload_path, f) for f in os.listdir(upload_path)]
# elif os.path.isfile(upload_path):
#     filenames = [upload_path]
# else:
#     raise ValueError("Invalid path")

# def extract_text_from_image(image_path):
#     image = Image.open(image_path)
#     text = pytesseract.image_to_string(image)
#     return text.strip()

# # Process images
# texts = []
# for filepath in filenames:
#     if filepath.lower().endswith((".jpg", ".jpeg", ".png")):
#         extracted_text = extract_text_from_image(filepath)
#         if extracted_text:
#             texts.append(extracted_text)

# # STEP 3: Generate Embeddings
# embeddings = [get_embedding(text) for text in texts]

# # ✅ STEP 4: Store embeddings in FAISS
# import faiss
# import numpy as np

# embedding_vectors = np.array(embeddings).astype("float32")
# dimension = embedding_vectors.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embedding_vectors)

# # Map vector index to text
# vector_to_text = {i: texts[i] for i in range(len(texts))}

# # ✅ STEP 5: Query
# query = "What is the summary of this document?"
# query_vector = np.array([get_embedding(query)]).astype("float32")
# D, I = index.search(query_vector, k=3)

# print("Top results:\n")
# for i in I[0]:
#     print(vector_to_text[i])
import os
from PIL import Image
import pytesseract
import ollama  

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ollama embedding function
def get_embedding(text, model='nomic-embed-text'):
    response = ollama.embeddings(
        model=model,
        prompt=text
    )
    return response['embedding']

# Image file or folder
upload_path = "C:\\Users\\nidhi\\OneDrive\\Desktop\\Task-3\\tata.jpg"

# Handle file/folder
if os.path.isdir(upload_path):
    filenames = [os.path.join(upload_path, f) for f in os.listdir(upload_path)]
elif os.path.isfile(upload_path):
    filenames = [upload_path]
else:
    raise ValueError("Invalid path")

# Extract text
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

texts = []
for filepath in filenames:
    if filepath.lower().endswith((".jpg", ".jpeg", ".png")):
        extracted_text = extract_text_from_image(filepath)
        if extracted_text:
            texts.append(extracted_text)

# STEP 3: Generate Embeddings
embeddings = [get_embedding(text) for text in texts]

# STEP 4: Store embeddings in FAISS
import faiss
import numpy as np

embedding_vectors = np.array(embeddings).astype("float32")
dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_vectors)

# STEP 5: Query
vector_to_text = {i: texts[i] for i in range(len(texts))}

query = "What is the summary of this document?"
query_vector = np.array([get_embedding(query)]).astype("float32")
D, I = index.search(query_vector, k=3)

# print("Top results:\n")
# for i in I[0]:
#     print(vector_to_text[i])
print("Top results:\n")
for i in I[0]:
    if i == -1:
        continue  # Skip invalid result
    print(vector_to_text[i])
