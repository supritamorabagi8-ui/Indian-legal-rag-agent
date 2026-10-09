import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load FAISS index
index = faiss.read_index("legal_vectors.index")


# Load legal chunks
with open("chunk_texts.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = content.split("\n" + "=" * 80 + "\n")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_legal_documents(query: str, k: int = 3):
    """
    Search the existing FAISS index
    and return the most relevant legal chunks.
    """

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks
