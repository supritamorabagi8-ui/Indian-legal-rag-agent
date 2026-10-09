import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. LOAD FAISS INDEX
# ==========================================

index = faiss.read_index("legal_vectors.index")

# ==========================================
# 2. LOAD CHUNKS
# ==========================================

with open("chunk_texts.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = content.split("\n" + "=" * 80 + "\n")

# Remove empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print("Total chunks loaded:", len(chunks))

# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# 4. TAKE USER QUERY
# ==========================================

query = input("\nEnter your question: ")

# ==========================================
# 5. CREATE QUERY EMBEDDING
# ==========================================

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

# ==========================================
# 6. SEARCH FAISS
# ==========================================

k = 3

distances, indices = index.search(query_embedding, k)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("TOP RELEVANT RESULTS")
print("======================================")

for i, idx in enumerate(indices[0]):

    print(f"\nResult {i + 1}")
    print("-" * 50)

    print(chunks[idx])

    print("\nDistance:", distances[0][i])