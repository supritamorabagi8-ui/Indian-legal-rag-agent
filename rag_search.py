import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# 1. GROQ API KEY
# ==========================================

client = Groq(
    api_key="add your key here"
)

# ==========================================
# 2. LOAD FAISS INDEX
# ==========================================

index = faiss.read_index("legal_vectors.index")

# ==========================================
# 3. LOAD CHUNKS
# ==========================================

with open("chunk_texts.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = content.split("\n" + "=" * 80 + "\n")

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print("Total chunks loaded:", len(chunks))

# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# 5. GET USER QUESTION
# ==========================================

query = input("\nEnter your legal question: ")

# ==========================================
# 6. CREATE QUESTION EMBEDDING
# ==========================================

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

# ==========================================
# 7. SEARCH FAISS
# ==========================================

k = 3

distances, indices = index.search(query_embedding, k)

# ==========================================
# 8. GET RELEVANT CHUNKS
# ==========================================

relevant_chunks = []

for idx in indices[0]:

    if idx < len(chunks):
        relevant_chunks.append(chunks[idx])

context = "\n\n".join(relevant_chunks)

# ==========================================
# 9. SEND CONTEXT TO GROQ
# ==========================================

prompt = f"""
You are a legal information assistant for Indian law.

Answer the user's question using ONLY the legal information
provided in the context below.

If the answer is not available in the context, say:
"I could not find sufficient information in the provided legal documents."

Do not invent laws, sections, articles, or legal information.

Explain the answer in simple and clear language.

User Question:
{query}

Legal Context:
{context}
"""

response = client.chat.completions.create(
    model="groq/compound-mini",
    
    messages=[
        {
            "role": "system",
            "content": "You are a helpful Indian legal information assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)

# ==========================================
# 10. DISPLAY FINAL ANSWER
# ==========================================

answer = response.choices[0].message.content

print("\n======================================")
print("FINAL ANSWER")
print("======================================")

print(answer)

# ==========================================
# 11. DISPLAY SOURCES
# ==========================================

print("\n======================================")
print("SOURCES")
print("======================================")

for i, idx in enumerate(indices[0]):

    if idx < len(chunks):

        first_line = chunks[idx].split("\n")[0]

        print(f"{i + 1}. {first_line}")