# ⚖️ Indian Legal RAG System

An AI-powered legal question-answering system that uses Indian legal documents to provide simple, relevant, and context-based answers.

## 📌 Project Overview

The Indian Legal RAG System allows users to ask questions about Indian laws and the Constitution.

The system searches a large collection of legal documents using semantic search and retrieves the most relevant information. This information is then provided to a Groq LLM to generate a simple answer.

## 📚 Dataset

The system processes Indian legal documents in PDF and JSON formats.

The current dataset contains:

- 26,720+ PDF files
- 7 JSON files
- 442,528 text chunks

Important legal documents include:

- Constitution of India
- Bharatiya Nyaya Sanhita (BNS), 2023
- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023
- Bharatiya Sakshya Adhiniyam (BSA), 2023
- Constitutional documents and amendments
- Other Indian legal documents and case materials

## 🔄 System Workflow

Legal Documents
        ↓
Text Extraction
        ↓
Chunking
        ↓
Text Embeddings
        ↓
FAISS Vector Database
        ↓
Semantic Search
        ↓
Top Relevant Chunks
        ↓
RAG Context
        ↓
Groq LLM
        ↓
Final Answer

## ⚙️ Technologies Used

- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- FAISS
- NumPy
- Groq LLM
- PDF and JSON processing

## 📂 Main Files

- `chunk_code.py` – Extracts documents, creates chunks and embeddings
- `search.py` – Performs semantic search using FAISS
- `rag_search.py` – Retrieves legal context and generates the final answer
- `test_groq.py` – Tests Groq API connectivity
- `chunk_metadata.json` – Stores chunk metadata
- `chunk_texts.txt` – Stores processed text chunks
- `legal_vectors.index` – FAISS vector database

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt


##COMMADS
1. Open the project folder
D:
cd D:\Legal_Project


2. Check project files
dir


3. Process the complete legal dataset
python chunk_code.py

→ Extracts PDF/JSON text
→ Creates chunks
→ Generates embeddings
→ Creates FAISS index


4. Check FAISS index
dir legal_vectors.index


5. Check embedding checkpoint
dir embeddings_checkpoint.npy


6. Test Groq API
python test_groq.py


7. Test semantic search
python search.py

→ Enter your legal question
→ Retrieves relevant chunks


8. Run the complete RAG system
python rag_search.py

→ Enter your legal question
→ Retrieves relevant chunks
→ Sends context to Groq LLM
→ Generates final answer


## Agentic RAG Implementation

Added an agentic workflow to the existing Indian Legal RAG system using LangGraph.

### What I Implemented

- Created a separate `Agentics` module for the agentic workflow.
- Implemented a **Router Agent** to classify legal questions into:
  - Constitution
  - Criminal Law
  - Civil Law
  - General Legal
- Implemented a **Retrieval Agent** to retrieve relevant legal documents using the existing FAISS index and Sentence Transformer model.
- Implemented a **Verification Agent** to check whether the retrieved context is sufficient to answer the user's question.
- Implemented an **Answer Agent** to generate a simple answer using only the retrieved legal context.
- Created a shared state using `TypedDict` to pass information between agents.
- Used **LangGraph** to connect the agents into a sequential workflow.
- Tested the complete workflow successfully with Indian legal questions.

### Agentic Workflow 


                    User Question
                         │
                         ▼
                  ┌─────────────┐
                  │ Router Agent│
                  └──────┬──────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieval Agent │
                │ FAISS + MiniLM  │
                └────────┬────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Verification Agent  │
              └──────────┬──────────┘
                         │
                      VALID
                         │
                         ▼
                 ┌──────────────┐
                 │ Answer Agent │
                 └──────┬───────┘
                        │
                        ▼
                   Final Answer