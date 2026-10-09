import os
from groq import Groq
from dotenv import load_dotenv

from tools import retrieve_legal_documents

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", "add your key here"))


# ==========================================
# 1. QUERY ROUTER AGENT
# ==========================================

def router_agent(query: str) -> str:
    """
    Identifies the type of legal question.
    """

    prompt = f"""
You are a legal query router.

Classify the following Indian legal question into one category:

- Constitution
- Criminal Law
- Civil Law
- General Legal

Return ONLY the category name.

Question:
{query}
"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ==========================================
# 2. RETRIEVAL AGENT
# ==========================================

def retrieval_agent(query: str):
    """
    Uses the existing FAISS RAG system
    to retrieve relevant legal documents.
    """

    return retrieve_legal_documents(query)


# ==========================================
# 3. VERIFICATION AGENT
# ==========================================

def verification_agent(query: str, retrieved_chunks: list) -> str:
    """
    Checks whether the retrieved information
    is relevant to the user's question.
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a legal evidence verification agent.

Check whether the provided legal context contains
enough relevant information to answer the question.

Question:
{query}

Legal Context:
{context}

Return ONLY one of these:
VALID
INSUFFICIENT
"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ==========================================
# 4. ANSWER AGENT
# ==========================================

def answer_agent(query: str, retrieved_chunks: list) -> str:
    """
    Generates the final answer using only
    the retrieved legal information.
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an Indian legal information assistant.

Answer the user's question using ONLY the
provided legal context.

Do not invent laws, sections, articles,
or legal information.

Explain the answer in simple language.

Question:
{query}

Legal Context:
{context}
"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {
                "role": "system",
                "content": "You provide simple, source-based information about Indian law."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
