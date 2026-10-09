from langgraph.graph import StateGraph, START, END

from state import LegalState
from agents import (
    router_agent,
    retrieval_agent,
    verification_agent,
    answer_agent
)


# ==========================================
# 1. ROUTER NODE
# ==========================================

def router_node(state: LegalState):
    query = state["query"]

    query_type = router_agent(query)

    return {
        "query_type": query_type
    }


# ==========================================
# 2. RETRIEVAL NODE
# ==========================================

def retrieval_node(state: LegalState):
    query = state["query"]

    chunks = retrieval_agent(query)

    return {
        "retrieved_chunks": chunks
    }


# ==========================================
# 3. VERIFICATION NODE
# ==========================================

def verification_node(state: LegalState):
    query = state["query"]
    chunks = state["retrieved_chunks"]

    result = verification_agent(query, chunks)

    return {
        "verification": result
    }


# ==========================================
# 4. ANSWER NODE
# ==========================================

def answer_node(state: LegalState):
    query = state["query"]
    chunks = state["retrieved_chunks"]

    answer = answer_agent(query, chunks)

    return {
        "final_answer": answer
    }


# ==========================================
# BUILD LANGGRAPH
# ==========================================

builder = StateGraph(LegalState)

builder.add_node("router", router_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("verification", verification_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "router")
builder.add_edge("router", "retrieval")
builder.add_edge("retrieval", "verification")
builder.add_edge("verification", "answer")
builder.add_edge("answer", END)

graph = builder.compile()
