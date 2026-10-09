from graph import graph


print("======================================")
print("   INDIAN LEGAL AGENTIC RAG SYSTEM")
print("======================================")

query = input("\nEnter your legal question: ")


initial_state = {
    "query": query,
    "query_type": "",
    "retrieved_chunks": [],
    "verification": "",
    "final_answer": ""
}


result = graph.invoke(initial_state)


print("\n======================================")
print("QUERY TYPE")
print("======================================")
print(result["query_type"])


print("\n======================================")
print("VERIFICATION")
print("======================================")
print(result["verification"])


print("\n======================================")
print("FINAL ANSWER")
print("======================================")
print(result["final_answer"])
