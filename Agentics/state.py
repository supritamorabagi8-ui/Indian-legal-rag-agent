from typing import TypedDict, List


class LegalState(TypedDict):
    query: str
    query_type: str
    retrieved_chunks: List[str]
    verification: str
    final_answer: str
