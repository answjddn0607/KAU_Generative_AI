from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

def keep_last_50_add_messages(existing: list, new: list) -> list:
    """
    add_messages 기능 + 최근 50개 메시지만 유지
    """
    merged = add_messages(existing, new)

    return merged[-50:]
class AgentState(TypedDict):
    """
    Agent State
    """
    messages: Annotated[list, keep_last_50_add_messages]
    tool_result: str | None
    iteration: int
    max_iterations: int

    query: str
    status: str

    rag_result: dict | None
    api_result: dict | None
    google_result: dict | None

    target_paper: dict | None
    related_papers: list | None

    user_interests: list | None
    recommendations: list | None

    final_result: dict | None