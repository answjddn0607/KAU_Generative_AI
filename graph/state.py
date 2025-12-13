from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Agent State
    """
    messages: Annotated[list, add_messages]
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
