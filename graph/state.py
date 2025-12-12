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
