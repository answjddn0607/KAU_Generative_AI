from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import agent_node, tools_node


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional router"""
    
    if state["iteration"] >= state["max_iterations"]:
        return "__end__"
    
    if state["tool_result"] is not None:
        return "tools"
    
    return "__end__"


def create_graph(interrupt: bool = True):
    builder = StateGraph(AgentState)
    
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    
    builder.set_entry_point("agent")
    
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    builder.add_edge("tools", "agent")
    
    memory = MemorySaver()
    
    if interrupt:
        graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"]
    )
    else:
        graph = builder.compile(
            checkpointer=memory
        )
    
    return graph
