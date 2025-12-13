from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import agent_node, tools_node
from .subgraphs.paper_search import PaperSearchNodes
from .subgraphs.paper_analysis import PaperAnalysisNodes
from .subgraphs.recommendation import RecommendationNodes
from langchain_core.messages import ToolMessage
import json

memory = MemorySaver()


# ============ Setup 노드들 ============
def setup_paper_search(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]
    return {"query": tool_call["args"].get("query", "")}

def setup_paper_analysis(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]
    return {"query": tool_call["args"].get("query", "")}

def setup_recommendation(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]
    return {"query": tool_call["args"].get("query", "AI research")}


# ============ Finish 노드들 ============
def finish_paper_search(state: AgentState) -> dict:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            break
    
    source = "none"
    if state.get("rag_result", {}).get("found"):
        source = "rag"
    elif state.get("api_result", {}).get("found"):
        source = "semantic_scholar_api"
    elif state.get("google_result", {}).get("found"):
        source = "google"
    
    print(f"[PAPER_SEARCH] source: {source}, status: {state.get('status')}")
    
    return {
        "messages": [ToolMessage(
            content=json.dumps(state.get("final_result", {}), ensure_ascii=False),
            tool_call_id=tool_call["id"]
        )]
    }

def finish_paper_analysis(state: AgentState) -> dict:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            break
    
    source = "none"
    if state.get("rag_result", {}).get("found"):
        source = "rag"
    elif state.get("api_result", {}).get("found"):
        source = "semantic_scholar_api"
    
    print(f"[PAPER_ANALYSIS] source: {source}, status: {state.get('status')}")
    
    return {
        "messages": [ToolMessage(
            content=json.dumps(state.get("final_result", {}), ensure_ascii=False),
            tool_call_id=tool_call["id"]
        )]
    }

def finish_recommendation(state: AgentState) -> dict:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            break
    
    print(f"[RECOMMENDATION] status: {state.get('status')}")
    
    return {
        "messages": [ToolMessage(
            content=json.dumps(state.get("final_result", {}), ensure_ascii=False),
            tool_call_id=tool_call["id"]
        )]
    }


# ============ 그래프 생성 ============
def create_graph(interrupt: bool = True):
    builder = StateGraph(AgentState)
    
    # 메인 노드
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    
    # Paper Search 노드들
    ps = PaperSearchNodes()
    builder.add_node("ps_setup", setup_paper_search)
    builder.add_node("ps_rag", ps.rag_node)
    builder.add_node("ps_api", ps.api_node)
    builder.add_node("ps_google", ps.google_node)
    builder.add_node("ps_finish", finish_paper_search)
    
    # Paper Analysis 노드들
    pa = PaperAnalysisNodes()
    builder.add_node("pa_setup", setup_paper_analysis)
    builder.add_node("pa_rag", pa.rag_node)
    builder.add_node("pa_api", pa.api_node)
    builder.add_node("pa_ask_user", pa.ask_user_node)
    builder.add_node("pa_finish", finish_paper_analysis)
    
    # Recommendation 노드들
    rec = RecommendationNodes()
    builder.add_node("rec_setup", setup_recommendation)
    builder.add_node("rec_interests", rec.get_interests_node)
    builder.add_node("rec_recommend", rec.recommend_node)
    builder.add_node("rec_finish", finish_recommendation)
    
    # 엔트리 포인트
    builder.set_entry_point("agent")
    
    # Agent 라우팅
    def route_agent(state: AgentState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            return "__end__"
        
        last_msg = state["messages"][-1]
        
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return "__end__"
        
        tool_name = last_msg.tool_calls[0]["name"]
        
        if tool_name == "paper_search":
            return "ps_setup"
        elif tool_name == "paper_analysis":
            return "pa_setup"
        elif tool_name == "paper_recommendation":
            return "rec_setup"
        
        return "tools"
    
    builder.add_conditional_edges("agent", route_agent)
    builder.add_edge("tools", "agent")
    
    # Paper Search 플로우: RAG → API → Google
    builder.add_edge("ps_setup", "ps_rag")
    builder.add_conditional_edges("ps_rag", 
        lambda s: "ps_finish" if s.get("rag_result", {}).get("found") else "ps_api")
    builder.add_conditional_edges("ps_api", 
        lambda s: "ps_finish" if s.get("api_result", {}).get("found") else "ps_google")
    builder.add_edge("ps_google", "ps_finish")
    builder.add_edge("ps_finish", "agent")
    
    # Paper Analysis 플로우: RAG → API → ask_user (interrupt)
    builder.add_edge("pa_setup", "pa_rag")
    builder.add_conditional_edges("pa_rag", 
        lambda s: "pa_finish" if s.get("rag_result", {}).get("found") else "pa_api")
    builder.add_conditional_edges("pa_api", 
        lambda s: "pa_finish" if s.get("api_result", {}).get("found") else "pa_ask_user")
    builder.add_edge("pa_ask_user", "pa_finish")
    builder.add_edge("pa_finish", "agent")
    
    # Recommendation 플로우: interests → recommend
    builder.add_edge("rec_setup", "rec_interests")
    builder.add_edge("rec_interests", "rec_recommend")
    builder.add_edge("rec_recommend", "rec_finish")
    builder.add_edge("rec_finish", "agent")
    
    # 컴파일
    interrupt_nodes = ["pa_ask_user"] if interrupt else []
    return builder.compile(
        checkpointer=memory,
        interrupt_before=interrupt_nodes if interrupt_nodes else None
    )