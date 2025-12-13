import sys
from pathlib import Path

TRANSPOTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TRANSPOTER_ROOT))

from graph.state import AgentState
from tools.tool_definitions import (
    rag_search_handler, RAGSearchInput,
    semantic_scholar_search_handler, SemanticScholarSearchInput
)

class PaperAnalysisNodes:
    """논문 분석 노드들"""
    
    def rag_node(self, state: AgentState) -> dict:
        print(f"[PAPER_ANALYSIS RAG] query: {state['query']}")
        result = rag_search_handler(RAGSearchInput(query=state["query"], top_k=3))
        print(f"[PAPER_ANALYSIS RAG] count: {result.get('count')}")
        
        if result.get("count", 0) == 0:
            print("[PAPER_ANALYSIS RAG] → not found")
            return {"rag_result": {"found": False}, "status": "not_found"}
        
        target = result.get("results", [])[0]
        print(f"[PAPER_ANALYSIS RAG] → found: {target.get('title')}")
        return {
            "rag_result": {"found": True},
            "target_paper": target,
            "final_result": {
                "target_paper": target,
                "analysis": {
                    "title": target.get("title"),
                    "authors": target.get("authors"),
                    "abstract": target.get("abstract"),
                }
            },
            "status": "success"
        }
    
    def api_node(self, state: AgentState) -> dict:
        print(f"[PAPER_ANALYSIS API] query: {state['query']}")
        result = semantic_scholar_search_handler(
            SemanticScholarSearchInput(query=state["query"], limit=3)
        )
        print(f"[PAPER_ANALYSIS API] count: {result.get('count')}")
        
        if result.get("count", 0) == 0:
            print("[PAPER_ANALYSIS API] → not found")
            return {"api_result": {"found": False}, "status": "not_found"}
        
        target = result["results"][0]
        print(f"[PAPER_ANALYSIS API] → found: {target.get('title')}")
        return {
            "api_result": {"found": True},
            "target_paper": target,
            "final_result": {
                "target_paper": target,
                "analysis": {
                    "title": target.get("title"),
                    "authors": target.get("authors"),
                    "abstract": target.get("abstract"),
                }
            },
            "status": "success"
        }
    
    def ask_user_node(self, state: AgentState) -> dict:
        return {
            "final_result": {
                "message": f"'{state['query']}' 관련 논문을 찾지 못했습니다. 정확한 제목을 입력해주세요."
            },
            "status": "need_input"
        }