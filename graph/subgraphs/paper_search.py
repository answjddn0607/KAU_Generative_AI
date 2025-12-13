import sys
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

TRANSPOTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TRANSPOTER_ROOT))

from graph.state import AgentState
from tools.tool_definitions import (
    rag_search_handler, RAGSearchInput,
    google_search_handler, GoogleSearchInput,
    semantic_scholar_search_handler, SemanticScholarSearchInput
)
from tools.reranker import rerank_results


class PaperSearchNodes:
    """논문 검색 노드들"""
    
    def rag_node(self, state: AgentState) -> dict:
        result = rag_search_handler(RAGSearchInput(query=state["query"], top_k=5))
        print(f"[PAPER_SEARCH RAG] count: {result.get('count')}")

        if result.get("count", 0) == 0:
            print("[PAPER_SEARCH RAG] → not found")
            return {"rag_result": {"found": False}, "status": "not_found"}
        
        return {
            "rag_result": {"found": True, "results": result.get("results", [])},
            "final_result": result,
            "status": "success"
        }
    
    def api_node(self, state: AgentState) -> dict:
        result = semantic_scholar_search_handler(
            SemanticScholarSearchInput(query=state["query"], limit=5)
        )
        print(f"[PAPER_SEARCH API] count: {result.get('count')}")
        
        if result.get("count", 0) == 0:
            print("[PAPER_SEARCH API] → not found")
            return {"api_result": {"found": False}, "status": "not_found"}
        
        return {
            "api_result": {"found": True, "results": result["results"]},
            "final_result": result,
            "status": "success"
        }
    
    def google_node(self, state: AgentState) -> dict:
        result = google_search_handler(
            GoogleSearchInput(query=state["query"] + " paper")
        )
        
        if not result.get("results"):
            return {
                "google_result": {"found": False},
                "final_result": {"message": "모든 소스에서 못 찾음"},
                "status": "not_found"
            }
        
        # 리랭킹용 text 필드 추가
        docs = [{"text": r["snippet"], **r} for r in result["results"]]
        
        # Cross-Encoder 리랭킹 적용
        reranked = rerank_results(state["query"], docs, top_k=5)
        
        # 유사도 검사: relevance_score 임계값 확인
        if not reranked or max(r.get("relevance_score", -1.0) for r in reranked) < 0.5:
            print("[PAPER_SEARCH GOOGLE] 유사도 낮음 - 관련 결과 없음")
            return {
                "google_result": {"found": False},
                "final_result": {"message": "모든 소스에서 못 찾음", "reason": "낮은 유사도"},
                "status": "not_found"
            }
        
        # text 필드 제거
        for doc in reranked:
            doc.pop("text", None)
        
        # 리랭킹된 결과로 교체
        result["results"] = reranked
        
        print(f"[PAPER_SEARCH GOOGLE] 리랭킹 완료: {len(reranked)}개")
        
        return {
            "google_result": {"found": True, "results": result["results"]},
            "final_result": result,
            "status": "success"
        }