import sys
from pathlib import Path

TRANSPOTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TRANSPOTER_ROOT))

from graph.state import AgentState
from tools.tool_definitions import (
    rag_search_handler, RAGSearchInput,
    memory_read_handler, MemoryReadInput
)


class RecommendationNodes:
    """논문 추천 노드들"""
    
    def get_interests_node(self, state: AgentState) -> dict:
        result = memory_read_handler(MemoryReadInput(query="interest", top_k=5))
        interests = [m.get("content", "") for m in result.get("results", [])]
        
        if not interests:
            interests = [state.get("query", "AI research")]
        
        print(f"[RECOMMENDATION] interests: {interests}")
        return {"user_interests": interests}
    
    def recommend_node(self, state: AgentState) -> dict:
        interests = state.get("user_interests", ["AI research"])
        query = " ".join(interests[:3])
        
        result = rag_search_handler(RAGSearchInput(query=query, top_k=5))
        recommendations = result.get("results", [])

        if not recommendations:
            from tools.tool_definitions import semantic_scholar_search_handler, SemanticScholarSearchInput
            api_result = semantic_scholar_search_handler(
                SemanticScholarSearchInput(query=query, limit=5)
            )
            recommendations = api_result.get("results", [])
        
        print(f"[RECOMMENDATION] count: {len(recommendations)}개")
        return {
            "recommendations": recommendations,
            "final_result": {
                "interests": interests,
                "recommendations": recommendations
            },
            "status": "success" if recommendations else "not_found"
        }