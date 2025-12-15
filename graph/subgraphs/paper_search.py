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


def extract_title_from_filename(filename: str) -> str:
    """
    파일명에서 실제 논문 제목 추출
    예: 2021_Artificial_intelligence_in_education__Ad_c0a8fe3a 
    → Artificial intelligence in education
    """
    import re
    
    # .pdf 제거
    name = filename.replace(".pdf", "")
    
    # 연도 제거 (앞의 4자리 숫자_)
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
        name = parts[1]
    
    # 마지막 ID 제거 (8자리 16진수)
    # 패턴: _로 시작하고 8자리 16진수로 끝남
    # 예: __Ad_c0a8fe3a, _c3df199c
    name = re.sub(r'_+[a-fA-F0-9]{8}$', '', name)
    
    # 언더스코어를 공백으로 변경
    title = name.replace("_", " ")
    
    return title.strip()


def find_paper_url_via_semantic_scholar(title: str) -> str:
    """
    Semantic Scholar API로 논문 URL 찾기
    """
    try:
        result = semantic_scholar_search_handler(
            SemanticScholarSearchInput(query=title, limit=1)
        )
        
        if result.get("count", 0) > 0:
            url = result["results"][0].get("url")
            if url:
                return url
        
        return None
        
    except Exception as e:
        print(f"[Semantic Scholar URL 검색 실패] {e}")
        return None


class PaperSearchNodes:
    """논문 검색 노드들"""
    
    def rag_node(self, state: AgentState) -> dict:
        result = rag_search_handler(RAGSearchInput(query=state["query"], top_k=5))
        print(f"[PAPER_SEARCH RAG] count: {result.get('count')}")

        if result.get("count", 0) == 0:
            print("[PAPER_SEARCH RAG] → not found")
            return {"rag_result": {"found": False}, "status": "not_found"}
        
        # URL 없는 논문에 대해 Semantic Scholar로 링크 찾기
        papers = result.get("results", [])
        for paper in papers:
            if not paper.get("url") or paper.get("url") == "":
                filename = paper.get("title", "")
                # 파일명에서 실제 논문 제목 추출
                actual_title = extract_title_from_filename(filename)
                print(f"[PAPER_SEARCH RAG] URL 없음 - Semantic Scholar 검색: {actual_title[:50]}...")
                
                url = find_paper_url_via_semantic_scholar(actual_title)
                if url:
                    paper["url"] = url
                    # print(f"[PAPER_SEARCH RAG] ✅ URL 획득: {url[:50]}...")
                else:
                    print(f"[PAPER_SEARCH RAG] ⚠️ URL 찾기 실패")
        
        return {
            "rag_result": {"found": True, "results": papers},
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