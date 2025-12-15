from typing import Type, Callable, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import os
import requests
import uuid
from datetime import datetime

from .chroma_client import get_memory_collection, get_rag_collection
from .reranker import rerank_results


# -------------------------------
#  ToolSpec 정의
# -------------------------------
class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: Type[BaseModel]                  # Pydantic 모델 타입
    handler: Callable[[BaseModel], Dict[str, Any]]

# -------------------------------
# 0. 기본 툴
#   calculator, 날씨 API
# -------------------------------

class CalculatorInput(BaseModel):
    a: float = Field(..., description="First operand")
    op: str = Field(..., pattern=r"^[+\-*/]$", description="Operator: +, -, *, /")
    b: float = Field(..., description="Second operand")

def calculator(input: CalculatorInput) -> Dict[str, Any]:
    if input.op == '+':
        val = input.a + input.b
    elif input.op == '-':
        val = input.a - input.b
    elif input.op == '*':
        val = input.a * input.b
    elif input.op == '/':
        if input.b == 0:
            raise RuntimeError("Division by zero")
        val = input.a / input.b
    else:
        raise RuntimeError(f"Unsupported operator: {input.op}")
    return {"expression": f"{input.a} {input.op} {input.b}", "value": val}

class GetWeatherInput(BaseModel):
    city: str = Field(..., description="City name in English, e.g., 'Seoul', 'Busan', 'Tokyo'")
    unit: str = Field(default="C", description="Temperature unit 'C' or 'F'")


# -------------------------------
# 1. Google CSE 검색 툴 구현
#    (실제 서비스 호출 + Pydantic 인자)
# -------------------------------

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")


def google_cse_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    if GOOGLE_CSE_API_KEY is None or GOOGLE_CSE_CX is None:
        raise RuntimeError(
            "GOOGLE_CSE_API_KEY 또는 GOOGLE_CSE_CX 환경변수가 설정되지 않았습니다."
        )

    if not (1 <= num_results <= 10):
        raise ValueError("num_results는 1~10 사이여야 합니다. (CSE API 제한)")

    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": num_results,
    }

    resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    results: List[Dict[str, Any]] = []
    for item in items:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return results


class GoogleSearchInput(BaseModel):
    query: str = Field(..., description="검색 질의 문자열")
    num_results: int = Field(
        5,
        ge=1,
        le=10,
        description="가져올 검색 결과 개수 (1~10, 기본값 5)",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query는 공백이 아닌 문자열이어야 합니다.")
        return v


def google_search_handler(args: GoogleSearchInput) -> Dict[str, Any]:
    results = google_cse_search(args.query, num_results=args.num_results)
    return {
        "results": results,
        "source": "google_cse",
    }

# -------------------------------
# 2. 메모리 관련 툴
#    (read & write)
# -------------------------------

## Write ##
class MemoryWriteInput(BaseModel):
    content: str = Field(..., description = "저장할 내용")
    memory_type: str = Field("episodic", description="profile / episodic / knowledge")
    importance: int = Field(3, ge = 1, le =5, description = "중요도 (1~5, 기본 값 3)")
    tags: List[str] = Field(default_factory = list, description = "태그 목록 선택")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str :
        if not v.strip():
            raise ValueError("content는 공백이 아닌 문자열이어야 합니다.")
        return v
    
def memory_write_handler(args: MemoryWriteInput) -> Dict[str, Any]:
    collection = get_memory_collection()

    doc_id = str(uuid.uuid4())

    metadata = {
        "created_at":datetime.now().isoformat(),
        "memory_type": args.memory_type,
        "importance": args.importance,
        "tags": ",".join(args.tags) if args.tags else ""
    }

    collection.add(
        ids=[doc_id],
        documents=[args.content],
        metadatas=[metadata]
    )
    return{
        "status" : "success",
        "message": "메모리에 저장되었습니다.",
        "id": doc_id,
        "content": args.content,
        "memory_type": args.memory_type,
        "importance": args.importance
    }


## Read ##
class MemoryReadInput(BaseModel):
    query: str = Field(..., description = "검색 질의")
    top_k : int = Field(3, ge = 1, le = 10, description = "반환 결과 수(1~10, 기본 값 3)")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query는 공백이 아닌 문자열이어야 합니다.")
        return v
    
def memory_read_handler(args: MemoryReadInput) -> Dict[str,Any]:
    collection = get_memory_collection()

    results = collection.query(
        query_texts = [args.query],
        n_results = args.top_k
    )

    memories = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            memory = {
                "content": doc,
                "id": results["ids"][0][i] if results["ids"] else None,
            }
            
            if results["metadatas"] and results["metadatas"][0]:
                meta = results["metadatas"][0][i]
                memory["created_at"] = meta.get("created_at")
                memory["tags"] = meta.get("tags", "").split(",") if meta.get("tags") else []
            
            if results["distances"] and results["distances"][0]:
                memory["distance"] = results["distances"][0][i]
            
            memories.append(memory)
    
    return {
        "query": args.query,
        "results": memories,
        "count": len(memories)
    }

# -------------------------------
# 3. RAG
#    Index
# -------------------------------
class RAGIndexInput(BaseModel):
    title: str = Field(..., description="논문 제목")
    abstract: str = Field(..., description="논문 초록")
    authors: List[str] = Field(default_factory=list, description="저자 목록")
    source: str = Field("", description="출처 (파일명 등)")

    @field_validator("abstract")
    @classmethod
    def abstract_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("abstract는 공백이 아닌 문자열이어야 합니다.")
        return v


def rag_index_handler(args: RAGIndexInput) -> Dict[str, Any]:
    collection = get_rag_collection()
    
    doc_id = str(uuid.uuid4())
    
    metadata = {
        "title": args.title,
        "authors": ",".join(args.authors) if args.authors else "",
        "source": args.source,
        "indexed_at": datetime.now().isoformat()
    }
    
    collection.add(
        ids=[doc_id],
        documents=[args.abstract],
        metadatas=[metadata]
    )
    
    return {
        "status": "success",
        "message": "논문이 인덱싱되었습니다.",
        "id": doc_id,
        "title": args.title
    }

# -------------------------------
# 3. RAG
#    Search
# -------------------------------

class RAGSearchInput(BaseModel):
    query: str = Field(..., description="검색 질의")
    top_k: int = Field(5, ge=1, le=10, description="반환할 결과 수 (1~10, 기본값 5)")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query는 공백이 아닌 문자열이어야 합니다.")
        return v


def rag_search_handler(args: RAGSearchInput) -> Dict[str, Any]:
    
    print(f"[RAG Search] 입력 쿼리: '{args.query}'") 
    
    collection = get_rag_collection()
    
    # 초기 검색: top_k의 2배 가져오기
    initial_k = min(args.top_k * 2, 20)
    
    results = collection.query(
        query_texts=[args.query],
        n_results=initial_k
    )
    
    papers = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            paper = {
                "abstract": doc,
                "text": doc,  # ← 이 줄 추가
                "id": results["ids"][0][i] if results["ids"] else None,
            }
            
            if results["metadatas"] and results["metadatas"][0]:
                meta = results["metadatas"][0][i]
                paper["title"] = meta.get("title")
                paper["authors"] = meta.get("authors", "").split(",") if meta.get("authors") else []
                paper["source"] = meta.get("source")
                paper["indexed_at"] = meta.get("indexed_at")
            
            if results["distances"] and results["distances"][0]:
                paper["distance"] = results["distances"][0][i]
            
            papers.append(paper)
    
    # 유사도 임계값 0.5
    if not papers or min(p.get("distance", 1.0) for p in papers) > 0.5:
        return {
            "query": args.query,
            "results": [],
            "count": 0,
            "reranked": False,
            "reason": "신뢰도 낮음"
        }
    
    # Cross-Encoder 리랭킹
    papers = rerank_results(args.query, papers, top_k=args.top_k)
    
    for paper in papers:
        paper.pop('text', None)
    return {
        "query": args.query,
        "results": papers,
        "count": len(papers),
        "reranked": True
    }

# -------------------------------
# 4. Semantic Scholar 
#    논문 검색 툴
# -------------------------------

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


class SemanticScholarSearchInput(BaseModel):
    query: str = Field(..., description="검색어 (논문 제목, 저자명, 키워드 조합 가능)")
    year_from: int | None = Field(None, ge=1900, description="검색 시작 연도")
    year_to: int | None = Field(None, ge=1900, description="검색 종료 연도")
    min_citations: int = Field(0, ge=0, description="최소 인용수")
    limit: int = Field(10, ge=1, le=50, description="반환할 논문 수")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query는 공백이 아닌 문자열이어야 합니다.")
        return v


def semantic_scholar_search_handler(args: SemanticScholarSearchInput) -> Dict[str, Any]:

    print(f"[Sematic] 입력 쿼리: '{args.query}'") 

    try:
        params = {
            "query": args.query,
            "limit": args.limit * 2,  
            "fields": "paperId,title,year,authors,abstract,citationCount,url",
        }

        if args.year_from and args.year_to:
            params["year"] = f"{args.year_from}-{args.year_to}"
        elif args.year_from:
            params["year"] = f"{args.year_from}-"
        elif args.year_to:
            params["year"] = f"-{args.year_to}"

        if args.min_citations > 0:
            params["minCitationCount"] = args.min_citations

        response = requests.get(
            f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
            params=params,
            headers={"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {},
            timeout=15
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        
        if not data:
            return {
                "query": args.query,
                "count": 0,
                "results": [],
                "reranked": False
            }
        
        # 리랭킹을 위한 형식 변환
        papers = []
        for p in data:
            papers.append({
                "text": f"{p.get('title', '')} {p.get('abstract', '')}",
                "paper_id": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "authors": [a.get("name") for a in p.get("authors", [])[:5]],
                "abstract": p.get("abstract"),
                "citation_count": p.get("citationCount", 0),
                "url": p.get("url")
            })

        # print(f"[Semantic Scholar] 리랭킹 전: {len(papers)}개")
        
        # Cross-Encoder 리랭킹
        reranked_papers = rerank_results(args.query, papers, top_k=args.limit)
        
        # print(f"[Semantic Scholar] 리랭킹 후: {len(reranked_papers)}개")
    
        # 유사도 임계값 체크 (0.5 미만이면 관련성 낮음)
        if not reranked_papers or reranked_papers[0].get("relevance_score", 0) < 0.5:
            print(f"[Semantic Scholar] 임계값 미달로 필터링됨")
            return {
                "query": args.query,
                "count": 0,
                "results": [],
                "reranked": True,
                "reason": "신뢰도 낮음"
            }
        
        # text 필드 제거
        for paper in reranked_papers:
            paper.pop('text', None)

        return {
            "query": args.query,
            "count": len(reranked_papers),
            "results": reranked_papers,
            "reranked": True
        }

    except requests.exceptions.RequestException as e:
        return {"error": "API_REQUEST_FAILED", "detail": str(e)}
    
# -------------------------------
# 5. 서브 그래프 
#    서브그래프 툴 설명
# -------------------------------
    
class PaperSearchToolInput(BaseModel):
    query: str = Field(..., description="검색할 논문 키워드나 주제")

def paper_search_placeholder(args: PaperSearchToolInput) -> Dict[str, Any]:
    """실제로 호출 안 됨 - 서브그래프 노드에서 처리"""
    return {}

class PaperAnalysisToolInput(BaseModel):
    query: str = Field(..., description="분석할 논문 제목이나 키워드")

def paper_analysis_placeholder(args: PaperAnalysisToolInput) -> Dict[str, Any]:
    """실제로 호출 안 됨 - 서브그래프 노드에서 처리"""
    return {}

class RecommendationToolInput(BaseModel):
    query: str = Field("AI research", description="관심 분야")
    
def recommendation_placeholder(args: RecommendationToolInput) -> Dict[str, Any]:
    """실제로 호출 안 됨 - 서브그래프 노드에서 처리"""
    return {}