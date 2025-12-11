from typing import Type, Callable, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import os
import requests
import uuid
from datetime import datetime

from .chroma_client import get_memory_collection, get_rag_collection

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



# -------------------------------
#  ToolSpec 정의
# -------------------------------
class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: Type[BaseModel]                  # Pydantic 모델 타입
    handler: Callable[[BaseModel], Dict[str, Any]]


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
    top_k : int = Field(3, ge = 1, le = 10, description = "빈환 결과 수(1~10, 기본 값 3")

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

DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def rag_search_handler(args: RAGSearchInput) -> Dict[str, Any]:
    # LangChain 방식으로 DB 로드
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="paper_abstracts" # 작성자님이 설정한 이름
    )
    
    # 검색 수행
    results = vector_store.similarity_search(args.query, k=args.top_k)
    
    # 결과 포맷팅
    papers = []
    for doc in results:
        papers.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0)
        })
    
    return {
        "query": args.query,
        "results": papers,
        "count": len(papers)
    }