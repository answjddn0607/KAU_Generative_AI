import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 싱글톤 패턴으로 클라이언트 관리
_client = None

def get_chroma_client() -> chromadb.PersistentClient:
    """
    ChromaDB PersistentClient 반환 (싱글톤)
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
    return _client


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """
    Multilingual 임베딩 함수 반환 (싱글톤)
    """
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedding_fn


def get_memory_collection():
    """
    메모리 저장용 Collection
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="agent_memory",
        metadata={"description": "Agent memory storage"},
        embedding_function=get_embedding_function()
    )


def get_rag_collection():
    """
    RAG용 Collection (논문 초록 등)
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="paper_abstracts",
        metadata={"description": "Paper abstracts for RAG"},
        embedding_function=get_embedding_function()
    )