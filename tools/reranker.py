from sentence_transformers import CrossEncoder

# 싱글톤 패턴으로 Cross-Encoder 관리
_reranker = None

def get_reranker() -> CrossEncoder:
    """
    Cross-Encoder 모델 반환 (싱글톤)
    다국어 지원 모델 사용
    """
    global _reranker
    if _reranker is None:
        print("[Reranker] Cross-Encoder 모델 로딩 중...")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        print("[Reranker] 모델 로딩 완료")
    return _reranker


def rerank_results(query: str, documents: list, top_k: int = None) -> list:
    """
    Cross-Encoder로 문서 리랭킹
    
    Args:
        query: 검색 질의
        documents: 리랭킹할 문서 리스트 (각 문서는 dict, 'text' 키 필요)
        top_k: 반환할 상위 k개 (None이면 전체 반환)
    
    Returns:
        리랭킹된 문서 리스트 (relevance_score 추가됨)
    """
    if not documents:
        return []
    
    print(f"[Reranker] 리랭킹 시작 - 입력 {len(documents)}개 문서")
    
    reranker = get_reranker()
    
    # query-document 쌍 생성
    pairs = [[query, doc['text']] for doc in documents]
    
    # Cross-Encoder로 점수 계산
    scores = reranker.predict(pairs)
    
    # 점수를 문서에 추가하고 정렬
    for doc, score in zip(documents, scores):
        doc['relevance_score'] = float(score)
    
    # 점수 기준 내림차순 정렬
    reranked = sorted(documents, key=lambda x: x['relevance_score'], reverse=True)
    
    # top_k 제한
    if top_k is not None:
        reranked = reranked[:top_k]
    
    print(f"[Reranker] 리랭킹 완료 - 최종 {len(reranked)}개 반환")
    
    return reranked