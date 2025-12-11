import os
from pathlib import Path  # 경로 계산을 위해 추가
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_rag(query_text):
    # 1. 경로 및 설정
    
    # 현재 스크립트(test_rag.py)의 위치를 구함 (예: .../kau_generative_ai/AI)
    current_dir = Path(__file__).resolve().parent
    
    # 상위 폴더(../)로 나간 뒤 'Data' 폴더 안의 'chroma_db'를 지정
    # 실제 경로: .../kau_generative_ai/Data/chroma_db
    db_path_obj = current_dir.parent / "Data" / "chroma_db"
    DB_PATH = str(db_path_obj) # 라이브러리 호환성을 위해 문자열로 변환

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # ★ 중요: paper_indexer.py에서 설정한 컬렉션 이름과 일치해야 함
    COLLECTION_NAME = "paper_abstracts"  

    print(f"🔎 검색 테스트 시작: '{query_text}'")
    print(f"📂 DB 경로 확인: {DB_PATH}") # 경로가 맞게 잡혔는지 출력 확인
    
    # DB 폴더 존재 여부 확인
    if not os.path.exists(DB_PATH):
        print(f"❌ 오류: '{DB_PATH}' 경로를 찾을 수 없습니다.")
        print("   1. 'paper_indexer.py'를 먼저 실행해서 DB를 만들었는지 확인하세요.")
        print("   2. 'Data' 폴더 안에 'chroma_db' 폴더가 있는지 확인하세요.")
        return

    # 2. DB 로드 (읽기 전용)
    print("📂 데이터베이스 로딩 중...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    try:
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # 3. 실제 검색 수행 (상위 3개)
        results = vector_store.similarity_search(query_text, k=3)

        # 4. 결과 출력
        if not results:
            print("⚠️ 검색 결과가 0건입니다.")
            print("   - DB에 데이터가 없거나, Collection 이름이 다를 수 있습니다.")
        else:
            print(f"\n✅ 검색 결과 ({len(results)}건):")
            for i, doc in enumerate(results):
                print("-" * 50)
                source = doc.metadata.get('source', 'Unknown')
                print(f"[{i+1}] 출처: {source}")
                # 가독성을 위해 줄바꿈 문자를 공백으로 변경하여 출력
                content_preview = doc.page_content[:200].replace('\n', ' ')
                print(f"내용: {content_preview}...") 
                print("-" * 50)
                
    except Exception as e:
        print(f"\n❌ 검색 중 에러 발생: {e}")
        print("   DB 파일이 손상되었거나 호환되지 않는 버전일 수 있습니다.")

if __name__ == "__main__":
    # 원하는 검색어로 테스트
    test_rag("금융 관련 논문 찾아줘")