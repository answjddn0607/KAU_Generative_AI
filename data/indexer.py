import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # data 폴더 안에 indexer가 있으니까
PDF_DIR = DATA_DIR
CHROMA_DIR = DATA_DIR / "chroma_db"
METADATA_FILE = DATA_DIR / "metadata.json"

def load_metadata():
    """metadata.json 로드"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def index_papers():
    print("=== PDF 인덱싱 시작 ===")
    
    # 1. 메타데이터 로드
    metadata = load_metadata()
    print(f"메타데이터: {len(metadata)}개 논문")
    
    # PDF 파일명 -> 메타데이터 매핑
    filename_to_meta = {}
    for pid, paper in metadata.items():
        filename_to_meta[paper['pdf_filename']] = paper
    
    # 2. PDF 로드
    print(f"PDF 로드 중... ({PDF_DIR})")
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    print(f"로드된 문서: {len(documents)}개")
    
    # 3. 문서에 메타데이터 추가
    for doc in documents:
        filename = Path(doc.metadata.get('source', '')).name
        if filename in filename_to_meta:
            paper = filename_to_meta[filename]
            doc.metadata['title'] = paper.get('title', 'Unknown')
            doc.metadata['year'] = paper.get('year', 'Unknown')
            doc.metadata['citationCount'] = paper.get('citationCount', 0)
            doc.metadata['authors'] = paper.get('authors', '')
            doc.metadata['paperId'] = paper.get('paperId', '')
    
    # 4. 텍스트 분할
    print("텍스트 분할 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"생성된 청크: {len(chunks)}개")
    
    # 5. 임베딩 모델 로드
    print("임베딩 모델 로드 중...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 6. ChromaDB 초기화
    print(f"ChromaDB 초기화... ({CHROMA_DIR})")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # 기존 컬렉션 삭제 후 새로 생성
    try:
        client.delete_collection("papers")
    except:
        pass
    
    collection = client.create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    # 7. 임베딩 및 저장
    print("임베딩 및 저장 중...")
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        texts = [chunk.page_content for chunk in batch]
        embeddings = embedder.embed_documents(texts)
        
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        metadatas = []
        for chunk in batch:
            metadatas.append({
                'title': str(chunk.metadata.get('title', 'Unknown')),
                'year': str(chunk.metadata.get('year', 'Unknown')),
                'citationCount': int(chunk.metadata.get('citationCount', 0)),
                'authors': str(chunk.metadata.get('authors', '')),
                'paperId': str(chunk.metadata.get('paperId', '')),
                'source': str(chunk.metadata.get('source', ''))
            })
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"진행: {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    print(f"\n=== 인덱싱 완료 ===")
    print(f"총 {collection.count()}개 청크 저장됨")
    print(f"DB 경로: {CHROMA_DIR}")


if __name__ == "__main__":
    index_papers()