from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import uuid
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)


def main():
    BASE_DIR = Path(__file__).resolve().parent
    PDF_DIR = BASE_DIR / "data"
    DB_PATH = BASE_DIR / "chroma_db"
    
    # PDF 로드
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    print(f"로드된 문서: {len(documents)}개")
    
    # 1500자 자르기
    for doc in documents:
        doc.page_content = doc.page_content[:1500]
    
    # 청킹
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"생성된 청크: {len(chunks)}개")
    
    # ChromaDB 네이티브 방식
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("papers")
    except:
        pass
    
    collection = client.get_or_create_collection(
        name="papers",
        embedding_function=embedding_fn
    )
    
    # 배치로 저장 (100개씩)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = [str(uuid.uuid4()) for _ in batch]
        docs = [c.page_content for c in batch]
        metas = [{
            "title": c.metadata.get("source", "").split("/")[-1].replace(".pdf", ""),
            "source": c.metadata.get("source", ""),
            "page": c.metadata.get("page", 0)
        } for c in batch]
        
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"진행: {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    print(f"인덱싱 완료: {collection.count()}개")


if __name__ == "__main__":
    main()