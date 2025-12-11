import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 현재 스크립트 파일의 위치를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data"

# 데이터베이스 저장 경로
DB_PATH = BASE_DIR / "chroma_db"

# 임베딩 모델 (한/영 다국어 지원)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def ingest_pdfs():
    print("RAG 데이터베이스 구축 시작 (Indexer)")
    print(f"읽어올 데이터 폴더: {PDF_DIR}")
    
    #  폴더 없을 경우
    if not PDF_DIR.exists():
        print(f"오류: '{PDF_DIR}' 폴더가 없습니다. paper_collector.py를 먼저 실행해서 데이터를 만드세요.")
        return

    # PDF 로드 (PDF Reader 사용)
    print("PDF 파일 로딩 중...")
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    
    if not documents:
        print("로드할 PDF 파일이 없습니다.")
        return
    print(f"총 {len(documents)} 페이지의 문서를 로드했습니다.")

    # 3. 텍스트 분할 (텍스트 분리기)
    print("텍스트 분할(Splitting) 수행 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 한 덩어리의 크기
        chunk_overlap=100,    # 문맥 유지를 위해 겹치는 구간
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"총 {len(splits)} 개의 청크(Chunk)가 생성되었습니다.")

    # 임베딩 및 DB 저장 (Persistent DB, Multi-lingual embedder)
    print(f"임베딩 모델 로드 중 ({MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print(f"ChromaDB에 저장 중 (경로: {DB_PATH})...")
    
    # DB가 없으면 생성하고, 있으면 내용을 추가
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(DB_PATH), 
        collection_name="paper_abstracts"
    )
    
    print("완료! RAG 데이터베이스가 성공적으로 구축되었습니다.")
    print(f"저장된 위치: {DB_PATH}")

if __name__ == "__main__":
    ingest_pdfs()