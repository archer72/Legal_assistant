import argparse
import os
from pathlib import Path

#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.docstore.document import Document
#from langchain.schema import Document
from langchain_core.documents import Document

#from ingestion.parse_documents import load_documents, chunk_documents
from ingestion.parse_documents import load_documents, chunk_documents
from utils.logger import get_logger

logger = get_logger('ingest')

def build_index(data_dir: str, index_dir: str, embedding_model: str, chunk_size: int, chunk_overlap: int):
    logger.info("Loading documents...")
    docs = load_documents(data_dir)
    logger.info(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(f"Encoding {len(chunks)} chunks with {embedding_model} ...")
    hf = HuggingFaceEmbeddings(model_name=embedding_model)
    documents = [Document(page_content=c['text'], metadata=c['metadata']) for c in chunks]

    logger.info('Building FAISS index...')
    vectorstore = FAISS.from_documents(documents, hf)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    logger.info(f'Index saved to {index_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--embedding-model', default=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    parser.add_argument('--chunk-size', type=int, default=int(os.getenv('CHUNK_SIZE', 800)))
    parser.add_argument('--chunk-overlap', type=int, default=int(os.getenv('CHUNK_OVERLAP', 100)))
    args = parser.parse_args()

    build_index(args.data_dir, args.index_dir, args.embedding_model, args.chunk_size, args.chunk_overlap)

