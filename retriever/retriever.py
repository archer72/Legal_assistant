from typing import List
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.docstore.document import Document
from langchain_core.documents import Document
from retriever.cross_encoder_ranker import CrossEncoderReranker
from utils.logger import get_logger

logger = get_logger('retriever')


class Retriever:
    def __init__(self, index_dir: str, embedding_model: str, reranker_model: str = None):
        self.embedding_model = embedding_model
        self.emb = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_dir = index_dir
        # FAISS.load_local expects the same directory used in save_local
        self.vs = FAISS.load_local(index_dir, self.emb, allow_dangerous_deserialization=True)
        self.reranker = CrossEncoderReranker(reranker_model) if reranker_model else None

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        docs = self.vs.similarity_search(query, k=k)
        if self.reranker:
            return self.reranker.rerank(query, docs)
        return docs
