from typing import List
#from cross_encoder import CrossEncoder
from sentence_transformers import CrossEncoder
#from langchain.docstore.document import Document
from langchain_core.documents import Document


class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        if model_name:
            self.model = CrossEncoder(model_name)
        else:
            self.model = None

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.model:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]
