from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import threading

from utils.logger import get_logger
from retriever.retriever import Retriever
from llm.generator import Generator
from cross_reference import build_cross_reference_chain
from summarizer import build_judgment_summarizer_chain

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

logger = get_logger('api')
app = FastAPI(title='Indian Law Query Assistant')


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 8


class QueryResponse(BaseModel):
    answer: str
    cross_reference: Optional[str] = None
    summary: Optional[str] = None
    sources: List[dict]
    conversation_id: str


class ResetResponse(BaseModel):
    conversation_id: str
    message: str


# Conversation store (in-memory). Replace with persistent store for production.
_conversations_lock = threading.Lock()
_conversations: Dict[str, Dict[str, Any]] = {}

# Config from env
INDEX_DIR = os.getenv('INDEX_DIR', 'faiss_index')
EMBED_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
RERANKER_MODEL = os.getenv('RERANKER_MODEL', None)
HF_LLM_PATH = os.getenv('HF_LLM_PATH')

# Initialize components
logger.info("Initializing retriever and LLM (may take time)...")
retriever = Retriever(INDEX_DIR, EMBED_MODEL, reranker_model=RERANKER_MODEL)

# Load generator LLM
gen = Generator(model_path=HF_LLM_PATH)
llm = gen.llm

# Chains
cross_ref_chain = build_cross_reference_chain(llm)
judgment_summarizer = build_judgment_summarizer_chain(llm)


def _create_conversational_chain(conversation_id: str):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever.vs.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain, memory


@app.post('/query', response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        conv_id = req.conversation_id or str(uuid.uuid4())

        with _conversations_lock:
            if conv_id not in _conversations:
                chain, memory = _create_conversational_chain(conv_id)
                _conversations[conv_id] = {'chain': chain, 'memory': memory}
            else:
                chain = _conversations[conv_id]['chain']

        # Base RAG answer
        result = chain({'question': req.query})
        answer = result.get('answer') or result.get('output_text') or ''
        src_docs = result.get('source_documents', [])

        # Prepare docs for cross-referencing and summarization
        docs_text = "\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in src_docs])

        # Cross-reference
        cross_ref_answer = cross_ref_chain.run({"query": req.query, "docs": docs_text})

        # Summarize if requested
        summary = None
        if any(k in req.query.lower() for k in ["summarize", "summary", "judgment", "supreme court", "sc judgment"]):
            summary = judgment_summarizer.run({"judgment_text": docs_text})

        sources = []
        for d in src_docs:
            md = getattr(d, 'metadata', {})
            sources.append({'source': md.get('source'), 'chunk': md.get('chunk'), 'text_snippet': d.page_content[:800]})

        return QueryResponse(answer=answer, cross_reference=cross_ref_answer, summary=summary, sources=sources, conversation_id=conv_id)

    except Exception as e:
        logger.exception('Query failed')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/reset', response_model=ResetResponse)
def reset_conversation():
    conv_id = str(uuid.uuid4())
    with _conversations_lock:
        chain, memory = _create_conversational_chain(conv_id)
        _conversations[conv_id] = {'chain': chain, 'memory': memory}
    return ResetResponse(conversation_id=conv_id, message='New conversation created and memory reset.')


@app.get('/conversations')
def list_conversations():
    return {'conversations': list(_conversations.keys())}
