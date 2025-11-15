from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import threading

# Internal modules
from utils.logger import get_logger
from retriever.retriever import Retriever
from llm.generator import Generator
from cross_reference import build_cross_reference_chain
from summarizer import build_judgment_summarizer_chain

# New LangChain Imports (2024–2025)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

logger = get_logger('api')
app = FastAPI(title='Indian Law Query Assistant')


# ==============================
# REQUEST / RESPONSE MODELS
# ==============================
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


# ===========================================
# In-memory conversation storage
# ===========================================
_conversations_lock = threading.Lock()
_conversations: Dict[str, Dict[str, Any]] = {}


# ===========================================
# ENV CONFIG
# ===========================================
INDEX_DIR = os.getenv('INDEX_DIR', 'faiss_index')
EMBED_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
RERANKER_MODEL = os.getenv('RERANKER_MODEL', None)
HF_LLM_PATH = os.getenv('HF_LLM_PATH')


# ===========================================
# INITIALIZATION
# ===========================================
logger.info("Initializing retriever + LLM…")

retriever = Retriever(INDEX_DIR, EMBED_MODEL, reranker_model=RERANKER_MODEL)

gen = Generator(model_path=HF_LLM_PATH)
llm = gen.llm

cross_ref_chain = build_cross_reference_chain(llm)
judgment_summarizer = build_judgment_summarizer_chain(llm)


# ===========================================
# Replace ConversationalRetrievalChain
# with a modern Runnable-based pipeline
# ===========================================
conversation_prompt = PromptTemplate.from_template(
    """
You are an Indian legal assistant.
Use the retrieved documents and the chat history to answer the user's query.

Chat History:
{history}

Retrieved Context:
{context}

User Question:
{query}

Provide a well-reasoned answer with citations like [source:path#chunk].
"""
)


def format_history(history):
    if not history:
        return ""
    out = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            out.append(f"User: {msg.content}")
        else:
            out.append(f"Assistant: {msg.content}")
    return "\n".join(out)


def build_conversational_rag_chain(retriever, llm):

    def prepare_inputs(inputs):
        query = inputs["query"]
        hist = inputs.get("history", [])

        docs = retriever.vs.similarity_search(query, k=inputs.get("top_k", 8))

        ctx = "\n\n".join(
            f"Source: {d.metadata.get('source')}\n{d.page_content}"
            for d in docs
        )

        return {
            "query": query,
            "context": ctx,
            "history": format_history(hist),
            "raw_docs": docs
        }

    return RunnableSequence(
        steps=[
            prepare_inputs,          # prepare context
            conversation_prompt,     # fill prompt
            llm,                     # generate
            StrOutputParser()        # extract string
        ]
    )


# Create chain instance
def _create_conversation_instance():
    return {
        "history": [],
        "chain": build_conversational_rag_chain(retriever, llm)
    }


# ===========================================
# QUERY ENDPOINT
# ===========================================
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        # Create conversation if none
        conv_id = req.conversation_id or str(uuid.uuid4())

        with _conversations_lock:
            if conv_id not in _conversations:
                _conversations[conv_id] = _create_conversation_instance()

            conv = _conversations[conv_id]
            history = conv["history"]
            chain = conv["chain"]

        # Run conversational RAG
        result = chain.invoke({
            "query": req.query,
            "history": history,
            "top_k": req.top_k
        })

        # Retrieve raw docs (from prepare_inputs)
        prep = chain.steps[0]({
            "query": req.query,
            "history": history,
            "top_k": req.top_k
        })
        src_docs = prep["raw_docs"]

        # Update conversation history
        history.append(HumanMessage(content=req.query))
        history.append(AIMessage(content=result))

        # Prepare docs text
        docs_text = prep["context"]

        # Cross reference
        cross_ref_answer = cross_ref_chain.invoke({
            "query": req.query,
            "docs": docs_text
        })

        # Summarization trigger
        summary = None
        if any(w in req.query.lower() for w in ["summarize", "summary", "judgment", "supreme court"]):
            summary = judgment_summarizer.invoke({"judgment_text": docs_text})

        # Format sources
        sources = [
            {
                "source": d.metadata.get("source"),
                "chunk": d.metadata.get("chunk"),
                "text_snippet": d.page_content[:800]
            }
            for d in src_docs
        ]

        return QueryResponse(
            answer=result,
            cross_reference=cross_ref_answer,
            summary=summary,
            sources=sources,
            conversation_id=conv_id
        )

    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# RESET CONVERSATION
# ===========================================
@app.post("/reset", response_model=ResetResponse)
def reset_conversation():
    conv_id = str(uuid.uuid4())
    with _conversations_lock:
        _conversations[conv_id] = _create_conversation_instance()

    return ResetResponse(
        conversation_id=conv_id,
        message="New conversation created."
    )


@app.get("/conversations")
def list_conversations():
    return {"conversations": list(_conversations.keys())}
