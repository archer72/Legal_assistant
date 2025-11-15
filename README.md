# Indian Law Query Assistant

AI-powered RAG system for answering legal queries grounded in Indian statutes, IPC, RTI laws, Supreme Court judgments, and legal articles.

Features
- Document ingestion and chunking
- FAISS vector store
- Embeddings with sentence-transformers (local)
- Cross-encoder re-ranker for higher precision
- Generator using Hugging Face local LLM via HuggingFacePipeline
- Conversational retrieval (multi-turn) with memory
- Cross-referencing across multiple Acts/judgments
- Supreme Court judgment summarizer
- FastAPI backend + Streamlit demo UI

See NOTES.md for further guidance.
