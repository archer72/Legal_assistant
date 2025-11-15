from pathlib import Path
from typing import List
from tqdm import tqdm
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def read_text_file(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')

def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("PyPDF2 is required to read PDFs. Install it or use text files.")
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def load_documents(data_dir: str) -> List[dict]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    files = list(p.rglob('*'))
    docs = []
    for f in tqdm(files, desc='Scanning files'):
        if f.is_dir():
            continue
        if f.suffix.lower() in ['.txt', '.md']:
            text = read_text_file(f)
        elif f.suffix.lower() == '.pdf':
            text = read_pdf(f)
        elif f.suffix.lower() in ['.html', '.htm']:
            text = read_text_file(f)
        else:
            continue
        docs.append({
            'source': str(f),
            'text': text,
        })
    return docs

def chunk_documents(docs: List[dict], chunk_size: int = 800, chunk_overlap: int = 100) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for d in tqdm(docs, desc='Chunking'):
        chunks = splitter.split_text(d['text'])
        for i, c in enumerate(chunks):
            out.append({
                'text': c,
                'metadata': {
                    'source': d['source'],
                    'chunk': i,
                }
            })
    return out

