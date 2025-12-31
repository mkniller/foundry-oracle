import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
SRD_DIR = INPUT_DIR / "SRDs"
TRANSCRIPTS_DIR = INPUT_DIR / "transcripts"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = Path(__file__).resolve().parent / "data"
VECTOR_DIR = DATA_DIR / "chroma"
STATE_PATH = DATA_DIR / "ingest_state.json"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION", "srd")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SEC", "5"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-3b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")

app = FastAPI()
logger = logging.getLogger("foundry-oracle")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_DIR / "service.log", maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_state() -> Dict[str, Dict[str, float]]:
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_state(state: Dict[str, Dict[str, float]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def chunk_text(text: str) -> List[str]:
    chunks = []
    if CHUNK_SIZE <= 0:
        return chunks
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    for start in range(0, len(text), step):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def iter_docs() -> List[Path]:
    docs: List[Path] = []
    for folder in (SRD_DIR, TRANSCRIPTS_DIR):
        if folder.exists():
            docs.extend(
                [
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower() in {".pdf", ".txt"}
                ]
            )
    return sorted(docs)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def answer(self, question: str, context: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using the provided SRD context. Be concise and cite sources.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("LLM request failed: %s", exc)
            return None


class DocumentIndexer:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.state = load_state()
        self.lock = threading.Lock()

    def needs_update(self, path: Path) -> bool:
        stat = path.stat()
        entry = self.state.get(str(path))
        if not entry:
            return True
        return entry.get("mtime") != stat.st_mtime or entry.get("size") != stat.st_size

    def update_state(self, path: Path) -> None:
        stat = path.stat()
        self.state[str(path)] = {"mtime": stat.st_mtime, "size": stat.st_size}

    def ingest_pdf(self, path: Path, source_type: str) -> None:
        logger.info("Ingesting %s", path.name)
        reader = PdfReader(str(path))
        page_labels = list(getattr(reader, "page_labels", []) or [])
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []
        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_label = ""
            if page_index < len(page_labels):
                page_label = str(page_labels[page_index])
            for chunk_index, chunk in enumerate(chunk_text(text)):
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": path.name,
                        "source_type": source_type,
                        "page_index": str(page_index + 1),
                        "page_label": page_label,
                    }
                )
                ids.append(f"{path.name}:{page_index}:{chunk_index}")

        if not documents:
            logger.info("No text extracted from %s", path.name)
            return

        embeddings = self.embedder.encode(documents, show_progress_bar=False)
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        self.update_state(path)
        save_state(self.state)
        logger.info("Ingested %s chunks from %s", len(documents), path.name)

    def ingest_text(self, path: Path, source_type: str) -> None:
        logger.info("Ingesting %s", path.name)
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.info("No text extracted from %s", path.name)
            return

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []
        for chunk_index, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": path.name,
                    "source_type": source_type,
                    "page_index": "",
                    "page_label": "",
                }
            )
            ids.append(f"{path.name}:0:{chunk_index}")

        embeddings = self.embedder.encode(documents, show_progress_bar=False)
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        self.update_state(path)
        save_state(self.state)
        logger.info("Ingested %s chunks from %s", len(documents), path.name)

    def scan_once(self) -> None:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        SRD_DIR.mkdir(parents=True, exist_ok=True)
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        for doc in iter_docs():
            if self.needs_update(doc):
                with self.lock:
                    source_type = "srd" if SRD_DIR in doc.parents else "transcript"
                    if doc.suffix.lower() == ".pdf":
                        self.ingest_pdf(doc, source_type)
                    elif doc.suffix.lower() == ".txt":
                        self.ingest_text(doc, source_type)

    def query(
        self, question: str, top_k: int = 4, source_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        embedding = self.embedder.encode([question], show_progress_bar=False)
        where = None
        if source_type in {"srd", "transcript"}:
            where = {"source_type": source_type}
        results = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        sources = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            sources.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "source_type": meta.get("source_type", ""),
                    "page_index": meta.get("page_index", ""),
                    "page_label": meta.get("page_label", ""),
                    "distance": f"{dist:.4f}",
                }
            )
        return {"sources": sources}


def monitor_folder(indexer: DocumentIndexer, stop_event: threading.Event) -> None:
    logger.info("Monitoring input folder for PDFs")
    while not stop_event.is_set():
        try:
            indexer.scan_once()
        except Exception as exc:
            logger.exception("Ingestion error: %s", exc)
        stop_event.wait(POLL_INTERVAL)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


indexer: Optional[DocumentIndexer] = None
llm_client = LLMClient(LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)
stop_event = threading.Event()


@app.on_event("startup")
def startup() -> None:
    global indexer
    configure_logging()
    logger.info("Starting Foundry Oracle service")
    indexer = DocumentIndexer()
    indexer.scan_once()
    threading.Thread(target=monitor_folder, args=(indexer, stop_event), daemon=True).start()


@app.on_event("shutdown")
def shutdown() -> None:
    stop_event.set()
    logger.info("Stopping Foundry Oracle service")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not indexer:
        return AskResponse(answer="Indexer not ready", sources=[])

    def classify_question(text: str) -> str:
        text_lower = text.lower()
        transcript_patterns = [
            "last session",
            "previous session",
            "last time",
            "previous time",
            "recap",
            "session recap",
            "what happened",
            "what did we do",
            "where did we leave off",
            "catch me up",
            "remind me",
        ]
        if any(pattern in text_lower for pattern in transcript_patterns):
            return "transcript"
        return "srd"

    preferred = classify_question(req.question)

    if preferred == "transcript":
        results = indexer.query(req.question, source_type="transcript")
        if not results["sources"]:
            results = indexer.query(req.question, source_type="srd")
    else:
        results = indexer.query(req.question, source_type="srd")
        if not results["sources"]:
            results = indexer.query(req.question, source_type="transcript")
    sources = results["sources"]
    context = "\n\n".join(
        f"Source: {s['source']} p.{s['page_label'] or s['page_index']}\n{s['text']}"
        for s in sources
    )
    answer = llm_client.answer(req.question, context)
    if not answer:
        answer = "I could not reach the local LLM. Here are the most relevant excerpts:\n" + context
    source_lines = []
    for source in sources:
        page = source.get("page_label") or source.get("page_index") or "unknown"
        title = source.get("source") or "unknown"
        source_lines.append(f"- {title} p.{page}")
    formatted = f"{answer}\n\nSources:\n" + "\n".join(source_lines) if source_lines else answer
    return AskResponse(answer=formatted, sources=sources)


if __name__ == "__main__":
    # NSSM setup (run from repo root):
    # 1) nssm install FoundryOracle "C:\\Path\\To\\python.exe" "-m" "uvicorn" "service.main:app" "--host" "127.0.0.1" "--port" "8000"
    # 2) nssm set FoundryOracle AppDirectory "C:\\Users\\Matt\\GitHub\\foundry-oracle"
    # 3) nssm start FoundryOracle
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
