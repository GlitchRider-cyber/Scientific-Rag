import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from config import PAPERS_DIR
from ingestion.pipeline import ingest_file
from embedding.dual_store import DualStore
from retrieval.retriever import retrieve
from retrieval.generator import generate_answer
from utils.helpers import allowed_file

router = APIRouter()

# Shared store instance
store = DualStore()


class QueryRequest(BaseModel):
    question: str


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a paper (PDF, image, or text) and ingest it."""
    if not allowed_file(file.filename):
        raise HTTPException(400, f"Unsupported file type: {file.filename}")

    # Save file
    file_path = os.path.join(PAPERS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest: extract text -> chunk -> embed -> store
    chunks = ingest_file(file_path)
    store.add_chunks(chunks)

    return {
        "message": f"Ingested {file.filename}",
        "chunks_created": len(chunks),
        "total_chunks": store.total_chunks(),
    }


@router.post("/query")
async def query(req: QueryRequest):
    """Ask a question across all ingested papers."""
    if store.total_chunks() == 0:
        raise HTTPException(400, "No papers ingested yet. Upload papers first.")

    # Retrieve top chunks
    top_chunks = retrieve(req.question, store)

    # Generate answer with citations
    result = generate_answer(req.question, top_chunks)

    return result


@router.get("/papers")
async def list_papers():
    """List all uploaded papers."""
    files = os.listdir(PAPERS_DIR) if os.path.exists(PAPERS_DIR) else []
    return {"papers": files, "total_chunks": store.total_chunks()}


@router.delete("/clear")
async def clear_data():
    """Clear all ingested data and indices."""
    store.clear()

    # Remove uploaded files
    for f in os.listdir(PAPERS_DIR):
        os.remove(os.path.join(PAPERS_DIR, f))

    return {"message": "All data cleared"}
