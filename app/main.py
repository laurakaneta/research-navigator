from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import Analyzer, QASystem
from .pubmed import PubMedClient

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="AI Research Navigator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_results: int = Field(default=12, ge=3, le=30)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    context: Dict[str, Any]


@lru_cache(maxsize=1)
def get_pubmed_client() -> PubMedClient:
    return PubMedClient()


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    return Analyzer()


@lru_cache(maxsize=1)
def get_qa() -> QASystem:
    return QASystem()


@app.get("/")
def root() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_file)


@app.post("/api/search")
def search_literature(payload: SearchRequest) -> Dict[str, Any]:
    client = get_pubmed_client()
    analyzer = get_analyzer()

    pmids = client.search_pmids(payload.query, payload.max_results)
    papers = client.fetch_details(pmids)
    if not papers:
        return {"query": payload.query, "overview": "No papers found.", "themes": [], "papers": [], "graph": {"nodes": [], "edges": []}}

    results = analyzer.run(payload.query, papers)
    return results


@app.post("/api/ask")
def ask_question(payload: AskRequest) -> Dict[str, Any]:
    qa = get_qa()
    papers = payload.context.get("papers", [])
    return qa.answer(payload.question, papers)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": "AI Research Navigator"}
