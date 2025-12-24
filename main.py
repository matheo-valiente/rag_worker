# API FastAPI pour le RAG

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from pydantic import BaseModel
import tempfile
import os

from src.extraction.extractor import TextExtractor
from src.chunking.chunker import TextChunker
from src.embedding.embedder import TextEmbedder
from src.storage.vector_store import VectorStore
from src.retrieval.rag_chain import RAGChain

app = FastAPI(title="RAG API", description="API pour interroger des documents avec RAG")

# Initialisation des composants
extractor = TextExtractor()
chunker = TextChunker()
embedder = TextEmbedder()
vector_store = VectorStore()
rag_chain = RAGChain()


# Schemas Pydantic
class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    user_id: str | None = None
    project_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadRequest(BaseModel):
    user_id: str | None = None
    project_id: str | None = None


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_count: int


class DeleteRequest(BaseModel):
    user_id: str | None = None
    project_id: str | None = None


# Endpoints
@app.get("/health")
def health_check():
    """Verifie que l'API fonctionne"""
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str | None = Query(None, description="ID de l'utilisateur"),
    project_id: str | None = Query(None, description="ID du projet")
):
    """
    Upload et indexe un document (PDF, DOCX, TXT)
    """
    # Verifier l'extension
    allowed_extensions = [".pdf", ".docx", ".txt"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporte. Formats acceptes: {allowed_extensions}"
        )

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Extraire le texte
        text = extractor.extract(tmp_path)

        # 2. Decouper en chunks
        chunks = chunker.chunk_with_metadata(text, source=file.filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="Aucun texte extrait du document")

        # 3. Generer les embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_batch(texts)

        # 4. Stocker dans la base vectorielle
        vector_store.add_batch(chunks, embeddings, user_id=user_id, project_id=project_id)

        return UploadResponse(
            message="Document indexe avec succes",
            filename=file.filename,
            chunks_count=len(chunks)
        )

    finally:
        # Nettoyer le fichier temporaire
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Pose une question et obtient une reponse basee sur les documents indexes
    """
    result = rag_chain.query(
        request.question,
        request.top_k,
        user_id=request.user_id,
        project_id=request.project_id
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )


@app.delete("/documents")
def clear_documents(
    user_id: str | None = Query(None, description="ID de l'utilisateur"),
    project_id: str | None = Query(None, description="ID du projet")
):
    """Supprime les documents indexes (filtre par user/project si fourni)"""
    vector_store.clear(user_id=user_id, project_id=project_id)
    return {"message": "Documents supprimes"}
