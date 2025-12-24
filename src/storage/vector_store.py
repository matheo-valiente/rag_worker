# handle storage and research of vectors in PostgresSQL

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from config.settings import settings

Base = declarative_base()


class DocumentChunk(Base):
    """Table to stock chunks and their embeddings"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(500))
    chunk_index = Column(Integer)
    embedding = Column(Vector(settings.embedding_dim))

    # Multi-tenancy: isolation par utilisateur et projet
    user_id = Column(String(100), nullable=True, index=True)
    project_id = Column(String(100), nullable=True, index=True)


class VectorStore:

    def __init__(self):
        self.engine = create_engine(settings.database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def add(self, content: str, embedding: list[float], source: str = None,
            chunk_index: int = None, user_id: str = None, project_id: str = None):
        """Add a chunk with its embedding"""
        chunk = DocumentChunk(
            content=content,
            source=source,
            chunk_index=chunk_index,
            embedding=embedding,
            user_id=user_id,
            project_id=project_id
        )
        self.session.add(chunk)
        self.session.commit()

    def add_batch(self, chunks: list[dict], embeddings: list[list[float]],
                  user_id: str = None, project_id: str = None):
        """Add many chunks at once"""
        for chunk, embedding in zip(chunks, embeddings):
            db_chunk = DocumentChunk(
                content=chunk["content"],
                source=chunk.get("source"),
                chunk_index=chunk.get("chunk_index"),
                embedding=embedding,
                user_id=user_id,
                project_id=project_id
            )
            self.session.add(db_chunk)
        self.session.commit()

    def search(self, query_embedding: list[float], top_k: int = None,
               user_id: str = None, project_id: str = None) -> list[dict]:
        """Search most similar chunks with optional user/project filter"""
        if top_k is None:
            top_k = settings.top_k_results

        # Build query with optional filters
        query = self.session.query(
            DocumentChunk.content,
            DocumentChunk.source,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        )

        # Filter by user_id and/or project_id if provided
        if user_id is not None:
            query = query.filter(DocumentChunk.user_id == user_id)
        if project_id is not None:
            query = query.filter(DocumentChunk.project_id == project_id)

        results = query.order_by("distance").limit(top_k).all()

        return [
            {
                "content": r.content,
                "source": r.source,
                "score": 1 - r.distance
            }
            for r in results
        ]

    def clear(self, user_id: str = None, project_id: str = None):
        """Delete chunks (optionally filtered by user/project)"""
        query = self.session.query(DocumentChunk)

        if user_id is not None:
            query = query.filter(DocumentChunk.user_id == user_id)
        if project_id is not None:
            query = query.filter(DocumentChunk.project_id == project_id)

        query.delete()
        self.session.commit()
