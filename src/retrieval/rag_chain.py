# RAG Chain - pipeline complet: query -> retrieval -> generation

import requests
from config.settings import settings
from src.embedding.embedder import TextEmbedder
from src.storage.vector_store import VectorStore


class RAGChain:

    def __init__(self):
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore()
        self.base_url = settings.ollama_base_url
        self.model = settings.llm_model

    def retrieve(self, query: str, top_k: int = None,
                 user_id: str = None, project_id: str = None) -> list[dict]:
        """Recherche les chunks les plus pertinents pour une question"""
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(
            query_embedding,
            top_k,
            user_id=user_id,
            project_id=project_id
        )
        return results

    def generate(self, query: str, context: list[dict]) -> str:
        """Genere une reponse basee sur le contexte recupere"""
        # Construire le contexte a partir des chunks
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk.get('source', 'Unknown')}\n{chunk['content']}"
            for chunk in context
        ])

        # Prompt pour le RAG
        prompt = f"""Tu es un assistant qui repond aux questions en te basant uniquement sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.
Reponds de maniere concise et precise.

Contexte:
{context_text}

Question: {query}

Reponds en te basant sur le contexte ci-dessus."""

        # Appel au LLM via Ollama
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()

        return response.json()["response"]

    def query(self, question: str, top_k: int = None,
              user_id: str = None, project_id: str = None) -> dict:
        """Pipeline RAG complet: recuperation + generation"""
        # 1. Recuperer les chunks pertinents
        context = self.retrieve(
            question,
            top_k,
            user_id=user_id,
            project_id=project_id
        )

        if not context:
            return {
                "answer": "Aucun document pertinent trouve pour repondre a cette question.",
                "sources": [],
                "context": []
            }

        # 2. Generer la reponse
        answer = self.generate(question, context)

        # 3. Extraire les sources uniques
        sources = list(set(chunk.get("source") for chunk in context if chunk.get("source")))

        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }
