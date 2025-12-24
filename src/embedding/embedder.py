# generates the vectors (embeddings) via Ollama

import requests
from config.settings import settings


class TextEmbedder:

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        """Genere l'embedding d'un seul texte via Ollama"""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Genere les embeddings de plusieurs textes"""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return embeddings
    

