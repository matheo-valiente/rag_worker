# Reranker cross-encoder pour améliorer la pertinence des résultats RAG

from sentence_transformers import CrossEncoder
from config.settings import settings


class Reranker:
    """
    Reranker utilisant un cross-encoder pour réordonner les résultats.

    Le cross-encoder analyse la paire (query, document) ensemble,
    ce qui donne une meilleure estimation de pertinence que la
    similarité cosinus des embeddings.
    """

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: Modèle cross-encoder à utiliser.
                        Par défaut: cross-encoder/ms-marco-MiniLM-L-6-v2 (rapide et efficace)
        """
        self.model_name = model_name or getattr(
            settings, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self._model = None  # Lazy loading

    @property
    def model(self) -> CrossEncoder:
        """Charge le modèle de manière paresseuse"""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = None,
        score_threshold: float = None
    ) -> list[dict]:
        """
        Réordonne les documents par pertinence avec le query.

        Args:
            query: La question de l'utilisateur
            documents: Liste de dicts avec au moins 'content'
            top_k: Nombre de documents à retourner (None = tous)
            score_threshold: Score minimum pour garder un document

        Returns:
            Documents réordonnés avec score de reranking ajouté
        """
        if not documents:
            return []

        # Préparer les paires (query, document)
        pairs = [(query, doc['content']) for doc in documents]

        # Calculer les scores de pertinence
        scores = self.model.predict(pairs)

        # Ajouter les scores aux documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # Trier par score de reranking (décroissant)
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        # Filtrer par seuil si spécifié
        if score_threshold is not None:
            reranked = [d for d in reranked if d['rerank_score'] >= score_threshold]

        # Limiter le nombre de résultats
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def rerank_with_fusion(
        self,
        query: str,
        documents: list[dict],
        embedding_weight: float = 0.3,
        rerank_weight: float = 0.7,
        top_k: int = None
    ) -> list[dict]:
        """
        Combine les scores d'embedding et de reranking pour un meilleur classement.

        Args:
            query: La question
            documents: Documents avec 'content' et 'score' (similarité embedding)
            embedding_weight: Poids du score d'embedding (0-1)
            rerank_weight: Poids du score de reranking (0-1)
            top_k: Nombre de résultats à retourner

        Returns:
            Documents avec score fusionné
        """
        if not documents:
            return []

        # Calculer les scores de reranking
        pairs = [(query, doc['content']) for doc in documents]
        rerank_scores = self.model.predict(pairs)

        # Normaliser les scores de reranking entre 0 et 1
        min_score = min(rerank_scores)
        max_score = max(rerank_scores)
        score_range = max_score - min_score if max_score != min_score else 1

        for doc, rerank_score in zip(documents, rerank_scores):
            normalized_rerank = (rerank_score - min_score) / score_range
            embedding_score = doc.get('score', 0.5)

            # Score fusionné
            doc['rerank_score'] = float(rerank_score)
            doc['fused_score'] = (
                embedding_weight * embedding_score +
                rerank_weight * normalized_rerank
            )

        # Trier par score fusionné
        fused = sorted(documents, key=lambda x: x['fused_score'], reverse=True)

        if top_k is not None:
            fused = fused[:top_k]

        return fused
