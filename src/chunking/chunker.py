# chunks the texte for RAG - Chunking sémantique adaptatif

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
import re


class TextChunker:
    """Chunker avec support pour chunking classique et sémantique"""

    def __init__(self, mode: str = "semantic"):
        """
        Args:
            mode: "classic" pour RecursiveCharacterTextSplitter,
                  "semantic" pour chunking par phrases/paragraphes
        """
        self.mode = mode
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # Splitter classique comme fallback
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Découpe le texte en phrases de manière intelligente"""
        # Pattern pour détecter les fins de phrases
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-ÿ])')
        sentences = sentence_endings.split(text)

        # Nettoyer les phrases vides
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Découpe le texte en paragraphes"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _semantic_chunk(self, text: str) -> list[str]:
        """
        Chunking sémantique: regroupe les phrases en chunks cohérents
        tout en respectant les limites de taille
        """
        # D'abord découper en paragraphes
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # Si le paragraphe est trop long, le découper en phrases
            if len(paragraph) > self.chunk_size:
                sentences = self._split_into_sentences(paragraph)

                for sentence in sentences:
                    # Si ajouter cette phrase dépasse la limite
                    if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                # Ajouter le paragraphe entier si possible
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = f"{current_chunk}\n\n{paragraph}".strip()

        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Appliquer l'overlap entre les chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Applique un overlap entre les chunks pour préserver le contexte"""
        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Prendre les derniers mots du chunk précédent
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Trouver un point de coupure propre (début de mot)
            space_idx = overlap_text.find(' ')
            if space_idx != -1:
                overlap_text = overlap_text[space_idx + 1:]

            # Ajouter l'overlap au début du chunk actuel
            overlapped_chunk = f"{overlap_text} {current_chunk}".strip()
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def chunk(self, text: str) -> list[str]:
        """Découpe le texte en chunks selon le mode configuré"""
        if self.mode == "semantic":
            return self._semantic_chunk(text)
        return self.splitter.split_text(text)

    def chunk_with_metadata(self, text: str, source: str) -> list[dict]:
        """Découpe le texte avec métadonnées enrichies"""
        chunks = self.chunk(text)

        return [
            {
                "content": chunk,
                "source": source,
                "chunk_index": i,
                "chunk_mode": self.mode,
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
