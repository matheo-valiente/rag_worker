from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "mistral"
    embedding_dim: int = 768

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_mode: str = "semantic"

    # Retrieval
    top_k_results: int = 5
    top_k_rerank: int = 3

    # Reranker
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Prompt
    language: str = "fr"

    class Config:
        env_file = ".env"


settings = Settings()
