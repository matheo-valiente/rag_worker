from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
    embedding_model: str = "nomic-embed-text"  # ou mxbai-embed-large
    llm_model: str = "mistral"
    embedding_dim: int = 768  # nomic-embed-text = 768, mxbai-embed-large = 1024

    class Config:
        env_file = ".env"

settings = Settings()

