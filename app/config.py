"""Central settings, loaded from environment variables / .env. No secrets in code."""
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI. The key comes from OPENAI_API_KEY (or LLM_API_KEY).
    llm_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"))
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-5.4-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    retrieval_top_k: int = 4
    relevance_threshold: float = 0.35
    cache_similarity_threshold: float = 0.92

    pdf_path: str = "data/ncert_science.pdf"
    raw_data_dir: str = "data/raw"
    index_dir: str = "data/index"
    db_path: str = "data/globuslearn.db"
    upload_dir: str = "data/uploads"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    history_turns: int = 4

    backend_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
