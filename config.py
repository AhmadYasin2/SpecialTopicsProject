"""
Configuration management for AutoPRISMA system.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"  # Ollama server URL
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:32b"  # Ollama Qwen 32B model
    llm_temperature: float = 0.1  # Low temperature for reproducibility
    
    # Academic API Keys
    semantic_scholar_api_key: Optional[str] = None
    ncbi_api_key: Optional[str] = None
    openalex_email: Optional[str] = None
    
    # Application Settings
    log_level: str = "INFO"
    enable_audit_trail: bool = True
    vector_store_path: str = "./data/vector_store"
    document_store_path: str = "./data/documents"
    state_store_path: str = "./data/state"
    
    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit
    streamlit_port: int = 8501
    
    # Agent Configuration
    max_papers_per_query: int = 500
    screening_batch_size: int = 10
    confidence_threshold: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create data directories if they don't exist
        for path in [self.vector_store_path, self.document_store_path, self.state_store_path]:
            os.makedirs(path, exist_ok=True)


# Global settings instance
settings = Settings()
