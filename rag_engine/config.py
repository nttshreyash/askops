"""
RAG Engine Configuration
Centralized configuration for the RAG system
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration for embeddings and completions"""
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://sweden-central-inst.openai.azure.com/")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    
    # Deployment names
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    chat_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    # Embedding model dimensions (ada-002 = 1536, text-embedding-3-small = 1536, text-embedding-3-large = 3072)
    embedding_dimensions: int = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
    
    verify_tls: bool = os.getenv("AZURE_VERIFY_TLS", "false").lower() == "true"


@dataclass
class VectorStoreConfig:
    """Vector database configuration"""
    # ChromaDB settings
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "askgen_knowledge")
    
    # Search settings
    default_top_k: int = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
    
    # Indexing settings
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))


@dataclass
class ServiceNowConfig:
    """ServiceNow configuration for knowledge sync"""
    instance: str = os.getenv("SERVICE_NOW_INSTANCE", "dev300144.service-now.com")
    user: str = os.getenv("SERVICE_NOW_USER", "admin")
    password: str = os.getenv("SERVICE_NOW_PASS", "")
    verify_tls: bool = os.getenv("VERIFY_TLS", "true").lower() == "true"
    
    # Sync settings
    kb_sync_limit: int = int(os.getenv("KB_SYNC_LIMIT", "500"))
    ticket_sync_days: int = int(os.getenv("TICKET_SYNC_DAYS", "90"))


@dataclass
class RAGConfig:
    """Main RAG engine configuration"""
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    servicenow: ServiceNowConfig = field(default_factory=ServiceNowConfig)
    
    # RAG behavior settings
    enable_kb_search: bool = os.getenv("RAG_ENABLE_KB", "true").lower() == "true"
    enable_ticket_search: bool = os.getenv("RAG_ENABLE_TICKETS", "true").lower() == "true"
    enable_learning: bool = os.getenv("RAG_ENABLE_LEARNING", "true").lower() == "true"
    
    # Context settings
    max_context_tokens: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "4000"))
    include_sources: bool = os.getenv("RAG_INCLUDE_SOURCES", "true").lower() == "true"
    
    # Confidence thresholds
    high_confidence_threshold: float = float(os.getenv("RAG_HIGH_CONFIDENCE", "0.85"))
    medium_confidence_threshold: float = float(os.getenv("RAG_MEDIUM_CONFIDENCE", "0.7"))


# Global configuration instance
_config: Optional[RAGConfig] = None

def get_config() -> RAGConfig:
    """Get or create the global RAG configuration"""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config

def reload_config() -> RAGConfig:
    """Force reload configuration from environment"""
    global _config
    _config = RAGConfig()
    return _config
