# RAG Engine for AskGen ITOps Platform
# Provides semantic search and knowledge-enhanced AI responses

from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, get_vector_store, Document, DocumentType
from .document_processor import DocumentProcessor, get_document_processor
from .rag_engine import RAGEngine, get_rag_engine, RAGResponse, RAGContext
from .knowledge_learner import KnowledgeLearner, get_knowledge_learner, FeedbackEntry

__all__ = [
    # Core classes
    'EmbeddingService',
    'VectorStore', 
    'DocumentProcessor',
    'RAGEngine',
    'KnowledgeLearner',
    # Singleton getters
    'get_embedding_service',
    'get_vector_store',
    'get_document_processor',
    'get_rag_engine',
    'get_knowledge_learner',
    # Data classes
    'Document',
    'DocumentType',
    'RAGResponse',
    'RAGContext',
    'FeedbackEntry'
]

__version__ = '1.0.0'
