"""
Vector Store for RAG Engine
Manages document storage and semantic search using ChromaDB
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config import get_config, VectorStoreConfig
from .embeddings import EmbeddingService, get_embedding_service


class DocumentType(str, Enum):
    """Types of documents stored in the vector database"""
    KB_ARTICLE = "kb_article"
    RESOLVED_TICKET = "resolved_ticket"
    RUNBOOK = "runbook"
    FAQ = "faq"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    POLICY = "policy"
    PROCEDURE = "procedure"
    CUSTOM = "custom"


@dataclass
class Document:
    """A document to be stored in the vector database"""
    id: str
    content: str
    doc_type: DocumentType
    title: str = ""
    source: str = ""  # e.g., "servicenow", "confluence", "manual"
    source_id: str = ""  # Original ID in source system
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Optional: pre-computed embedding
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, DocumentType) else self.doc_type,
            "title": self.title,
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary"""
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        doc_type = data.get("doc_type", DocumentType.CUSTOM)
        if isinstance(doc_type, str):
            try:
                doc_type = DocumentType(doc_type)
            except:
                doc_type = DocumentType.CUSTOM
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            doc_type=doc_type,
            title=data.get("title", ""),
            source=data.get("source", ""),
            source_id=data.get("source_id", ""),
            url=data.get("url", ""),
            metadata=metadata,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat())
        )


@dataclass
class SearchResult:
    """A search result from the vector store"""
    document: Document
    score: float  # Similarity score (0-1, higher is better)
    rank: int
    
    @property
    def confidence(self) -> str:
        """Human-readable confidence level"""
        if self.score >= 0.85:
            return "high"
        elif self.score >= 0.70:
            return "medium"
        elif self.score >= 0.50:
            return "low"
        else:
            return "very_low"


class VectorStore:
    """
    Vector database for storing and searching documents
    
    Uses ChromaDB for local persistence with semantic search capabilities.
    Falls back to in-memory storage if ChromaDB is not available.
    """
    
    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.config = config or get_config().vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Try to use ChromaDB, fall back to in-memory
        self.use_chroma = False
        self.client = None
        self.collection = None
        
        # In-memory fallback storage
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}
        
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store backend"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if needed
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB with persistence
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.persist_directory,
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.use_chroma = True
            print(f"[RAG] Initialized ChromaDB at {self.config.persist_directory}")
            
        except ImportError:
            print("[RAG] ChromaDB not available, using in-memory storage")
            self.use_chroma = False
        except Exception as e:
            print(f"[RAG] ChromaDB initialization failed: {e}, using in-memory storage")
            self.use_chroma = False
    
    def add_document(self, document: Document) -> str:
        """
        Add a document to the vector store
        
        Args:
            document: The document to add
            
        Returns:
            The document ID
        """
        # Generate embedding if not provided
        if document.embedding is None:
            result = self.embedding_service.embed_text(document.content)
            document.embedding = result.embedding
        
        document.updated_at = datetime.utcnow().isoformat()
        
        if self.use_chroma:
            self.collection.add(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[document.to_dict()]
            )
        else:
            self._documents[document.id] = document
            self._embeddings[document.id] = document.embedding
        
        return document.id
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add multiple documents to the vector store
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for embedding generation
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate embeddings for documents that don't have them
        texts_to_embed = []
        embed_indices = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                embed_indices.append(i)
        
        if texts_to_embed:
            embeddings = self.embedding_service.embed_batch(texts_to_embed, batch_size=batch_size)
            for j, result in enumerate(embeddings):
                documents[embed_indices[j]].embedding = result.embedding
        
        # Update timestamps
        now = datetime.utcnow().isoformat()
        for doc in documents:
            doc.updated_at = now
        
        ids = [doc.id for doc in documents]
        
        if self.use_chroma:
            # Add in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.collection.add(
                    ids=[doc.id for doc in batch],
                    embeddings=[doc.embedding for doc in batch],
                    documents=[doc.content for doc in batch],
                    metadatas=[doc.to_dict() for doc in batch]
                )
        else:
            for doc in documents:
                self._documents[doc.id] = doc
                self._embeddings[doc.id] = doc.embedding
        
        return ids
    
    def update_document(self, document: Document) -> bool:
        """
        Update an existing document
        
        Args:
            document: The document with updated content
            
        Returns:
            True if updated successfully
        """
        # Regenerate embedding
        result = self.embedding_service.embed_text(document.content)
        document.embedding = result.embedding
        document.updated_at = datetime.utcnow().isoformat()
        
        if self.use_chroma:
            self.collection.update(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[document.to_dict()]
            )
        else:
            self._documents[document.id] = document
            self._embeddings[document.id] = document.embedding
        
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store
        
        Args:
            doc_id: The document ID to delete
            
        Returns:
            True if deleted successfully
        """
        if self.use_chroma:
            self.collection.delete(ids=[doc_id])
        else:
            self._documents.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID
        
        Args:
            doc_id: The document ID
            
        Returns:
            The document or None if not found
        """
        if self.use_chroma:
            results = self.collection.get(ids=[doc_id], include=["metadatas", "documents"])
            if results and results["ids"]:
                metadata = results["metadatas"][0]
                return Document.from_dict(metadata)
        else:
            return self._documents.get(doc_id)
        
        return None
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_types: Optional[List[DocumentType]] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search for documents
        
        Args:
            query: The search query
            top_k: Number of results to return
            doc_types: Filter by document types
            min_score: Minimum similarity score
            filters: Additional metadata filters
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.config.default_top_k
        min_score = min_score or self.config.similarity_threshold
        
        # Generate query embedding
        query_result = self.embedding_service.embed_text(query)
        query_embedding = query_result.embedding
        
        results = []
        
        if self.use_chroma:
            # Build where clause for filters
            where = {}
            if doc_types:
                where["doc_type"] = {"$in": [dt.value for dt in doc_types]}
            if filters:
                where.update(filters)
            
            # Search
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results for filtering
                where=where if where else None,
                include=["metadatas", "documents", "distances"]
            )
            
            if search_results and search_results["ids"] and search_results["ids"][0]:
                for i, doc_id in enumerate(search_results["ids"][0]):
                    # ChromaDB returns distance, convert to similarity
                    distance = search_results["distances"][0][i]
                    score = 1 - distance  # Cosine distance to similarity
                    
                    if score >= min_score:
                        metadata = search_results["metadatas"][0][i]
                        doc = Document.from_dict(metadata)
                        results.append(SearchResult(
                            document=doc,
                            score=score,
                            rank=len(results) + 1
                        ))
        else:
            # In-memory search
            scored_docs = []
            for doc_id, embedding in self._embeddings.items():
                doc = self._documents.get(doc_id)
                if not doc:
                    continue
                
                # Apply doc_type filter
                if doc_types and doc.doc_type not in doc_types:
                    continue
                
                # Calculate similarity
                score = self.embedding_service.similarity(query_embedding, embedding)
                
                if score >= min_score:
                    scored_docs.append((doc, score))
            
            # Sort by score descending
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (doc, score) in enumerate(scored_docs[:top_k]):
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    rank=i + 1
                ))
        
        # Ensure we only return top_k results
        return results[:top_k]
    
    def search_similar(
        self,
        doc_id: str,
        top_k: Optional[int] = None,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: The document ID to find similar documents for
            top_k: Number of results to return
            exclude_self: Whether to exclude the query document
            
        Returns:
            List of similar documents
        """
        doc = self.get_document(doc_id)
        if not doc:
            return []
        
        results = self.search(doc.content, top_k=(top_k or self.config.default_top_k) + 1)
        
        if exclude_self:
            results = [r for r in results if r.document.id != doc_id]
        
        return results[:top_k or self.config.default_top_k]
    
    def count(self, doc_type: Optional[DocumentType] = None) -> int:
        """
        Count documents in the store
        
        Args:
            doc_type: Optional filter by document type
            
        Returns:
            Number of documents
        """
        if self.use_chroma:
            if doc_type:
                results = self.collection.get(
                    where={"doc_type": doc_type.value},
                    include=[]
                )
                return len(results["ids"]) if results["ids"] else 0
            return self.collection.count()
        else:
            if doc_type:
                return sum(1 for d in self._documents.values() if d.doc_type == doc_type)
            return len(self._documents)
    
    def stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        total = self.count()
        by_type = {}
        for dt in DocumentType:
            by_type[dt.value] = self.count(dt)
        
        return {
            "backend": "chromadb" if self.use_chroma else "in_memory",
            "total_documents": total,
            "by_type": by_type,
            "persist_directory": self.config.persist_directory if self.use_chroma else None
        }
    
    def persist(self):
        """Persist the vector store to disk (ChromaDB only)"""
        if self.use_chroma and self.client:
            self.client.persist()
            print(f"[RAG] Persisted vector store to {self.config.persist_directory}")


# Singleton instance
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get or create the global vector store"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
