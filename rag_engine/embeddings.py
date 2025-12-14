"""
Embedding Service for RAG Engine
Generates vector embeddings using Azure OpenAI
"""

import os
import hashlib
import json
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import requests
from functools import lru_cache

from .config import get_config, AzureOpenAIConfig


@dataclass
class EmbeddingResult:
    """Result of an embedding operation"""
    text: str
    embedding: List[float]
    model: str
    tokens_used: int
    cached: bool = False


class EmbeddingCache:
    """Simple in-memory cache for embeddings to reduce API calls"""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Create a hash key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists"""
        key = self._hash_text(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cache.keys())[:self.max_size // 10]
            for key in keys_to_remove:
                del self.cache[key]
        
        key = self._hash_text(text)
        self.cache[key] = embedding
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0
        }


class EmbeddingService:
    """
    Service for generating text embeddings using Azure OpenAI
    
    Features:
    - Batch embedding support
    - Automatic retry with exponential backoff
    - In-memory caching to reduce API calls
    - Token counting and cost estimation
    """
    
    def __init__(self, config: Optional[AzureOpenAIConfig] = None):
        self.config = config or get_config().azure_openai
        self.cache = EmbeddingCache()
        self.session = requests.Session()
        self.session.trust_env = False  # Bypass proxy settings
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
    
    def _get_embedding_url(self) -> str:
        """Construct the Azure OpenAI embedding API URL"""
        base = self.config.endpoint.rstrip('/')
        return f"{base}/openai/deployments/{self.config.embedding_deployment}/embeddings?api-version={self.config.api_version}"
    
    def _call_api(self, texts: List[str], retry_count: int = 3) -> Dict[str, Any]:
        """Make API call with retry logic"""
        url = self._get_embedding_url()
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key
        }
        payload = {
            "input": texts,
            "model": self.config.embedding_deployment
        }
        
        last_error = None
        for attempt in range(retry_count):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    verify=self.config.verify_tls
                )
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 5))
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                self.total_requests += 1
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        self.total_errors += 1
        raise Exception(f"Embedding API call failed after {retry_count} retries: {last_error}")
    
    def embed_text(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        Generate embedding for a single text
        
        Args:
            text: The text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            EmbeddingResult with embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        text = text.strip()
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self.config.embedding_deployment,
                    tokens_used=0,
                    cached=True
                )
        
        # Call API
        response = self._call_api([text])
        
        embedding = response["data"][0]["embedding"]
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        self.total_tokens += tokens_used
        
        # Cache the result
        if use_cache:
            self.cache.set(text, embedding)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=response.get("model", self.config.embedding_deployment),
            tokens_used=tokens_used,
            cached=False
        )
    
    def embed_batch(self, texts: List[str], use_cache: bool = True, batch_size: int = 100) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            batch_size: Maximum texts per API call
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        results = []
        texts_to_embed = []
        text_indices = []  # Track which original texts need embedding
        
        # Check cache first
        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if not text:
                results.append(None)
                continue
                
            if use_cache:
                cached = self.cache.get(text)
                if cached is not None:
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=cached,
                        model=self.config.embedding_deployment,
                        tokens_used=0,
                        cached=True
                    ))
                    continue
            
            texts_to_embed.append(text)
            text_indices.append(i)
            results.append(None)  # Placeholder
        
        # Embed uncached texts in batches
        for batch_start in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[batch_start:batch_start + batch_size]
            batch_indices = text_indices[batch_start:batch_start + batch_size]
            
            response = self._call_api(batch_texts)
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            self.total_tokens += tokens_used
            
            for j, embedding_data in enumerate(response["data"]):
                idx = batch_indices[j]
                text = batch_texts[j]
                embedding = embedding_data["embedding"]
                
                if use_cache:
                    self.cache.set(text, embedding)
                
                results[idx] = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=response.get("model", self.config.embedding_deployment),
                    tokens_used=tokens_used // len(batch_texts),  # Approximate per-text tokens
                    cached=False
                )
        
        # Filter out None values (empty texts)
        return [r for r in results if r is not None]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimensions")
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "cache": self.cache.stats()
        }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
