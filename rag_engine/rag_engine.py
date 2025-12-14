"""
RAG Engine - Core Retrieval-Augmented Generation System
Combines semantic search with LLM generation for enhanced responses
"""

import json
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests

from .config import get_config, RAGConfig
from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, Document, DocumentType, SearchResult, get_vector_store


class ResponseConfidence(str, Enum):
    """Confidence level of RAG response"""
    HIGH = "high"  # Strong match, likely correct answer
    MEDIUM = "medium"  # Good match, may need verification
    LOW = "low"  # Weak match, use with caution
    NO_MATCH = "no_match"  # No relevant documents found


@dataclass
class RAGContext:
    """Context retrieved for a query"""
    query: str
    search_results: List[SearchResult]
    context_text: str
    confidence: ResponseConfidence
    sources: List[Dict[str, Any]]
    retrieval_time_ms: float
    
    @property
    def has_context(self) -> bool:
        return len(self.search_results) > 0
    
    @property
    def top_score(self) -> float:
        if not self.search_results:
            return 0.0
        return self.search_results[0].score


@dataclass
class RAGResponse:
    """Response from the RAG engine"""
    query: str
    response: str
    context: RAGContext
    confidence: ResponseConfidence
    sources: List[Dict[str, Any]]
    generation_time_ms: float
    total_time_ms: float
    tokens_used: int = 0
    model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "confidence": self.confidence.value,
            "sources": self.sources,
            "has_context": self.context.has_context,
            "top_score": self.context.top_score,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_used": self.tokens_used,
            "model": self.model
        }


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    
    Combines semantic search with LLM generation to provide
    context-aware, knowledge-enhanced responses.
    
    Features:
    - Semantic search across KB articles, tickets, runbooks
    - Context-aware response generation
    - Confidence scoring
    - Source attribution
    - Fallback handling
    """
    
    # System prompt for RAG-enhanced responses
    RAG_SYSTEM_PROMPT = """You are AskGen, an enterprise ITSM assistant enhanced with a knowledge base.
You have access to relevant documentation, past resolved tickets, and troubleshooting guides.

When answering questions:
1. Use the provided context to give accurate, specific answers
2. If the context contains a direct solution, provide it step-by-step
3. If the context is partially relevant, use it to inform your response but note any limitations
4. If no relevant context is provided, acknowledge this and provide general guidance
5. Always cite sources when available (e.g., "According to KB article...")
6. Be concise but thorough
7. Prioritize actionable steps the user can take
8. Never fabricate information not in the context or your training

Response format:
- Start with a direct answer or acknowledgment
- Provide step-by-step instructions when applicable
- Include relevant warnings or prerequisites
- End with next steps or follow-up options
"""

    # Prompt template for RAG queries
    RAG_QUERY_TEMPLATE = """## Retrieved Knowledge Context
{context}

## User Query
{query}

## Instructions
Based on the retrieved knowledge above, provide a helpful response to the user's query.
If the knowledge context is relevant, use it to give a specific, accurate answer.
If the context doesn't fully address the query, acknowledge what you can answer and what requires further investigation.
Always maintain a professional, helpful tone appropriate for enterprise IT support."""

    # Prompt for no-context fallback
    NO_CONTEXT_TEMPLATE = """## User Query
{query}

## Instructions
No relevant knowledge base articles or past tickets were found for this query.
Provide general IT support guidance based on your training.
Be clear that this is general advice and recommend creating a ticket if the issue persists.
If the query is outside IT support scope, politely redirect the user."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        
        # HTTP session for LLM calls
        self.session = requests.Session()
        self.session.trust_env = False
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.avg_retrieval_time = 0
        self.avg_generation_time = 0
    
    def _get_completion_url(self) -> str:
        """Get Azure OpenAI completion endpoint URL"""
        base = self.config.azure_openai.endpoint.rstrip('/')
        deployment = self.config.azure_openai.chat_deployment
        version = self.config.azure_openai.api_version
        return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Tuple[str, int]:
        """
        Call the LLM for response generation
        
        Returns:
            Tuple of (response_text, tokens_used)
        """
        url = self._get_completion_url()
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.azure_openai.api_key
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
            verify=self.config.azure_openai.verify_tls
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return content.strip(), tokens
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """Format search results into context string"""
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result.document
            
            # Format based on document type
            if doc.doc_type == DocumentType.KB_ARTICLE:
                header = f"### KB Article: {doc.title}"
            elif doc.doc_type == DocumentType.RESOLVED_TICKET:
                ticket_num = doc.metadata.get("ticket_number", "")
                header = f"### Resolved Ticket {ticket_num}: {doc.title}"
            elif doc.doc_type == DocumentType.RUNBOOK:
                header = f"### Runbook: {doc.title}"
            elif doc.doc_type == DocumentType.FAQ:
                header = f"### FAQ: {doc.title}"
            elif doc.doc_type == DocumentType.TROUBLESHOOTING_GUIDE:
                header = f"### Troubleshooting Guide: {doc.title}"
            else:
                header = f"### Document: {doc.title}"
            
            # Add relevance indicator
            confidence = result.confidence
            relevance = f"[Relevance: {confidence}, Score: {result.score:.2f}]"
            
            context_parts.append(f"{header}\n{relevance}\n\n{doc.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format search results into source citations"""
        sources = []
        for result in results:
            doc = result.document
            sources.append({
                "id": doc.id,
                "type": doc.doc_type.value,
                "title": doc.title,
                "url": doc.url,
                "score": round(result.score, 3),
                "confidence": result.confidence,
                "source": doc.source,
                "source_id": doc.source_id
            })
        return sources
    
    def _determine_confidence(self, results: List[SearchResult]) -> ResponseConfidence:
        """Determine overall confidence based on search results"""
        if not results:
            return ResponseConfidence.NO_MATCH
        
        top_score = results[0].score
        
        if top_score >= self.config.high_confidence_threshold:
            return ResponseConfidence.HIGH
        elif top_score >= self.config.medium_confidence_threshold:
            return ResponseConfidence.MEDIUM
        else:
            return ResponseConfidence.LOW
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_types: Optional[List[DocumentType]] = None,
        min_score: Optional[float] = None
    ) -> RAGContext:
        """
        Retrieve relevant context for a query
        
        Args:
            query: The user's query
            top_k: Number of results to retrieve
            doc_types: Filter by document types
            min_score: Minimum similarity score
            
        Returns:
            RAGContext with retrieved documents
        """
        start_time = time.time()
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k or self.config.vector_store.default_top_k,
            doc_types=doc_types,
            min_score=min_score or self.config.vector_store.similarity_threshold
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        # Format context
        context_text = self._format_context(results)
        sources = self._format_sources(results)
        confidence = self._determine_confidence(results)
        
        return RAGContext(
            query=query,
            search_results=results,
            context_text=context_text,
            confidence=confidence,
            sources=sources,
            retrieval_time_ms=retrieval_time
        )
    
    def generate(
        self,
        query: str,
        context: RAGContext,
        additional_context: Optional[str] = None,
        temperature: float = 0.3
    ) -> RAGResponse:
        """
        Generate a response using retrieved context
        
        Args:
            query: The user's query
            context: Retrieved RAG context
            additional_context: Extra context (e.g., session state)
            temperature: LLM temperature for generation
            
        Returns:
            RAGResponse with generated answer
        """
        start_time = time.time()
        
        # Build messages
        messages = [
            {"role": "system", "content": self.RAG_SYSTEM_PROMPT}
        ]
        
        # Format user message based on context availability
        if context.has_context:
            user_content = self.RAG_QUERY_TEMPLATE.format(
                context=context.context_text,
                query=query
            )
        else:
            user_content = self.NO_CONTEXT_TEMPLATE.format(query=query)
        
        # Add additional context if provided
        if additional_context:
            user_content += f"\n\n## Additional Context\n{additional_context}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Generate response
        response_text, tokens_used = self._call_llm(
            messages,
            temperature=temperature
        )
        
        generation_time = (time.time() - start_time) * 1000
        total_time = context.retrieval_time_ms + generation_time
        
        # Update statistics
        self.total_queries += 1
        
        return RAGResponse(
            query=query,
            response=response_text,
            context=context,
            confidence=context.confidence,
            sources=context.sources,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            tokens_used=tokens_used,
            model=self.config.azure_openai.chat_deployment
        )
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_types: Optional[List[DocumentType]] = None,
        min_score: Optional[float] = None,
        additional_context: Optional[str] = None,
        temperature: float = 0.3
    ) -> RAGResponse:
        """
        Complete RAG query: retrieve then generate
        
        Args:
            query: The user's query
            top_k: Number of documents to retrieve
            doc_types: Filter by document types
            min_score: Minimum similarity score
            additional_context: Extra context
            temperature: LLM temperature
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve context
        context = self.retrieve(
            query=query,
            top_k=top_k,
            doc_types=doc_types,
            min_score=min_score
        )
        
        # Generate response
        return self.generate(
            query=query,
            context=context,
            additional_context=additional_context,
            temperature=temperature
        )
    
    def query_with_history(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
        session_state: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """
        RAG query with conversation history
        
        Args:
            query: Current user query
            chat_history: List of (speaker, message) tuples
            session_state: Current session state
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with context-aware answer
        """
        # Build context from history
        history_context = []
        
        # Include recent conversation for context
        recent_history = chat_history[-6:] if chat_history else []
        for speaker, message in recent_history:
            role = "User" if speaker.lower() in ("you", "user") else "Assistant"
            history_context.append(f"{role}: {message}")
        
        # Add relevant session state
        state_context = []
        if session_state:
            if session_state.get("original_problem"):
                state_context.append(f"Original Problem: {session_state['original_problem']}")
            if session_state.get("type"):
                state_context.append(f"Issue Type: {session_state['type']}")
            if session_state.get("classification"):
                cls = session_state["classification"]
                if isinstance(cls, dict):
                    state_context.append(f"Category: {cls.get('category', 'unknown')}")
        
        additional_context = ""
        if history_context:
            additional_context += "## Conversation History\n" + "\n".join(history_context)
        if state_context:
            additional_context += "\n\n## Session Context\n" + "\n".join(state_context)
        
        return self.query(
            query=query,
            top_k=top_k,
            additional_context=additional_context if additional_context else None
        )
    
    def find_similar_issues(
        self,
        problem_description: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find similar past issues/tickets
        
        Args:
            problem_description: Description of the current issue
            top_k: Number of similar issues to find
            
        Returns:
            List of similar resolved tickets
        """
        results = self.vector_store.search(
            query=problem_description,
            top_k=top_k,
            doc_types=[DocumentType.RESOLVED_TICKET, DocumentType.TROUBLESHOOTING_GUIDE]
        )
        return results
    
    def find_relevant_kb(
        self,
        query: str,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Find relevant KB articles
        
        Args:
            query: User query
            top_k: Number of articles to find
            
        Returns:
            List of relevant KB articles
        """
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            doc_types=[DocumentType.KB_ARTICLE, DocumentType.FAQ]
        )
        return results
    
    def find_runbook(
        self,
        issue_description: str,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Find relevant runbooks for an issue
        
        Args:
            issue_description: Description of the issue
            top_k: Number of runbooks to find
            
        Returns:
            List of relevant runbooks
        """
        results = self.vector_store.search(
            query=issue_description,
            top_k=top_k,
            doc_types=[DocumentType.RUNBOOK, DocumentType.PROCEDURE]
        )
        return results
    
    def stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            "total_queries": self.total_queries,
            "vector_store": self.vector_store.stats(),
            "embedding_service": self.embedding_service.stats()
        }


# Singleton instance
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
