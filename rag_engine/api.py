"""
RAG Engine API Endpoints
FastAPI router for RAG operations
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from .rag_engine import RAGEngine, RAGResponse, get_rag_engine
from .vector_store import DocumentType, Document, get_vector_store
from .knowledge_learner import KnowledgeLearner, FeedbackEntry, get_knowledge_learner
from .document_processor import get_document_processor


# Create router
router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ============================================================================
# Request/Response Models
# ============================================================================

class RAGQueryRequest(BaseModel):
    """Request for RAG query"""
    query: str = Field(..., min_length=1, description="The user's query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    doc_types: Optional[List[str]] = Field(None, description="Filter by document types")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum similarity score")
    include_sources: Optional[bool] = Field(True, description="Include source documents")
    session_context: Optional[Dict[str, Any]] = Field(None, description="Session context")


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""
    query: str
    response: str
    confidence: str
    sources: List[Dict[str, Any]]
    has_context: bool
    top_score: float
    generation_time_ms: float
    total_time_ms: float


class SemanticSearchRequest(BaseModel):
    """Request for semantic search"""
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(10, ge=1, le=50)
    doc_types: Optional[List[str]] = Field(None)
    min_score: Optional[float] = Field(0.5, ge=0, le=1)


class SearchResultItem(BaseModel):
    """Single search result"""
    id: str
    title: str
    content: str
    doc_type: str
    score: float
    confidence: str
    source: str
    url: Optional[str]


class SemanticSearchResponse(BaseModel):
    """Response from semantic search"""
    query: str
    results: List[SearchResultItem]
    total_results: int
    retrieval_time_ms: float


class AddDocumentRequest(BaseModel):
    """Request to add a document"""
    content: str = Field(..., min_length=10)
    title: str = Field(..., min_length=1)
    doc_type: str = Field("custom")
    source: str = Field("manual")
    source_id: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddKBArticleRequest(BaseModel):
    """Request to add a KB article"""
    sys_id: str
    title: str
    content: str
    short_description: Optional[str] = ""
    category: Optional[str] = ""
    url: Optional[str] = ""


class AddTroubleshootingGuideRequest(BaseModel):
    """Request to add a troubleshooting guide"""
    title: str
    symptoms: List[str]
    causes: List[str]
    solutions: List[str]
    category: Optional[str] = ""
    tags: Optional[List[str]] = None


class AddFAQRequest(BaseModel):
    """Request to add an FAQ"""
    question: str
    answer: str
    category: Optional[str] = ""
    tags: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    """Request to submit feedback"""
    query: str
    response: str
    helpful: bool
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    sources_used: Optional[List[str]] = None


class SyncRequest(BaseModel):
    """Request to sync from ServiceNow"""
    sync_type: str = Field(..., description="Type: 'kb', 'incidents', 'requests', 'all'")
    days_back: Optional[int] = Field(90, ge=1, le=365)
    limit: Optional[int] = Field(500, ge=1, le=2000)
    categories: Optional[List[str]] = None


class SyncResponse(BaseModel):
    """Response from sync operation"""
    source: str
    documents_processed: int
    documents_added: int
    documents_updated: int
    documents_skipped: int
    errors: List[str]
    duration_ms: float
    success: bool


# ============================================================================
# RAG Query Endpoints
# ============================================================================

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system for an enhanced response
    
    This endpoint:
    1. Searches the knowledge base for relevant context
    2. Generates a response using the LLM with retrieved context
    3. Returns the response with source citations
    """
    try:
        rag_engine = get_rag_engine()
        
        # Convert doc_types strings to enums
        doc_types = None
        if request.doc_types:
            doc_types = [DocumentType(dt) for dt in request.doc_types]
        
        # Build additional context from session
        additional_context = None
        if request.session_context:
            ctx_parts = []
            if request.session_context.get("original_problem"):
                ctx_parts.append(f"Original Problem: {request.session_context['original_problem']}")
            if request.session_context.get("chat_history"):
                history = request.session_context["chat_history"][-4:]
                ctx_parts.append("Recent conversation:")
                for speaker, msg in history:
                    ctx_parts.append(f"  {speaker}: {msg[:200]}")
            if ctx_parts:
                additional_context = "\n".join(ctx_parts)
        
        # Query RAG
        result = rag_engine.query(
            query=request.query,
            top_k=request.top_k,
            doc_types=doc_types,
            min_score=request.min_score,
            additional_context=additional_context
        )
        
        return RAGQueryResponse(
            query=result.query,
            response=result.response,
            confidence=result.confidence.value,
            sources=result.sources if request.include_sources else [],
            has_context=result.context.has_context,
            top_score=result.context.top_score,
            generation_time_ms=result.generation_time_ms,
            total_time_ms=result.total_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search across the knowledge base
    
    Returns relevant documents without generating a response
    """
    try:
        vector_store = get_vector_store()
        
        doc_types = None
        if request.doc_types:
            doc_types = [DocumentType(dt) for dt in request.doc_types]
        
        import time
        start = time.time()
        
        results = vector_store.search(
            query=request.query,
            top_k=request.top_k,
            doc_types=doc_types,
            min_score=request.min_score
        )
        
        retrieval_time = (time.time() - start) * 1000
        
        items = []
        for r in results:
            items.append(SearchResultItem(
                id=r.document.id,
                title=r.document.title,
                content=r.document.content[:500] + "..." if len(r.document.content) > 500 else r.document.content,
                doc_type=r.document.doc_type.value,
                score=round(r.score, 3),
                confidence=r.confidence,
                source=r.document.source,
                url=r.document.url
            ))
        
        return SemanticSearchResponse(
            query=request.query,
            results=items,
            total_results=len(items),
            retrieval_time_ms=retrieval_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar-issues")
async def find_similar_issues(query: str, top_k: int = 5):
    """
    Find similar past issues/tickets for a given problem description
    """
    try:
        rag_engine = get_rag_engine()
        results = rag_engine.find_similar_issues(query, top_k=top_k)
        
        return {
            "query": query,
            "similar_issues": [
                {
                    "id": r.document.id,
                    "title": r.document.title,
                    "ticket_number": r.document.metadata.get("ticket_number", ""),
                    "resolution": r.document.content,
                    "score": round(r.score, 3),
                    "confidence": r.confidence
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relevant-kb")
async def find_relevant_kb(query: str, top_k: int = 3):
    """
    Find relevant KB articles for a query
    """
    try:
        rag_engine = get_rag_engine()
        results = rag_engine.find_relevant_kb(query, top_k=top_k)
        
        return {
            "query": query,
            "articles": [
                {
                    "id": r.document.id,
                    "title": r.document.title,
                    "summary": r.document.content[:300] + "..." if len(r.document.content) > 300 else r.document.content,
                    "url": r.document.url,
                    "score": round(r.score, 3),
                    "confidence": r.confidence
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document Management Endpoints
# ============================================================================

@router.post("/documents")
async def add_document(request: AddDocumentRequest):
    """
    Add a custom document to the knowledge base
    """
    try:
        doc_processor = get_document_processor()
        vector_store = get_vector_store()
        
        doc_type = DocumentType.CUSTOM
        try:
            doc_type = DocumentType(request.doc_type)
        except:
            pass
        
        docs = doc_processor.process_raw_text(
            text=request.content,
            title=request.title,
            doc_type=doc_type,
            source=request.source,
            metadata=request.metadata
        )
        
        ids = vector_store.add_documents(docs)
        
        return {
            "success": True,
            "document_ids": ids,
            "chunks_created": len(ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/kb-article")
async def add_kb_article(request: AddKBArticleRequest):
    """
    Add a KB article to the knowledge base
    """
    try:
        doc_processor = get_document_processor()
        vector_store = get_vector_store()
        
        docs = doc_processor.process_kb_article(
            sys_id=request.sys_id,
            title=request.title,
            content=request.content,
            short_description=request.short_description,
            category=request.category,
            url=request.url
        )
        
        ids = vector_store.add_documents(docs)
        
        return {
            "success": True,
            "document_ids": ids,
            "chunks_created": len(ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/troubleshooting-guide")
async def add_troubleshooting_guide(request: AddTroubleshootingGuideRequest):
    """
    Add a troubleshooting guide to the knowledge base
    """
    try:
        learner = get_knowledge_learner()
        
        doc_id = learner.add_troubleshooting_guide(
            title=request.title,
            symptoms=request.symptoms,
            causes=request.causes,
            solutions=request.solutions,
            category=request.category,
            tags=request.tags
        )
        
        return {
            "success": True,
            "document_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/faq")
async def add_faq(request: AddFAQRequest):
    """
    Add an FAQ to the knowledge base
    """
    try:
        learner = get_knowledge_learner()
        
        doc_id = learner.add_faq(
            question=request.question,
            answer=request.answer,
            category=request.category,
            tags=request.tags
        )
        
        return {
            "success": True,
            "document_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """
    Get a document by ID
    """
    try:
        vector_store = get_vector_store()
        doc = vector_store.get_document(doc_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "doc_type": doc.doc_type.value,
            "source": doc.source,
            "source_id": doc.source_id,
            "url": doc.url,
            "metadata": doc.metadata,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from the knowledge base
    """
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_document(doc_id)
        
        return {"success": success, "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Learning & Sync Endpoints
# ============================================================================

@router.post("/sync", response_model=SyncResponse)
async def sync_knowledge(request: SyncRequest, background_tasks: BackgroundTasks):
    """
    Sync knowledge from ServiceNow
    
    Sync types:
    - 'kb': Sync KB articles
    - 'incidents': Sync resolved incidents
    - 'requests': Sync resolved requests
    - 'all': Sync everything
    """
    try:
        learner = get_knowledge_learner()
        
        if request.sync_type == "kb":
            result = learner.sync_kb_articles(
                limit=request.limit,
                categories=request.categories
            )
        elif request.sync_type == "incidents":
            result = learner.sync_resolved_incidents(
                days_back=request.days_back,
                limit=request.limit,
                categories=request.categories
            )
        elif request.sync_type == "requests":
            result = learner.sync_resolved_requests(
                days_back=request.days_back,
                limit=request.limit
            )
        elif request.sync_type == "all":
            # Run full sync in background for 'all'
            results = learner.sync_all()
            # Return combined stats
            total_added = sum(r.documents_added for r in results.values())
            total_processed = sum(r.documents_processed for r in results.values())
            all_errors = []
            for r in results.values():
                all_errors.extend(r.errors)
            
            return SyncResponse(
                source="all",
                documents_processed=total_processed,
                documents_added=total_added,
                documents_updated=0,
                documents_skipped=0,
                errors=all_errors[:10],  # Limit errors
                duration_ms=sum(r.duration_ms for r in results.values()),
                success=len(all_errors) == 0
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown sync_type: {request.sync_type}")
        
        return SyncResponse(**result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a RAG response
    """
    try:
        learner = get_knowledge_learner()
        
        feedback = FeedbackEntry(
            query=request.query,
            response=request.response,
            helpful=request.helpful,
            rating=request.rating,
            comment=request.comment,
            sources_used=request.sources_used or []
        )
        
        success = learner.record_feedback(feedback)
        
        return {"success": success, "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn-resolution")
async def learn_from_resolution(
    ticket_number: str,
    problem_description: str,
    resolution_steps: str,
    category: str = "",
    was_helpful: bool = True
):
    """
    Learn from a manual resolution
    """
    try:
        learner = get_knowledge_learner()
        
        doc_id = learner.learn_from_resolution(
            ticket_number=ticket_number,
            problem_description=problem_description,
            resolution_steps=resolution_steps,
            category=category,
            was_helpful=was_helpful
        )
        
        if doc_id:
            return {"success": True, "document_id": doc_id}
        else:
            return {"success": False, "message": "Resolution did not meet quality threshold"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics & Health Endpoints
# ============================================================================

@router.get("/stats")
async def get_stats():
    """
    Get RAG system statistics
    """
    try:
        rag_engine = get_rag_engine()
        learner = get_knowledge_learner()
        
        return {
            "rag_engine": rag_engine.stats(),
            "learning": learner.stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    RAG system health check
    """
    try:
        vector_store = get_vector_store()
        
        return {
            "status": "healthy",
            "vector_store": vector_store.stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/document-types")
async def get_document_types():
    """
    Get available document types
    """
    return {
        "document_types": [
            {"value": dt.value, "name": dt.name}
            for dt in DocumentType
        ]
    }
