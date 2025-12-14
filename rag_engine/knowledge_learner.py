"""
Knowledge Learner for RAG Engine
Automatically learns from resolved tickets and user feedback
"""

import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
from requests.auth import HTTPBasicAuth

from .config import get_config, RAGConfig
from .vector_store import VectorStore, Document, DocumentType, get_vector_store
from .document_processor import DocumentProcessor, get_document_processor


class LearningSource(str, Enum):
    """Sources for knowledge learning"""
    SERVICENOW_INCIDENT = "servicenow_incident"
    SERVICENOW_REQUEST = "servicenow_request"
    SERVICENOW_KB = "servicenow_kb"
    USER_FEEDBACK = "user_feedback"
    MANUAL_IMPORT = "manual_import"


@dataclass
class LearningResult:
    """Result of a learning operation"""
    source: LearningSource
    documents_processed: int
    documents_added: int
    documents_updated: int
    documents_skipped: int
    errors: List[str]
    duration_ms: float
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "documents_processed": self.documents_processed,
            "documents_added": self.documents_added,
            "documents_updated": self.documents_updated,
            "documents_skipped": self.documents_skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "success": self.success
        }


@dataclass
class FeedbackEntry:
    """User feedback on a response"""
    query: str
    response: str
    helpful: bool
    rating: Optional[int] = None  # 1-5 scale
    comment: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: Optional[str] = None


class KnowledgeLearner:
    """
    Learns from various sources to continuously improve the knowledge base
    
    Features:
    - Sync resolved tickets from ServiceNow
    - Sync KB articles from ServiceNow
    - Learn from user feedback
    - Automatic quality scoring
    - Duplicate detection
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        vector_store: Optional[VectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or get_vector_store()
        self.doc_processor = document_processor or get_document_processor()
        
        # HTTP session for ServiceNow
        self.session = requests.Session()
        self.session.trust_env = False
        
        # Learning statistics
        self.total_syncs = 0
        self.last_sync_time: Optional[datetime] = None
        self.feedback_entries: List[FeedbackEntry] = []
    
    def _sn_api_call(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make a ServiceNow API call"""
        base_url = f"https://{self.config.servicenow.instance}/api/now/table"
        url = f"{base_url}/{endpoint}"
        
        try:
            response = self.session.get(
                url,
                auth=HTTPBasicAuth(
                    self.config.servicenow.user,
                    self.config.servicenow.password
                ),
                params=params or {},
                headers={"Accept": "application/json"},
                timeout=30,
                verify=self.config.servicenow.verify_tls
            )
            response.raise_for_status()
            return response.json().get("result", [])
        except Exception as e:
            print(f"[RAG Learning] ServiceNow API error: {e}")
            return []
    
    def _is_quality_resolution(self, resolution: str) -> bool:
        """Check if a resolution is detailed enough to be useful"""
        if not resolution:
            return False
        
        # Minimum length requirement
        if len(resolution) < 50:
            return False
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            r"^resolved$",
            r"^fixed$",
            r"^done$",
            r"^completed$",
            r"^closed$",
            r"^n/a$",
            r"^not applicable$",
            r"^user error$",
            r"^duplicate$",
            r"^test",
        ]
        
        resolution_lower = resolution.lower().strip()
        for pattern in low_quality_patterns:
            if re.match(pattern, resolution_lower):
                return False
        
        return True
    
    def sync_resolved_incidents(
        self,
        days_back: Optional[int] = None,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> LearningResult:
        """
        Sync resolved incidents from ServiceNow
        
        Args:
            days_back: Number of days to look back
            limit: Maximum incidents to sync
            categories: Filter by categories
            
        Returns:
            LearningResult with sync statistics
        """
        start_time = time.time()
        
        days = days_back or self.config.servicenow.ticket_sync_days
        max_records = limit or 500
        
        # Build query for resolved incidents with resolution notes
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"state=6^ORstate=7^sys_updated_on>={cutoff_date}^close_notesISNOTEMPTY"
        
        if categories:
            cat_filter = "^categoryIN" + ",".join(categories)
            query += cat_filter
        
        params = {
            "sysparm_query": query,
            "sysparm_fields": "sys_id,number,short_description,description,close_notes,category,subcategory,close_code,sys_updated_on",
            "sysparm_limit": max_records,
            "sysparm_display_value": "true"
        }
        
        incidents = self._sn_api_call("incident", params)
        
        result = LearningResult(
            source=LearningSource.SERVICENOW_INCIDENT,
            documents_processed=len(incidents),
            documents_added=0,
            documents_updated=0,
            documents_skipped=0,
            errors=[],
            duration_ms=0
        )
        
        documents_to_add = []
        
        for inc in incidents:
            try:
                resolution = inc.get("close_notes", "")
                
                # Quality check
                if not self._is_quality_resolution(resolution):
                    result.documents_skipped += 1
                    continue
                
                # Check if already exists
                existing = self.vector_store.get_document(f"ticket_{inc['sys_id']}")
                if existing:
                    # Update if newer
                    result.documents_updated += 1
                    continue
                
                # Process into document
                docs = self.doc_processor.process_resolved_ticket(
                    sys_id=inc["sys_id"],
                    number=inc.get("number", ""),
                    short_description=inc.get("short_description", ""),
                    description=inc.get("description", ""),
                    resolution_notes=resolution,
                    category=inc.get("category", ""),
                    subcategory=inc.get("subcategory", ""),
                    close_code=inc.get("close_code", ""),
                    url=f"https://{self.config.servicenow.instance}/incident.do?sys_id={inc['sys_id']}",
                    metadata={"synced_at": datetime.utcnow().isoformat()}
                )
                
                documents_to_add.extend(docs)
                result.documents_added += len(docs)
                
            except Exception as e:
                result.errors.append(f"Error processing incident {inc.get('number', 'unknown')}: {str(e)}")
        
        # Batch add documents
        if documents_to_add:
            try:
                self.vector_store.add_documents(documents_to_add)
            except Exception as e:
                result.errors.append(f"Error adding documents: {str(e)}")
        
        result.duration_ms = (time.time() - start_time) * 1000
        self.total_syncs += 1
        self.last_sync_time = datetime.utcnow()
        
        print(f"[RAG Learning] Synced {result.documents_added} incidents, skipped {result.documents_skipped}")
        return result
    
    def sync_resolved_requests(
        self,
        days_back: Optional[int] = None,
        limit: Optional[int] = None
    ) -> LearningResult:
        """
        Sync resolved service requests from ServiceNow
        
        Args:
            days_back: Number of days to look back
            limit: Maximum requests to sync
            
        Returns:
            LearningResult with sync statistics
        """
        start_time = time.time()
        
        days = days_back or self.config.servicenow.ticket_sync_days
        max_records = limit or 500
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"state=3^ORstate=4^sys_updated_on>={cutoff_date}"  # Closed complete or Closed incomplete
        
        params = {
            "sysparm_query": query,
            "sysparm_fields": "sys_id,number,short_description,description,close_notes,sys_updated_on",
            "sysparm_limit": max_records,
            "sysparm_display_value": "true"
        }
        
        requests_data = self._sn_api_call("sc_request", params)
        
        result = LearningResult(
            source=LearningSource.SERVICENOW_REQUEST,
            documents_processed=len(requests_data),
            documents_added=0,
            documents_updated=0,
            documents_skipped=0,
            errors=[],
            duration_ms=0
        )
        
        documents_to_add = []
        
        for req in requests_data:
            try:
                resolution = req.get("close_notes", "") or req.get("description", "")
                
                if not resolution or len(resolution) < 30:
                    result.documents_skipped += 1
                    continue
                
                existing = self.vector_store.get_document(f"ticket_{req['sys_id']}")
                if existing:
                    result.documents_updated += 1
                    continue
                
                docs = self.doc_processor.process_resolved_ticket(
                    sys_id=req["sys_id"],
                    number=req.get("number", ""),
                    short_description=req.get("short_description", ""),
                    description=req.get("description", ""),
                    resolution_notes=resolution,
                    url=f"https://{self.config.servicenow.instance}/sc_request.do?sys_id={req['sys_id']}",
                    metadata={"synced_at": datetime.utcnow().isoformat(), "type": "request"}
                )
                
                documents_to_add.extend(docs)
                result.documents_added += len(docs)
                
            except Exception as e:
                result.errors.append(f"Error processing request {req.get('number', 'unknown')}: {str(e)}")
        
        if documents_to_add:
            try:
                self.vector_store.add_documents(documents_to_add)
            except Exception as e:
                result.errors.append(f"Error adding documents: {str(e)}")
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    def sync_kb_articles(
        self,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> LearningResult:
        """
        Sync KB articles from ServiceNow
        
        Args:
            limit: Maximum articles to sync
            categories: Filter by KB categories
            
        Returns:
            LearningResult with sync statistics
        """
        start_time = time.time()
        
        max_records = limit or self.config.servicenow.kb_sync_limit
        
        query = "workflow_state=published^active=true"
        if categories:
            cat_filter = "^kb_categoryIN" + ",".join(categories)
            query += cat_filter
        
        params = {
            "sysparm_query": query,
            "sysparm_fields": "sys_id,number,short_description,text,kb_category,sys_updated_on",
            "sysparm_limit": max_records,
            "sysparm_display_value": "true"
        }
        
        articles = self._sn_api_call("kb_knowledge", params)
        
        result = LearningResult(
            source=LearningSource.SERVICENOW_KB,
            documents_processed=len(articles),
            documents_added=0,
            documents_updated=0,
            documents_skipped=0,
            errors=[],
            duration_ms=0
        )
        
        documents_to_add = []
        
        for article in articles:
            try:
                content = article.get("text", "")
                
                if not content or len(content) < 50:
                    result.documents_skipped += 1
                    continue
                
                # Check if already exists
                doc_id = f"kb_{article['sys_id']}"
                existing = self.vector_store.get_document(doc_id)
                if existing:
                    # Could implement update logic here
                    result.documents_updated += 1
                    continue
                
                docs = self.doc_processor.process_kb_article(
                    sys_id=article["sys_id"],
                    title=article.get("short_description", ""),
                    content=content,
                    short_description=article.get("short_description", ""),
                    category=article.get("kb_category", ""),
                    url=f"https://{self.config.servicenow.instance}/kb_view.do?sys_kb_id={article['sys_id']}",
                    metadata={"synced_at": datetime.utcnow().isoformat()}
                )
                
                documents_to_add.extend(docs)
                result.documents_added += len(docs)
                
            except Exception as e:
                result.errors.append(f"Error processing KB {article.get('number', 'unknown')}: {str(e)}")
        
        if documents_to_add:
            try:
                self.vector_store.add_documents(documents_to_add)
            except Exception as e:
                result.errors.append(f"Error adding documents: {str(e)}")
        
        result.duration_ms = (time.time() - start_time) * 1000
        self.total_syncs += 1
        
        print(f"[RAG Learning] Synced {result.documents_added} KB articles")
        return result
    
    def record_feedback(self, feedback: FeedbackEntry) -> bool:
        """
        Record user feedback for learning
        
        Args:
            feedback: The feedback entry
            
        Returns:
            True if recorded successfully
        """
        self.feedback_entries.append(feedback)
        
        # If negative feedback with good detail, could use to improve
        if not feedback.helpful and feedback.comment and len(feedback.comment) > 20:
            # Could implement: 
            # - Flag documents that led to unhelpful responses
            # - Adjust ranking weights
            # - Create improvement suggestions
            print(f"[RAG Learning] Recorded negative feedback: {feedback.comment[:50]}...")
        
        return True
    
    def learn_from_resolution(
        self,
        ticket_number: str,
        problem_description: str,
        resolution_steps: str,
        category: str = "",
        was_helpful: bool = True
    ) -> Optional[str]:
        """
        Learn from a manual resolution
        
        Args:
            ticket_number: The ticket number
            problem_description: What the problem was
            resolution_steps: How it was resolved
            category: Issue category
            was_helpful: Whether the resolution was marked helpful
            
        Returns:
            Document ID if added, None otherwise
        """
        if not self._is_quality_resolution(resolution_steps):
            return None
        
        docs = self.doc_processor.process_resolved_ticket(
            sys_id=f"manual_{ticket_number}_{int(time.time())}",
            number=ticket_number,
            short_description=problem_description[:160],
            description=problem_description,
            resolution_notes=resolution_steps,
            category=category,
            metadata={
                "source": "manual_learning",
                "was_helpful": was_helpful,
                "learned_at": datetime.utcnow().isoformat()
            }
        )
        
        if docs:
            self.vector_store.add_documents(docs)
            return docs[0].id
        
        return None
    
    def add_troubleshooting_guide(
        self,
        title: str,
        symptoms: List[str],
        causes: List[str],
        solutions: List[str],
        category: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a troubleshooting guide to the knowledge base
        
        Args:
            title: Issue title
            symptoms: List of symptoms
            causes: Possible causes
            solutions: Solution steps
            category: Category
            tags: Related tags
            
        Returns:
            Document ID
        """
        guide_id = f"guide_{int(time.time())}"
        
        docs = self.doc_processor.process_troubleshooting_guide(
            guide_id=guide_id,
            title=title,
            symptoms=symptoms,
            causes=causes,
            solutions=solutions,
            category=category,
            tags=tags,
            metadata={"created_at": datetime.utcnow().isoformat()}
        )
        
        self.vector_store.add_documents(docs)
        return docs[0].id if docs else guide_id
    
    def add_faq(
        self,
        question: str,
        answer: str,
        category: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add an FAQ to the knowledge base
        
        Args:
            question: The question
            answer: The answer
            category: FAQ category
            tags: Related tags
            
        Returns:
            Document ID
        """
        faq_id = f"faq_{int(time.time())}"
        
        docs = self.doc_processor.process_faq(
            faq_id=faq_id,
            question=question,
            answer=answer,
            category=category,
            tags=tags
        )
        
        self.vector_store.add_documents(docs)
        return docs[0].id if docs else faq_id
    
    def sync_all(self) -> Dict[str, LearningResult]:
        """
        Run all sync operations
        
        Returns:
            Dictionary of learning results by source
        """
        results = {}
        
        print("[RAG Learning] Starting full sync...")
        
        # Sync KB articles
        results["kb_articles"] = self.sync_kb_articles()
        
        # Sync resolved incidents
        results["incidents"] = self.sync_resolved_incidents()
        
        # Sync resolved requests
        results["requests"] = self.sync_resolved_requests()
        
        # Persist vector store
        self.vector_store.persist()
        
        total_added = sum(r.documents_added for r in results.values())
        print(f"[RAG Learning] Full sync complete. Total documents added: {total_added}")
        
        return results
    
    def stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "total_syncs": self.total_syncs,
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "feedback_count": len(self.feedback_entries),
            "positive_feedback": sum(1 for f in self.feedback_entries if f.helpful),
            "negative_feedback": sum(1 for f in self.feedback_entries if not f.helpful)
        }


# Singleton instance
_knowledge_learner: Optional[KnowledgeLearner] = None

def get_knowledge_learner() -> KnowledgeLearner:
    """Get or create the global knowledge learner"""
    global _knowledge_learner
    if _knowledge_learner is None:
        _knowledge_learner = KnowledgeLearner()
    return _knowledge_learner
