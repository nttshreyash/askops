"""
Document Processor for RAG Engine
Handles document ingestion, chunking, and preprocessing
"""

import re
import uuid
import html
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass

from .vector_store import Document, DocumentType


@dataclass
class ChunkConfig:
    """Configuration for text chunking"""
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size to keep
    respect_sentences: bool = True  # Try to break at sentence boundaries
    respect_paragraphs: bool = True  # Try to break at paragraph boundaries


class TextCleaner:
    """Utilities for cleaning and normalizing text"""
    
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags and decode entities"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def clean_servicenow(text: str) -> str:
        """Clean ServiceNow-specific formatting"""
        if not text:
            return ""
        
        # Remove ServiceNow code blocks
        text = re.sub(r'\[code\].*?\[/code\]', ' [code snippet] ', text, flags=re.DOTALL)
        
        # Remove ServiceNow wiki markup
        text = re.sub(r'\{code[^}]*\}.*?\{code\}', ' [code snippet] ', text, flags=re.DOTALL)
        
        # Clean up lists
        text = re.sub(r'^\s*[\*\-\•]\s*', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        return TextCleaner.clean_html(text)
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for embedding"""
        if not text:
            return ""
        
        # Convert to lowercase for comparison
        # Note: We keep original case for display but normalize for processing
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (usually noise)
        # text = ' '.join(w for w in text.split() if len(w) > 1)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract potential keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction based on capitalized words and technical terms
        words = text.split()
        
        # Common IT keywords to look for
        it_keywords = {
            'vpn', 'wifi', 'network', 'password', 'login', 'email', 'outlook',
            'teams', 'sharepoint', 'onedrive', 'azure', 'active directory',
            'printer', 'laptop', 'desktop', 'mobile', 'phone', 'tablet',
            'software', 'application', 'install', 'update', 'upgrade',
            'error', 'crash', 'slow', 'frozen', 'blue screen', 'bsod',
            'access', 'permission', 'rights', 'admin', 'administrator',
            'certificate', 'ssl', 'tls', 'security', 'firewall', 'proxy',
            'backup', 'restore', 'recovery', 'sync', 'synchronization'
        }
        
        keywords = []
        text_lower = text.lower()
        
        for kw in it_keywords:
            if kw in text_lower:
                keywords.append(kw)
        
        # Add capitalized words that might be product/service names
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.lower() not in {'the', 'this', 'that', 'with', 'from'}:
                keywords.append(word)
        
        return list(set(keywords))[:max_keywords]


class TextChunker:
    """Splits text into overlapping chunks for embedding"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def _find_break_point(self, text: str, target_pos: int) -> int:
        """Find the best position to break text near target_pos"""
        if target_pos >= len(text):
            return len(text)
        
        # Look for paragraph break first
        if self.config.respect_paragraphs:
            para_pos = text.rfind('\n\n', 0, target_pos)
            if para_pos > target_pos - 200:
                return para_pos + 2
        
        # Look for sentence break
        if self.config.respect_sentences:
            # Look backwards for sentence ending
            for i in range(target_pos, max(0, target_pos - 200), -1):
                if i < len(text) and text[i] in '.!?' and (i + 1 >= len(text) or text[i + 1] == ' '):
                    return i + 1
        
        # Look for word break
        space_pos = text.rfind(' ', max(0, target_pos - 50), target_pos)
        if space_pos > 0:
            return space_pos + 1
        
        return target_pos
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text or len(text) <= self.config.chunk_size:
            return [text] if text and len(text) >= self.config.min_chunk_size else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.config.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if len(chunk) >= self.config.min_chunk_size:
                    chunks.append(chunk)
                break
            
            # Find good break point
            end = self._find_break_point(text, end)
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.config.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks


class DocumentProcessor:
    """
    Processes various document types into embedable chunks
    
    Handles:
    - ServiceNow KB articles
    - Resolved tickets
    - Runbooks and procedures
    - FAQs
    - Custom documents
    """
    
    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        self.chunker = TextChunker(chunk_config)
        self.cleaner = TextCleaner()
    
    def process_kb_article(
        self,
        sys_id: str,
        title: str,
        content: str,
        short_description: str = "",
        category: str = "",
        url: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a ServiceNow KB article into documents
        
        Args:
            sys_id: ServiceNow sys_id
            title: Article title
            content: Article body content
            short_description: Article summary
            category: KB category
            url: URL to the article
            metadata: Additional metadata
            
        Returns:
            List of Document objects (may be multiple if chunked)
        """
        # Clean content
        clean_content = self.cleaner.clean_servicenow(content)
        
        # Build full text with context
        full_text = f"Title: {title}\n"
        if short_description:
            full_text += f"Summary: {short_description}\n"
        if category:
            full_text += f"Category: {category}\n"
        full_text += f"\n{clean_content}"
        
        # Extract keywords for metadata
        keywords = self.cleaner.extract_keywords(full_text)
        
        # Chunk if necessary
        chunks = self.chunker.chunk_text(full_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"kb_{sys_id}" if len(chunks) == 1 else f"kb_{sys_id}_chunk_{i}"
            
            doc_metadata = {
                "category": category,
                "keywords": keywords,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            documents.append(Document(
                id=doc_id,
                content=chunk,
                doc_type=DocumentType.KB_ARTICLE,
                title=title,
                source="servicenow",
                source_id=sys_id,
                url=url,
                metadata=doc_metadata
            ))
        
        return documents
    
    def process_resolved_ticket(
        self,
        sys_id: str,
        number: str,
        short_description: str,
        description: str,
        resolution_notes: str,
        category: str = "",
        subcategory: str = "",
        close_code: str = "",
        url: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a resolved ticket for learning
        
        Args:
            sys_id: Ticket sys_id
            number: Ticket number (INC/REQ)
            short_description: Ticket title
            description: Full description
            resolution_notes: How the ticket was resolved
            category: Issue category
            subcategory: Issue subcategory
            close_code: Resolution close code
            url: URL to the ticket
            metadata: Additional metadata
            
        Returns:
            List of Document objects
        """
        # Clean content
        clean_desc = self.cleaner.clean_servicenow(description)
        clean_resolution = self.cleaner.clean_servicenow(resolution_notes)
        
        # Build structured content
        full_text = f"Issue: {short_description}\n"
        if category:
            full_text += f"Category: {category}"
            if subcategory:
                full_text += f" > {subcategory}"
            full_text += "\n"
        full_text += f"\nProblem Description:\n{clean_desc}\n"
        full_text += f"\nResolution:\n{clean_resolution}"
        if close_code:
            full_text += f"\n\nClose Code: {close_code}"
        
        # Extract keywords
        keywords = self.cleaner.extract_keywords(full_text)
        
        # For tickets, we typically don't chunk - we want the full context
        doc_metadata = {
            "ticket_number": number,
            "category": category,
            "subcategory": subcategory,
            "close_code": close_code,
            "keywords": keywords,
            **(metadata or {})
        }
        
        return [Document(
            id=f"ticket_{sys_id}",
            content=full_text,
            doc_type=DocumentType.RESOLVED_TICKET,
            title=short_description,
            source="servicenow",
            source_id=sys_id,
            url=url,
            metadata=doc_metadata
        )]
    
    def process_runbook(
        self,
        runbook_id: str,
        title: str,
        description: str,
        steps: List[str],
        prerequisites: str = "",
        tags: Optional[List[str]] = None,
        url: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a runbook/procedure document
        
        Args:
            runbook_id: Unique runbook identifier
            title: Runbook title
            description: What the runbook does
            steps: List of step descriptions
            prerequisites: Required prerequisites
            tags: Categorization tags
            url: URL to the runbook
            metadata: Additional metadata
            
        Returns:
            List of Document objects
        """
        # Build structured content
        full_text = f"Runbook: {title}\n"
        full_text += f"Description: {description}\n"
        
        if prerequisites:
            full_text += f"\nPrerequisites:\n{prerequisites}\n"
        
        full_text += "\nSteps:\n"
        for i, step in enumerate(steps, 1):
            full_text += f"{i}. {step}\n"
        
        doc_metadata = {
            "tags": tags or [],
            "step_count": len(steps),
            **(metadata or {})
        }
        
        return [Document(
            id=f"runbook_{runbook_id}",
            content=full_text,
            doc_type=DocumentType.RUNBOOK,
            title=title,
            source="manual",
            source_id=runbook_id,
            url=url,
            metadata=doc_metadata
        )]
    
    def process_faq(
        self,
        faq_id: str,
        question: str,
        answer: str,
        category: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process an FAQ entry
        
        Args:
            faq_id: FAQ identifier
            question: The question
            answer: The answer
            category: FAQ category
            tags: Related tags
            metadata: Additional metadata
            
        Returns:
            List of Document objects
        """
        full_text = f"Question: {question}\n\nAnswer: {answer}"
        if category:
            full_text = f"Category: {category}\n\n{full_text}"
        
        doc_metadata = {
            "category": category,
            "tags": tags or [],
            "question": question,
            **(metadata or {})
        }
        
        return [Document(
            id=f"faq_{faq_id}",
            content=full_text,
            doc_type=DocumentType.FAQ,
            title=question[:100],
            source="manual",
            source_id=faq_id,
            metadata=doc_metadata
        )]
    
    def process_troubleshooting_guide(
        self,
        guide_id: str,
        title: str,
        symptoms: List[str],
        causes: List[str],
        solutions: List[str],
        category: str = "",
        tags: Optional[List[str]] = None,
        url: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a troubleshooting guide
        
        Args:
            guide_id: Guide identifier
            title: Issue title
            symptoms: List of symptoms
            causes: Possible causes
            solutions: Solution steps
            category: Category
            tags: Related tags
            url: URL to guide
            metadata: Additional metadata
            
        Returns:
            List of Document objects
        """
        full_text = f"Issue: {title}\n"
        if category:
            full_text += f"Category: {category}\n"
        
        full_text += "\nSymptoms:\n"
        for symptom in symptoms:
            full_text += f"• {symptom}\n"
        
        full_text += "\nPossible Causes:\n"
        for cause in causes:
            full_text += f"• {cause}\n"
        
        full_text += "\nSolutions:\n"
        for i, solution in enumerate(solutions, 1):
            full_text += f"{i}. {solution}\n"
        
        doc_metadata = {
            "category": category,
            "tags": tags or [],
            "symptom_count": len(symptoms),
            "solution_count": len(solutions),
            **(metadata or {})
        }
        
        return [Document(
            id=f"troubleshoot_{guide_id}",
            content=full_text,
            doc_type=DocumentType.TROUBLESHOOTING_GUIDE,
            title=title,
            source="manual",
            source_id=guide_id,
            url=url,
            metadata=doc_metadata
        )]
    
    def process_raw_text(
        self,
        text: str,
        title: str = "",
        doc_type: DocumentType = DocumentType.CUSTOM,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process raw text into documents
        
        Args:
            text: The raw text content
            title: Document title
            doc_type: Type of document
            source: Source system
            metadata: Additional metadata
            
        Returns:
            List of Document objects (chunked if necessary)
        """
        clean_text = self.cleaner.normalize_text(text)
        chunks = self.chunker.chunk_text(clean_text)
        
        documents = []
        base_id = str(uuid.uuid4())[:8]
        
        for i, chunk in enumerate(chunks):
            doc_id = f"doc_{base_id}" if len(chunks) == 1 else f"doc_{base_id}_chunk_{i}"
            
            doc_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            documents.append(Document(
                id=doc_id,
                content=chunk,
                doc_type=doc_type,
                title=title or f"Document {base_id}",
                source=source,
                metadata=doc_metadata
            ))
        
        return documents


# Singleton instance
_document_processor: Optional[DocumentProcessor] = None

def get_document_processor() -> DocumentProcessor:
    """Get or create the global document processor"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
