"""
Learning Engine
===============
Learns from ticket resolutions to continuously improve auto-resolution
and prescriptive recommendations.

This engine:
1. Records successful resolutions
2. Updates knowledge base with new solutions
3. Improves resolution pattern matching
4. Tracks resolution effectiveness
5. Identifies knowledge gaps
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResolutionOutcome(str, Enum):
    """Outcome of a resolution"""
    SUCCESSFUL = "successful"        # Fixed the issue
    PARTIAL = "partial"              # Partially fixed
    FAILED = "failed"                # Did not fix
    WORKAROUND = "workaround"        # Temporary fix applied
    ESCALATED = "escalated"          # Had to escalate
    USER_RESOLVED = "user_resolved"  # User fixed it themselves


class FeedbackType(str, Enum):
    """Types of feedback"""
    AGENT_FEEDBACK = "agent_feedback"      # Agent rated the suggestion
    USER_FEEDBACK = "user_feedback"        # End user feedback
    RESOLUTION_VERIFIED = "resolution_verified"  # System verified fix
    AUTO_DETECTED = "auto_detected"        # System detected outcome


@dataclass
class ResolutionRecord:
    """Record of a ticket resolution"""
    record_id: str
    ticket_id: str
    issue_description: str
    issue_category: str
    resolution_description: str
    resolution_steps: List[str]
    outcome: ResolutionOutcome
    time_to_resolve_minutes: int
    auto_resolved: bool
    runbook_used: Optional[str] = None
    agent_id: Optional[str] = None
    feedback_score: float = 0.0  # 0-5 rating
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGap:
    """Identified gap in knowledge base"""
    gap_id: str
    issue_pattern: str
    frequency: int  # How often this pattern appears without solution
    sample_tickets: List[str]
    suggested_category: str
    recommended_action: str
    identified_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningMetrics:
    """Metrics from the learning engine"""
    total_resolutions_learned: int
    successful_resolutions: int
    average_feedback_score: float
    knowledge_gaps_identified: int
    auto_resolution_improvement: float  # Percentage improvement
    most_learned_categories: List[Dict[str, Any]]
    least_successful_patterns: List[Dict[str, Any]]


class LearningEngine:
    """
    Continuous Learning Engine
    
    Learns from every ticket resolution to improve future
    auto-resolution and agent recommendations.
    """
    
    def __init__(self, rag_engine=None, vector_store=None):
        """
        Initialize the learning engine.
        
        Args:
            rag_engine: RAG engine for knowledge retrieval and storage
            vector_store: Vector store for embedding new knowledge
        """
        self.rag_engine = rag_engine
        self.vector_store = vector_store
        
        # Resolution history
        self.resolution_records: List[ResolutionRecord] = []
        
        # Pattern success tracking
        self.pattern_success: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "avg_time": 0,
            "feedback_scores": []
        })
        
        # Knowledge gaps
        self.knowledge_gaps: Dict[str, KnowledgeGap] = {}
        
        # Feedback queue for batch processing
        self.feedback_queue: List[Dict[str, Any]] = []
        
        # Learning statistics
        self.stats = {
            "total_learned": 0,
            "successful_learned": 0,
            "knowledge_added": 0,
            "patterns_improved": 0
        }
    
    async def record_resolution(
        self,
        ticket_id: str,
        issue_description: str,
        issue_category: str,
        resolution_description: str,
        resolution_steps: List[str] = None,
        outcome: ResolutionOutcome = ResolutionOutcome.SUCCESSFUL,
        time_to_resolve: int = 0,
        auto_resolved: bool = False,
        runbook_used: str = None,
        agent_id: str = None,
        feedback_score: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> ResolutionRecord:
        """
        Record a ticket resolution for learning.
        
        Args:
            ticket_id: The resolved ticket ID
            issue_description: Original issue description
            issue_category: Issue category
            resolution_description: How it was resolved
            resolution_steps: Steps taken to resolve
            outcome: Resolution outcome
            time_to_resolve: Time in minutes
            auto_resolved: Whether auto-resolved
            runbook_used: Runbook ID if used
            agent_id: Agent who resolved
            feedback_score: User/agent rating
            metadata: Additional metadata
            
        Returns:
            ResolutionRecord created
        """
        record_id = f"res_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticket_id}"
        
        record = ResolutionRecord(
            record_id=record_id,
            ticket_id=ticket_id,
            issue_description=issue_description,
            issue_category=issue_category,
            resolution_description=resolution_description,
            resolution_steps=resolution_steps or [],
            outcome=outcome,
            time_to_resolve_minutes=time_to_resolve,
            auto_resolved=auto_resolved,
            runbook_used=runbook_used,
            agent_id=agent_id,
            feedback_score=feedback_score,
            metadata=metadata or {}
        )
        
        self.resolution_records.append(record)
        self.stats["total_learned"] += 1
        
        # Update pattern tracking
        pattern_key = self._extract_pattern_key(issue_description, issue_category)
        self._update_pattern_stats(pattern_key, record)
        
        # If successful, potentially add to knowledge base
        if outcome in [ResolutionOutcome.SUCCESSFUL, ResolutionOutcome.WORKAROUND]:
            self.stats["successful_learned"] += 1
            await self._consider_knowledge_addition(record)
        
        # Check for knowledge gaps
        if outcome in [ResolutionOutcome.FAILED, ResolutionOutcome.ESCALATED]:
            self._track_knowledge_gap(record)
        
        logger.info(f"[LearningEngine] Recorded resolution: {record_id} (outcome: {outcome.value})")
        
        return record
    
    async def add_feedback(
        self,
        ticket_id: str,
        feedback_type: FeedbackType,
        score: float,
        comments: str = None,
        agent_id: str = None
    ):
        """
        Add feedback for a resolution.
        
        Args:
            ticket_id: Ticket ID
            feedback_type: Type of feedback
            score: Rating (0-5)
            comments: Optional comments
            agent_id: Agent providing feedback
        """
        feedback = {
            "ticket_id": ticket_id,
            "feedback_type": feedback_type.value,
            "score": min(max(score, 0), 5),  # Clamp to 0-5
            "comments": comments,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_queue.append(feedback)
        
        # Update existing record if found
        for record in self.resolution_records:
            if record.ticket_id == ticket_id:
                # Average with existing score
                if record.feedback_score > 0:
                    record.feedback_score = (record.feedback_score + score) / 2
                else:
                    record.feedback_score = score
                record.verified = True
                break
        
        # Process feedback queue if it gets large
        if len(self.feedback_queue) >= 10:
            await self._process_feedback_queue()
    
    def _extract_pattern_key(self, description: str, category: str) -> str:
        """Extract a pattern key from issue description"""
        # Normalize and extract key terms
        desc_lower = description.lower()
        
        # Key patterns to look for
        pattern_keywords = {
            "password": ["password", "reset", "forgot", "expired"],
            "vpn": ["vpn", "remote", "connection"],
            "email": ["email", "outlook", "mailbox"],
            "network": ["network", "internet", "wifi", "slow"],
            "access": ["access", "permission", "denied", "can't access"],
            "software": ["install", "software", "application", "crash"],
            "printer": ["printer", "print", "printing"]
        }
        
        for pattern, keywords in pattern_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return f"{category}_{pattern}"
        
        return f"{category}_general"
    
    def _update_pattern_stats(self, pattern_key: str, record: ResolutionRecord):
        """Update statistics for a pattern"""
        stats = self.pattern_success[pattern_key]
        stats["attempts"] += 1
        
        if record.outcome == ResolutionOutcome.SUCCESSFUL:
            stats["successes"] += 1
        
        # Update average time
        if record.time_to_resolve_minutes > 0:
            total_time = stats["avg_time"] * (stats["attempts"] - 1) + record.time_to_resolve_minutes
            stats["avg_time"] = total_time / stats["attempts"]
        
        if record.feedback_score > 0:
            stats["feedback_scores"].append(record.feedback_score)
    
    async def _consider_knowledge_addition(self, record: ResolutionRecord):
        """Consider adding this resolution to knowledge base"""
        # Criteria for adding to KB:
        # 1. Has clear resolution steps
        # 2. Good feedback score (if available)
        # 3. Not a trivial issue
        
        should_add = True
        
        # Check resolution quality
        if not record.resolution_description or len(record.resolution_description) < 50:
            should_add = False
        
        if not record.resolution_steps or len(record.resolution_steps) < 2:
            should_add = False
        
        # Check feedback if available
        if record.feedback_score > 0 and record.feedback_score < 3:
            should_add = False
        
        # Check if similar knowledge already exists
        if self.rag_engine and should_add:
            try:
                result = self.rag_engine.retrieve(
                    query=record.issue_description,
                    top_k=1,
                    min_score=0.9
                )
                if result.search_results:
                    # Very similar knowledge already exists
                    should_add = False
            except Exception:
                pass
        
        if should_add:
            await self._add_to_knowledge_base(record)
    
    async def _add_to_knowledge_base(self, record: ResolutionRecord):
        """Add resolution to knowledge base"""
        if not self.rag_engine:
            logger.warning("[LearningEngine] No RAG engine available for knowledge addition")
            return
        
        # Format for KB
        kb_document = {
            "title": f"Resolution: {record.issue_category} - {record.issue_description[:100]}",
            "content": self._format_resolution_content(record),
            "category": record.issue_category,
            "resolution": record.resolution_description,
            "resolution_steps": record.resolution_steps,
            "source": "learned_resolution",
            "ticket_id": record.ticket_id,
            "learned_at": datetime.now().isoformat(),
            "success_rate": self._get_pattern_success_rate(record.issue_description, record.issue_category),
            "avg_resolution_time": record.time_to_resolve_minutes
        }
        
        try:
            # Add to RAG engine
            if hasattr(self.rag_engine, 'add_document'):
                await self.rag_engine.add_document(
                    content=kb_document["content"],
                    metadata=kb_document,
                    doc_type="learned_resolution"
                )
                self.stats["knowledge_added"] += 1
                logger.info(f"[LearningEngine] Added resolution to KB: {record.ticket_id}")
        except Exception as e:
            logger.error(f"[LearningEngine] Failed to add to KB: {e}")
    
    def _format_resolution_content(self, record: ResolutionRecord) -> str:
        """Format resolution record for KB storage"""
        content = f"""Issue: {record.issue_description}

Category: {record.issue_category}

Resolution: {record.resolution_description}

Steps Taken:
"""
        for i, step in enumerate(record.resolution_steps, 1):
            content += f"{i}. {step}\n"
        
        if record.runbook_used:
            content += f"\nRunbook Used: {record.runbook_used}"
        
        content += f"\nResolution Time: {record.time_to_resolve_minutes} minutes"
        
        if record.auto_resolved:
            content += "\nNote: This issue was auto-resolved without human intervention."
        
        return content
    
    def _get_pattern_success_rate(self, description: str, category: str) -> float:
        """Get success rate for a pattern"""
        pattern_key = self._extract_pattern_key(description, category)
        stats = self.pattern_success.get(pattern_key)
        
        if stats and stats["attempts"] > 0:
            return stats["successes"] / stats["attempts"]
        return 0.0
    
    def _track_knowledge_gap(self, record: ResolutionRecord):
        """Track potential knowledge gap"""
        pattern_key = self._extract_pattern_key(record.issue_description, record.issue_category)
        
        if pattern_key in self.knowledge_gaps:
            gap = self.knowledge_gaps[pattern_key]
            gap.frequency += 1
            gap.sample_tickets.append(record.ticket_id)
            gap.sample_tickets = gap.sample_tickets[-10:]  # Keep last 10
        else:
            self.knowledge_gaps[pattern_key] = KnowledgeGap(
                gap_id=f"gap_{len(self.knowledge_gaps) + 1}",
                issue_pattern=pattern_key,
                frequency=1,
                sample_tickets=[record.ticket_id],
                suggested_category=record.issue_category,
                recommended_action="Create KB article or runbook for this pattern"
            )
    
    async def _process_feedback_queue(self):
        """Process queued feedback"""
        # Group feedback by ticket
        feedback_by_ticket: Dict[str, List[Dict]] = defaultdict(list)
        for fb in self.feedback_queue:
            feedback_by_ticket[fb["ticket_id"]].append(fb)
        
        # Process each ticket's feedback
        for ticket_id, feedbacks in feedback_by_ticket.items():
            avg_score = sum(f["score"] for f in feedbacks) / len(feedbacks)
            
            # Update resolution record
            for record in self.resolution_records:
                if record.ticket_id == ticket_id:
                    record.feedback_score = avg_score
                    record.verified = True
                    
                    # If consistently negative feedback, mark for review
                    if avg_score < 2:
                        logger.warning(f"[LearningEngine] Low feedback for {ticket_id}: {avg_score}")
        
        self.feedback_queue.clear()
    
    def get_learning_recommendations(self, category: str = None) -> List[Dict[str, Any]]:
        """Get recommendations for improving knowledge base"""
        recommendations = []
        
        # Find patterns with low success rates
        for pattern_key, stats in self.pattern_success.items():
            if stats["attempts"] >= 5:  # Minimum sample size
                success_rate = stats["successes"] / stats["attempts"]
                if success_rate < 0.7:
                    recommendations.append({
                        "type": "improve_pattern",
                        "pattern": pattern_key,
                        "success_rate": f"{success_rate * 100:.1f}%",
                        "recommendation": f"Review and improve resolution process for '{pattern_key}'"
                    })
        
        # Knowledge gaps
        for gap_id, gap in self.knowledge_gaps.items():
            if gap.frequency >= 3:
                recommendations.append({
                    "type": "knowledge_gap",
                    "pattern": gap.issue_pattern,
                    "frequency": gap.frequency,
                    "recommendation": gap.recommended_action,
                    "sample_tickets": gap.sample_tickets[:5]
                })
        
        # Runbook recommendations
        high_volume_patterns = [
            (k, v) for k, v in self.pattern_success.items()
            if v["attempts"] >= 10 and v["successes"] / v["attempts"] >= 0.8
        ]
        
        for pattern_key, stats in high_volume_patterns:
            # Check if runbook exists for this pattern
            has_runbook = any(
                r.runbook_used for r in self.resolution_records
                if self._extract_pattern_key(r.issue_description, r.issue_category) == pattern_key
            )
            
            if not has_runbook:
                recommendations.append({
                    "type": "create_runbook",
                    "pattern": pattern_key,
                    "volume": stats["attempts"],
                    "recommendation": f"Consider creating automated runbook for high-volume pattern '{pattern_key}'"
                })
        
        return recommendations
    
    def get_metrics(self) -> LearningMetrics:
        """Get learning engine metrics"""
        successful = sum(1 for r in self.resolution_records if r.outcome == ResolutionOutcome.SUCCESSFUL)
        
        avg_feedback = 0
        feedback_count = 0
        for r in self.resolution_records:
            if r.feedback_score > 0:
                avg_feedback += r.feedback_score
                feedback_count += 1
        
        if feedback_count > 0:
            avg_feedback = avg_feedback / feedback_count
        
        # Category breakdown
        category_counts: Dict[str, int] = defaultdict(int)
        for r in self.resolution_records:
            category_counts[r.issue_category] += 1
        
        most_learned = [
            {"category": cat, "count": count}
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Least successful patterns
        least_successful = []
        for pattern_key, stats in self.pattern_success.items():
            if stats["attempts"] >= 3:
                success_rate = stats["successes"] / stats["attempts"]
                if success_rate < 0.7:
                    least_successful.append({
                        "pattern": pattern_key,
                        "success_rate": f"{success_rate * 100:.1f}%",
                        "attempts": stats["attempts"]
                    })
        
        least_successful.sort(key=lambda x: float(x["success_rate"].rstrip('%')))
        
        # Calculate improvement
        # Compare recent success rate vs older
        recent_records = [r for r in self.resolution_records if r.created_at > datetime.now() - timedelta(days=7)]
        older_records = [r for r in self.resolution_records if r.created_at <= datetime.now() - timedelta(days=7)]
        
        improvement = 0.0
        if older_records and recent_records:
            recent_success = sum(1 for r in recent_records if r.outcome == ResolutionOutcome.SUCCESSFUL) / len(recent_records)
            older_success = sum(1 for r in older_records if r.outcome == ResolutionOutcome.SUCCESSFUL) / len(older_records)
            if older_success > 0:
                improvement = ((recent_success - older_success) / older_success) * 100
        
        return LearningMetrics(
            total_resolutions_learned=len(self.resolution_records),
            successful_resolutions=successful,
            average_feedback_score=round(avg_feedback, 2),
            knowledge_gaps_identified=len(self.knowledge_gaps),
            auto_resolution_improvement=round(improvement, 1),
            most_learned_categories=most_learned,
            least_successful_patterns=least_successful[:5]
        )
    
    def get_pattern_insights(self, pattern_key: str = None) -> Dict[str, Any]:
        """Get insights for a specific pattern or all patterns"""
        if pattern_key:
            stats = self.pattern_success.get(pattern_key)
            if stats:
                return {
                    "pattern": pattern_key,
                    "total_attempts": stats["attempts"],
                    "success_rate": f"{(stats['successes'] / stats['attempts'] * 100):.1f}%" if stats["attempts"] > 0 else "N/A",
                    "avg_resolution_time": f"{stats['avg_time']:.1f} minutes",
                    "avg_feedback": sum(stats["feedback_scores"]) / len(stats["feedback_scores"]) if stats["feedback_scores"] else "N/A"
                }
            return None
        
        # Return all patterns
        return {
            key: {
                "total_attempts": stats["attempts"],
                "success_rate": f"{(stats['successes'] / stats['attempts'] * 100):.1f}%" if stats["attempts"] > 0 else "N/A",
                "avg_resolution_time": f"{stats['avg_time']:.1f} minutes"
            }
            for key, stats in self.pattern_success.items()
            if stats["attempts"] > 0
        }


# Singleton instance
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine(rag_engine=None, vector_store=None) -> LearningEngine:
    """Get or create the learning engine singleton."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine(rag_engine, vector_store)
    return _learning_engine
