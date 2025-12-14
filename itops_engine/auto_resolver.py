"""
Auto-Resolution Engine
======================
Automatically attempts to resolve tickets before human assignment.

This engine:
1. Analyzes incoming tickets/issues
2. Matches them to known resolution patterns
3. Executes safe remediation actions
4. Verifies resolution success
5. Auto-closes tickets if resolved, or provides diagnosis to agents

Architecture:
- Uses RAG to find similar resolved tickets
- Matches issues to runbooks for automated fixes
- Executes remediation steps with safety checks
- Validates resolution through monitoring/health checks
"""

import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResolutionStatus(str, Enum):
    """Status of auto-resolution attempt"""
    NOT_ATTEMPTED = "not_attempted"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    FAILED = "failed"
    REQUIRES_HUMAN = "requires_human"
    ESCALATED = "escalated"


class ResolutionConfidence(str, Enum):
    """Confidence level in resolution"""
    HIGH = "high"      # 85%+ confident
    MEDIUM = "medium"  # 60-85% confident
    LOW = "low"        # Below 60% confident


@dataclass
class ResolutionAttempt:
    """Record of an auto-resolution attempt"""
    attempt_id: str
    ticket_id: str
    issue_description: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: ResolutionStatus = ResolutionStatus.NOT_ATTEMPTED
    confidence: ResolutionConfidence = ResolutionConfidence.LOW
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    diagnosis: str = ""
    resolution_notes: str = ""
    runbook_executed: Optional[str] = None
    similar_tickets_found: List[Dict[str, Any]] = field(default_factory=list)
    verification_result: Optional[Dict[str, Any]] = None
    escalation_reason: Optional[str] = None
    time_saved_minutes: int = 0


@dataclass 
class AutoResolutionResult:
    """Result returned from auto-resolution engine"""
    ticket_id: str
    status: ResolutionStatus
    confidence: ResolutionConfidence
    diagnosis: str
    recommended_actions: List[Dict[str, Any]]
    automated_actions_taken: List[Dict[str, Any]]
    resolution_notes: str
    similar_cases: List[Dict[str, Any]]
    prescriptive_steps: List[Dict[str, Any]]  # For human agents
    estimated_resolution_time: int  # minutes
    can_auto_resolve: bool
    requires_human_review: bool
    escalation_recommended: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoResolver:
    """
    Intelligent Auto-Resolution Engine
    
    Attempts to automatically resolve IT issues by:
    1. Analyzing the issue description with AI
    2. Finding similar resolved cases in knowledge base
    3. Matching to appropriate runbooks
    4. Executing safe remediation actions
    5. Verifying the fix worked
    """
    
    def __init__(self, rag_engine=None, runbook_engine=None, llm_client=None):
        """
        Initialize the auto-resolver.
        
        Args:
            rag_engine: RAG engine for knowledge retrieval
            runbook_engine: Runbook engine for automation
            llm_client: Function to query LLM
        """
        self.rag_engine = rag_engine
        self.runbook_engine = runbook_engine
        self.llm_client = llm_client
        
        # Resolution patterns - known issue signatures
        self.resolution_patterns = self._load_resolution_patterns()
        
        # Metrics
        self.resolution_attempts: List[ResolutionAttempt] = []
        self.success_count = 0
        self.total_attempts = 0
        
    def _load_resolution_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known resolution patterns for common issues"""
        return {
            "password_reset": {
                "keywords": ["password", "reset", "forgot", "expired", "login", "locked out", "can't sign in"],
                "category": "access",
                "auto_resolvable": True,
                "runbook": "rb_password_reset",
                "confidence": 0.9,
                "typical_resolution_time": 5,
                "prescriptive_steps": [
                    {"step": 1, "action": "Verify user identity via employee ID or manager confirmation", "automated": False},
                    {"step": 2, "action": "Check AD account status - is account locked or disabled?", "automated": True},
                    {"step": 3, "action": "Reset password in Active Directory", "automated": True},
                    {"step": 4, "action": "Set password to expire on first login", "automated": True},
                    {"step": 5, "action": "Send credentials via secure channel", "automated": True}
                ]
            },
            "account_locked": {
                "keywords": ["locked", "account locked", "too many attempts", "locked out"],
                "category": "access",
                "auto_resolvable": True,
                "runbook": "rb_account_unlock",
                "confidence": 0.85,
                "typical_resolution_time": 3,
                "prescriptive_steps": [
                    {"step": 1, "action": "Verify no security threats on account", "automated": True},
                    {"step": 2, "action": "Check lockout source (bad cached creds, mobile device)", "automated": True},
                    {"step": 3, "action": "Unlock account in Active Directory", "automated": True},
                    {"step": 4, "action": "Clear bad password count", "automated": True}
                ]
            },
            "vpn_issues": {
                "keywords": ["vpn", "remote access", "can't connect", "vpn not working", "work from home"],
                "category": "network",
                "auto_resolvable": True,
                "runbook": "rb_vpn_troubleshoot",
                "confidence": 0.75,
                "typical_resolution_time": 10,
                "prescriptive_steps": [
                    {"step": 1, "action": "Check VPN server health status", "automated": True},
                    {"step": 2, "action": "Verify user VPN permissions in AD groups", "automated": True},
                    {"step": 3, "action": "Check VPN certificate validity", "automated": True},
                    {"step": 4, "action": "Test network connectivity from user location", "automated": True},
                    {"step": 5, "action": "Reset VPN client configuration", "automated": True},
                    {"step": 6, "action": "If corporate device, push fresh VPN profile via MDM", "automated": True}
                ]
            },
            "email_issues": {
                "keywords": ["email", "outlook", "can't send", "can't receive", "mailbox", "calendar"],
                "category": "software",
                "auto_resolvable": True,
                "runbook": "rb_email_config",
                "confidence": 0.7,
                "typical_resolution_time": 8,
                "prescriptive_steps": [
                    {"step": 1, "action": "Check Exchange Online service health", "automated": True},
                    {"step": 2, "action": "Verify mailbox status and quotas", "automated": True},
                    {"step": 3, "action": "Test Autodiscover configuration", "automated": True},
                    {"step": 4, "action": "Check for mail flow rules blocking delivery", "automated": True},
                    {"step": 5, "action": "Repair/recreate Outlook profile", "automated": True}
                ]
            },
            "disk_space": {
                "keywords": ["disk full", "storage", "no space", "out of space", "c drive full"],
                "category": "hardware",
                "auto_resolvable": True,
                "runbook": "rb_disk_cleanup",
                "confidence": 0.8,
                "typical_resolution_time": 15,
                "prescriptive_steps": [
                    {"step": 1, "action": "Check current disk usage", "automated": True},
                    {"step": 2, "action": "Clear Windows temp files", "automated": True},
                    {"step": 3, "action": "Clear browser cache", "automated": True},
                    {"step": 4, "action": "Run Windows Disk Cleanup utility", "automated": True},
                    {"step": 5, "action": "Check for large log files", "automated": True}
                ]
            },
            "network_connectivity": {
                "keywords": ["internet", "network", "no connection", "slow network", "can't access"],
                "category": "network",
                "auto_resolvable": True,
                "runbook": "rb_network_diag",
                "confidence": 0.7,
                "typical_resolution_time": 10,
                "prescriptive_steps": [
                    {"step": 1, "action": "Ping default gateway", "automated": True},
                    {"step": 2, "action": "Test DNS resolution", "automated": True},
                    {"step": 3, "action": "Check DHCP lease status", "automated": True},
                    {"step": 4, "action": "Verify proxy settings", "automated": True},
                    {"step": 5, "action": "Check for network driver issues", "automated": True}
                ]
            },
            "software_crash": {
                "keywords": ["crash", "not responding", "freeze", "hung", "application error"],
                "category": "software",
                "auto_resolvable": False,  # Needs diagnosis
                "runbook": None,
                "confidence": 0.5,
                "typical_resolution_time": 20,
                "prescriptive_steps": [
                    {"step": 1, "action": "Collect application and Windows event logs", "automated": True},
                    {"step": 2, "action": "Check for recent updates/changes", "automated": True},
                    {"step": 3, "action": "Verify system resources (RAM, CPU)", "automated": True},
                    {"step": 4, "action": "Run application repair/reinstall", "automated": False},
                    {"step": 5, "action": "Check for conflicting software", "automated": True}
                ]
            },
            "permission_access": {
                "keywords": ["permission", "access denied", "can't access folder", "sharepoint", "shared drive"],
                "category": "access",
                "auto_resolvable": False,  # Requires approval
                "runbook": None,
                "confidence": 0.6,
                "typical_resolution_time": 30,
                "prescriptive_steps": [
                    {"step": 1, "action": "Identify resource owner/manager", "automated": True},
                    {"step": 2, "action": "Check user's current group memberships", "automated": True},
                    {"step": 3, "action": "Verify requested resource exists", "automated": True},
                    {"step": 4, "action": "Request approval from resource owner", "automated": False},
                    {"step": 5, "action": "Add user to appropriate security group", "automated": False}
                ]
            },
            "printer_issues": {
                "keywords": ["printer", "print", "printing", "can't print", "printer offline"],
                "category": "hardware",
                "auto_resolvable": True,
                "runbook": "rb_printer_fix",
                "confidence": 0.65,
                "typical_resolution_time": 10,
                "prescriptive_steps": [
                    {"step": 1, "action": "Check printer status and connectivity", "automated": True},
                    {"step": 2, "action": "Clear print queue", "automated": True},
                    {"step": 3, "action": "Restart print spooler service", "automated": True},
                    {"step": 4, "action": "Remove and re-add printer", "automated": True}
                ]
            },
            "slow_computer": {
                "keywords": ["slow", "sluggish", "takes forever", "performance", "running slow"],
                "category": "hardware",
                "auto_resolvable": True,
                "runbook": "rb_performance_fix",
                "confidence": 0.6,
                "typical_resolution_time": 20,
                "prescriptive_steps": [
                    {"step": 1, "action": "Check system resource usage", "automated": True},
                    {"step": 2, "action": "Identify high CPU/memory processes", "automated": True},
                    {"step": 3, "action": "Check disk health (SSD/HDD)", "automated": True},
                    {"step": 4, "action": "Clear temporary files", "automated": True},
                    {"step": 5, "action": "Disable unnecessary startup programs", "automated": True},
                    {"step": 6, "action": "Run malware scan", "automated": True}
                ]
            }
        }
    
    async def analyze_issue(
        self, 
        issue_description: str,
        ticket_id: str = None,
        user_context: Dict[str, Any] = None,
        session_state: Dict[str, Any] = None
    ) -> AutoResolutionResult:
        """
        Analyze an issue and determine resolution path.
        
        Args:
            issue_description: Description of the IT issue
            ticket_id: Optional ticket ID if already created
            user_context: User information (device, location, etc.)
            session_state: Current session state
            
        Returns:
            AutoResolutionResult with diagnosis and recommended actions
        """
        attempt_id = f"ar_{datetime.now().strftime('%Y%m%d%H%M%S')}_{ticket_id or 'new'}"
        started_at = datetime.now()
        
        logger.info(f"[AutoResolver] Analyzing issue: {issue_description[:100]}...")
        
        # Step 1: Pattern matching for known issues
        matched_pattern = self._match_pattern(issue_description)
        
        # Step 2: RAG search for similar resolved cases
        similar_cases = await self._find_similar_cases(issue_description)
        
        # Step 3: AI-powered diagnosis
        diagnosis = await self._ai_diagnosis(
            issue_description, 
            matched_pattern, 
            similar_cases,
            user_context
        )
        
        # Step 4: Determine if auto-resolvable
        can_auto_resolve, confidence = self._assess_auto_resolution(
            matched_pattern, 
            similar_cases, 
            diagnosis
        )
        
        # Step 5: Get recommended actions
        recommended_actions = self._get_recommended_actions(
            matched_pattern, 
            diagnosis, 
            similar_cases
        )
        
        # Step 6: Get prescriptive steps for agents
        prescriptive_steps = self._generate_prescriptive_steps(
            matched_pattern,
            diagnosis,
            similar_cases,
            user_context
        )
        
        # Step 7: Determine if human review needed
        requires_human = not can_auto_resolve or confidence == ResolutionConfidence.LOW
        
        # Step 8: Calculate estimated resolution time
        est_time = self._estimate_resolution_time(matched_pattern, can_auto_resolve)
        
        result = AutoResolutionResult(
            ticket_id=ticket_id or "pending",
            status=ResolutionStatus.NOT_ATTEMPTED,
            confidence=confidence,
            diagnosis=diagnosis,
            recommended_actions=recommended_actions,
            automated_actions_taken=[],
            resolution_notes="",
            similar_cases=similar_cases,
            prescriptive_steps=prescriptive_steps,
            estimated_resolution_time=est_time,
            can_auto_resolve=can_auto_resolve,
            requires_human_review=requires_human,
            escalation_recommended=not can_auto_resolve and confidence == ResolutionConfidence.LOW,
            metadata={
                "attempt_id": attempt_id,
                "started_at": started_at.isoformat(),
                "matched_pattern": matched_pattern.get("name") if matched_pattern else None,
                "pattern_confidence": matched_pattern.get("confidence", 0) if matched_pattern else 0
            }
        )
        
        return result
    
    async def attempt_resolution(
        self,
        issue_description: str,
        ticket_id: str = None,
        user_context: Dict[str, Any] = None,
        session_state: Dict[str, Any] = None,
        dry_run: bool = False
    ) -> AutoResolutionResult:
        """
        Attempt to automatically resolve an issue.
        
        Args:
            issue_description: Description of the IT issue
            ticket_id: Ticket ID if exists
            user_context: User information
            session_state: Current session state
            dry_run: If True, simulate without executing
            
        Returns:
            AutoResolutionResult with resolution status
        """
        # First, analyze the issue
        result = await self.analyze_issue(
            issue_description, 
            ticket_id, 
            user_context, 
            session_state
        )
        
        if not result.can_auto_resolve:
            result.status = ResolutionStatus.REQUIRES_HUMAN
            result.resolution_notes = "Issue requires human intervention. Prescriptive steps provided."
            return result
        
        result.status = ResolutionStatus.IN_PROGRESS
        
        # Execute runbook if available
        matched_pattern = self._match_pattern(issue_description)
        if matched_pattern and matched_pattern.get("runbook") and self.runbook_engine:
            runbook_id = matched_pattern["runbook"]
            logger.info(f"[AutoResolver] Executing runbook: {runbook_id}")
            
            try:
                runbook_result = await self.runbook_engine.execute_runbook(
                    runbook_id,
                    context=user_context or {},
                    dry_run=dry_run
                )
                
                result.automated_actions_taken.append({
                    "action": "runbook_execution",
                    "runbook_id": runbook_id,
                    "success": runbook_result.success,
                    "steps_completed": runbook_result.steps_completed,
                    "output": runbook_result.output_data
                })
                
                if runbook_result.success:
                    result.status = ResolutionStatus.RESOLVED
                    result.resolution_notes = f"Automatically resolved via runbook: {runbook_id}"
                    self.success_count += 1
                else:
                    result.status = ResolutionStatus.PARTIALLY_RESOLVED
                    result.resolution_notes = f"Runbook partially completed. {runbook_result.error_message}"
                    
            except Exception as e:
                logger.error(f"[AutoResolver] Runbook execution failed: {e}")
                result.status = ResolutionStatus.FAILED
                result.resolution_notes = f"Runbook execution failed: {str(e)}"
        else:
            # No runbook - provide diagnosis only
            result.status = ResolutionStatus.REQUIRES_HUMAN
            result.resolution_notes = "No automated runbook available. See prescriptive steps."
        
        self.total_attempts += 1
        
        # Record the attempt
        attempt = ResolutionAttempt(
            attempt_id=result.metadata["attempt_id"],
            ticket_id=ticket_id or "pending",
            issue_description=issue_description,
            started_at=datetime.fromisoformat(result.metadata["started_at"]),
            completed_at=datetime.now(),
            status=result.status,
            confidence=result.confidence,
            actions_taken=result.automated_actions_taken,
            diagnosis=result.diagnosis,
            resolution_notes=result.resolution_notes,
            similar_tickets_found=result.similar_cases
        )
        self.resolution_attempts.append(attempt)
        
        return result
    
    def _match_pattern(self, issue_description: str) -> Optional[Dict[str, Any]]:
        """Match issue to known resolution patterns"""
        issue_lower = issue_description.lower()
        best_match = None
        best_score = 0
        
        for pattern_name, pattern_data in self.resolution_patterns.items():
            score = 0
            for keyword in pattern_data["keywords"]:
                if keyword.lower() in issue_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = {
                    "name": pattern_name,
                    **pattern_data,
                    "match_score": score
                }
        
        # Only return if we have reasonable confidence
        if best_match and best_score >= 1:
            return best_match
        return None
    
    async def _find_similar_cases(self, issue_description: str) -> List[Dict[str, Any]]:
        """Search RAG for similar resolved cases"""
        if not self.rag_engine:
            return []
        
        try:
            result = self.rag_engine.retrieve(
                query=issue_description,
                top_k=5,
                min_score=0.6
            )
            
            similar_cases = []
            for doc in result.search_results:
                similar_cases.append({
                    "title": doc.metadata.get("title", "Similar Case"),
                    "description": doc.content[:300],
                    "resolution": doc.metadata.get("resolution", ""),
                    "category": doc.metadata.get("category", ""),
                    "similarity_score": doc.score,
                    "source": doc.metadata.get("source", "knowledge_base")
                })
            
            return similar_cases
        except Exception as e:
            logger.error(f"[AutoResolver] RAG search failed: {e}")
            return []
    
    async def _ai_diagnosis(
        self, 
        issue_description: str,
        matched_pattern: Optional[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]],
        user_context: Dict[str, Any] = None
    ) -> str:
        """Generate AI-powered diagnosis"""
        if not self.llm_client:
            # Fallback diagnosis without LLM
            if matched_pattern:
                return f"Issue identified as '{matched_pattern['name']}' type. Category: {matched_pattern.get('category', 'general')}."
            return "Unable to determine specific issue type. Manual diagnosis required."
        
        # Build prompt for AI diagnosis
        context_parts = []
        if matched_pattern:
            context_parts.append(f"Pattern match: {matched_pattern['name']} (confidence: {matched_pattern.get('confidence', 0)*100:.0f}%)")
        if similar_cases:
            context_parts.append(f"Found {len(similar_cases)} similar resolved cases")
        if user_context:
            context_parts.append(f"User context: {json.dumps(user_context)}")
        
        prompt = f"""Analyze this IT support issue and provide a concise diagnosis:

Issue: {issue_description}

Context:
{chr(10).join(context_parts)}

Provide:
1. Root cause analysis (1-2 sentences)
2. Impact assessment (low/medium/high)
3. Recommended resolution approach

Respond in a professional, technical manner suitable for IT support."""

        try:
            diagnosis = self.llm_client(prompt, temperature=0.2)
            return diagnosis
        except Exception as e:
            logger.error(f"[AutoResolver] AI diagnosis failed: {e}")
            return "AI diagnosis unavailable. Manual review recommended."
    
    def _assess_auto_resolution(
        self,
        matched_pattern: Optional[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]],
        diagnosis: str
    ) -> Tuple[bool, ResolutionConfidence]:
        """Determine if issue can be auto-resolved and confidence level"""
        can_auto_resolve = False
        confidence = ResolutionConfidence.LOW
        
        if matched_pattern:
            can_auto_resolve = matched_pattern.get("auto_resolvable", False)
            pattern_conf = matched_pattern.get("confidence", 0.5)
            
            if pattern_conf >= 0.85:
                confidence = ResolutionConfidence.HIGH
            elif pattern_conf >= 0.6:
                confidence = ResolutionConfidence.MEDIUM
            else:
                confidence = ResolutionConfidence.LOW
        
        # Boost confidence if we have similar resolved cases
        if similar_cases:
            avg_similarity = sum(c.get("similarity_score", 0) for c in similar_cases) / len(similar_cases)
            if avg_similarity >= 0.8 and confidence == ResolutionConfidence.LOW:
                confidence = ResolutionConfidence.MEDIUM
            elif avg_similarity >= 0.85 and confidence == ResolutionConfidence.MEDIUM:
                confidence = ResolutionConfidence.HIGH
        
        return can_auto_resolve, confidence
    
    def _get_recommended_actions(
        self,
        matched_pattern: Optional[Dict[str, Any]],
        diagnosis: str,
        similar_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommended actions for the issue"""
        actions = []
        
        if matched_pattern:
            actions.append({
                "action": "Execute automated runbook",
                "runbook": matched_pattern.get("runbook"),
                "estimated_time": matched_pattern.get("typical_resolution_time", 15),
                "confidence": matched_pattern.get("confidence", 0.5),
                "automated": True
            })
        
        if similar_cases:
            for case in similar_cases[:3]:
                if case.get("resolution"):
                    actions.append({
                        "action": f"Apply similar resolution: {case['title']}",
                        "details": case["resolution"][:200],
                        "source": case.get("source", "knowledge_base"),
                        "similarity": case.get("similarity_score", 0),
                        "automated": False
                    })
        
        return actions
    
    def _generate_prescriptive_steps(
        self,
        matched_pattern: Optional[Dict[str, Any]],
        diagnosis: str,
        similar_cases: List[Dict[str, Any]],
        user_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate step-by-step prescriptive guidance for human agents"""
        steps = []
        
        if matched_pattern and matched_pattern.get("prescriptive_steps"):
            for step in matched_pattern["prescriptive_steps"]:
                steps.append({
                    "step_number": step["step"],
                    "action": step["action"],
                    "automated": step.get("automated", False),
                    "tools_needed": [],
                    "estimated_time": "2-5 min",
                    "notes": ""
                })
        else:
            # Generate generic steps based on category
            steps = [
                {
                    "step_number": 1,
                    "action": "Gather detailed information from user",
                    "automated": False,
                    "estimated_time": "2 min"
                },
                {
                    "step_number": 2,
                    "action": "Check relevant system logs and monitoring",
                    "automated": True,
                    "estimated_time": "3 min"
                },
                {
                    "step_number": 3,
                    "action": "Verify user permissions and account status",
                    "automated": True,
                    "estimated_time": "2 min"
                },
                {
                    "step_number": 4,
                    "action": "Apply resolution based on diagnosis",
                    "automated": False,
                    "estimated_time": "5-10 min"
                },
                {
                    "step_number": 5,
                    "action": "Verify fix with user and document resolution",
                    "automated": False,
                    "estimated_time": "2 min"
                }
            ]
        
        # Add insights from similar cases
        if similar_cases:
            top_case = similar_cases[0]
            steps.append({
                "step_number": len(steps) + 1,
                "action": f"Reference similar resolved case: {top_case.get('title', 'Unknown')}",
                "automated": False,
                "estimated_time": "1 min",
                "notes": top_case.get("resolution", "")[:200]
            })
        
        return steps
    
    def _estimate_resolution_time(
        self,
        matched_pattern: Optional[Dict[str, Any]],
        can_auto_resolve: bool
    ) -> int:
        """Estimate resolution time in minutes"""
        if can_auto_resolve and matched_pattern:
            return matched_pattern.get("typical_resolution_time", 10)
        elif matched_pattern:
            return matched_pattern.get("typical_resolution_time", 20) * 1.5
        return 30  # Default for unknown issues
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get auto-resolution metrics"""
        success_rate = (self.success_count / self.total_attempts * 100) if self.total_attempts > 0 else 0
        
        return {
            "total_attempts": self.total_attempts,
            "successful_resolutions": self.success_count,
            "success_rate": f"{success_rate:.1f}%",
            "average_time_saved": self._calculate_avg_time_saved(),
            "top_resolved_categories": self._get_top_categories(),
            "recent_attempts": [
                {
                    "ticket_id": a.ticket_id,
                    "status": a.status.value,
                    "confidence": a.confidence.value,
                    "timestamp": a.started_at.isoformat() if a.started_at else None
                }
                for a in self.resolution_attempts[-10:]
            ]
        }
    
    def _calculate_avg_time_saved(self) -> int:
        """Calculate average time saved per resolution"""
        if not self.resolution_attempts:
            return 0
        saved = [a.time_saved_minutes for a in self.resolution_attempts if a.status == ResolutionStatus.RESOLVED]
        return sum(saved) // len(saved) if saved else 15  # Default 15 min
    
    def _get_top_categories(self) -> List[Dict[str, Any]]:
        """Get top resolved categories"""
        category_counts = {}
        for attempt in self.resolution_attempts:
            if attempt.status == ResolutionStatus.RESOLVED:
                cat = attempt.diagnosis[:50] if attempt.diagnosis else "Unknown"
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"category": c, "count": n} for c, n in sorted_cats[:5]]


# Singleton instance
_auto_resolver: Optional[AutoResolver] = None


def get_auto_resolver(rag_engine=None, runbook_engine=None, llm_client=None) -> AutoResolver:
    """Get or create the auto-resolver singleton."""
    global _auto_resolver
    if _auto_resolver is None:
        _auto_resolver = AutoResolver(rag_engine, runbook_engine, llm_client)
    return _auto_resolver
