"""
Triage Engine
=============
Intelligent ticket triage that determines the best resolution path
before assigning to human agents.

This engine:
1. Analyzes incoming tickets immediately
2. Determines if auto-resolvable
3. Attempts auto-resolution first
4. Provides prescriptive guidance if human needed
5. Routes to appropriate team/agent
"""

import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TriageDecision(str, Enum):
    """Possible triage decisions"""
    AUTO_RESOLVE = "auto_resolve"           # Attempt automated resolution
    SELF_SERVICE = "self_service"           # Guide user to self-service
    ASSIGN_TIER1 = "assign_tier1"           # Route to Tier 1 agent
    ASSIGN_TIER2 = "assign_tier2"           # Route to Tier 2 specialist
    ASSIGN_SPECIALIST = "assign_specialist"  # Route to specific team
    ESCALATE_IMMEDIATE = "escalate_immediate"  # Urgent escalation needed
    REQUEST_INFO = "request_info"           # Need more information


class TriagePriority(str, Enum):
    """Triage priority levels"""
    P1_CRITICAL = "p1_critical"  # Major outage
    P2_HIGH = "p2_high"          # Significant impact
    P3_MEDIUM = "p3_medium"      # Moderate impact  
    P4_LOW = "p4_low"            # Minor impact


@dataclass
class TriageResult:
    """Result of ticket triage"""
    ticket_id: str
    decision: TriageDecision
    priority: TriagePriority
    confidence: float
    reasoning: str
    auto_resolution_attempted: bool
    auto_resolution_successful: bool
    prescriptive_steps: List[Dict[str, Any]]
    recommended_team: Optional[str]
    recommended_agent: Optional[str]
    estimated_resolution_time: int  # minutes
    similar_resolved_cases: List[Dict[str, Any]]
    knowledge_articles: List[Dict[str, Any]]
    customer_message: str  # What to tell the customer
    agent_notes: str  # Internal notes for agent
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriageEngine:
    """
    Intelligent Triage Engine
    
    First point of contact for all incoming tickets.
    Determines the optimal resolution path.
    """
    
    def __init__(
        self, 
        auto_resolver=None, 
        agent_assist=None,
        incident_correlator=None,
        rag_engine=None,
        llm_client=None
    ):
        """
        Initialize the triage engine.
        
        Args:
            auto_resolver: AutoResolver instance
            agent_assist: AgentAssist instance
            incident_correlator: IncidentCorrelator instance
            rag_engine: RAG engine for knowledge search
            llm_client: LLM query function
        """
        self.auto_resolver = auto_resolver
        self.agent_assist = agent_assist
        self.incident_correlator = incident_correlator
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        
        # Team routing rules
        self.routing_rules = self._load_routing_rules()
        
        # SLA definitions
        self.sla_definitions = {
            TriagePriority.P1_CRITICAL: {"response_minutes": 15, "resolution_hours": 4},
            TriagePriority.P2_HIGH: {"response_minutes": 30, "resolution_hours": 8},
            TriagePriority.P3_MEDIUM: {"response_minutes": 120, "resolution_hours": 24},
            TriagePriority.P4_LOW: {"response_minutes": 480, "resolution_hours": 72}
        }
        
        # Self-service capable issue types
        self.self_service_capable = [
            "password_reset",
            "account_unlock",
            "vpn_reconnect",
            "software_request",
            "access_request"
        ]
        
        # Tracking
        self.triage_history: List[TriageResult] = []
        self.auto_resolution_stats = {
            "attempted": 0,
            "successful": 0
        }
    
    def _load_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load routing rules for different issue types"""
        return {
            "network": {
                "team": "Network Operations",
                "tier": 2,
                "keywords": ["network", "firewall", "router", "switch", "connectivity", "outage"],
                "critical_keywords": ["network down", "complete outage", "all users affected"]
            },
            "security": {
                "team": "Security Operations",
                "tier": 2,
                "keywords": ["security", "breach", "malware", "phishing", "unauthorized", "suspicious"],
                "critical_keywords": ["data breach", "ransomware", "compromised"]
            },
            "email": {
                "team": "Messaging Team",
                "tier": 1,
                "keywords": ["email", "outlook", "exchange", "calendar", "mailbox"],
                "critical_keywords": ["email down", "all email", "exchange outage"]
            },
            "access": {
                "team": "Identity Management",
                "tier": 1,
                "keywords": ["password", "login", "access", "permission", "locked", "reset"],
                "critical_keywords": ["all logins failing", "ad down"]
            },
            "hardware": {
                "team": "Desktop Support",
                "tier": 1,
                "keywords": ["laptop", "monitor", "keyboard", "mouse", "printer", "dock"],
                "critical_keywords": ["executive", "vip", "ceo"]
            },
            "software": {
                "team": "Application Support",
                "tier": 1,
                "keywords": ["install", "software", "application", "update", "crash", "error"],
                "critical_keywords": ["production down", "critical application"]
            },
            "database": {
                "team": "Database Administration",
                "tier": 2,
                "keywords": ["database", "sql", "oracle", "postgres", "query"],
                "critical_keywords": ["database down", "data corruption"]
            },
            "server": {
                "team": "Server Operations",
                "tier": 2,
                "keywords": ["server", "vm", "virtual", "reboot", "restart server"],
                "critical_keywords": ["server down", "production server"]
            }
        }
    
    async def triage_ticket(
        self,
        ticket_id: str,
        description: str,
        category: str = None,
        reporter: Dict[str, Any] = None,
        priority: str = None,
        session_state: Dict[str, Any] = None
    ) -> TriageResult:
        """
        Triage an incoming ticket.
        
        Args:
            ticket_id: Ticket ID
            description: Issue description
            category: Issue category if known
            reporter: Reporter information
            priority: Suggested priority
            session_state: Current session state
            
        Returns:
            TriageResult with decision and guidance
        """
        logger.info(f"[TriageEngine] Triaging ticket {ticket_id}")
        
        # Step 1: Analyze the ticket
        analysis = await self._analyze_ticket(description, category, reporter)
        
        # Step 2: Determine priority
        determined_priority = self._determine_priority(analysis, priority)
        
        # Step 3: Check for correlations (major incidents)
        correlation_result = None
        if self.incident_correlator:
            correlation_result = await self.incident_correlator.correlate_incident(
                incident_id=ticket_id,
                description=description,
                category=category or analysis.get("detected_category", ""),
                affected_user=reporter.get("username") if reporter else None,
                service=analysis.get("affected_service"),
                created_at=datetime.now()
            )
        
        # Step 4: Check if auto-resolvable
        auto_resolution_attempted = False
        auto_resolution_successful = False
        prescriptive_steps = []
        
        if self.auto_resolver and analysis.get("auto_resolvable", False):
            logger.info(f"[TriageEngine] Attempting auto-resolution for {ticket_id}")
            self.auto_resolution_stats["attempted"] += 1
            
            auto_result = await self.auto_resolver.attempt_resolution(
                issue_description=description,
                ticket_id=ticket_id,
                user_context=reporter,
                session_state=session_state,
                dry_run=False
            )
            
            auto_resolution_attempted = True
            auto_resolution_successful = auto_result.status.value == "resolved"
            prescriptive_steps = auto_result.prescriptive_steps
            
            if auto_resolution_successful:
                self.auto_resolution_stats["successful"] += 1
        
        # Step 5: Get prescriptive steps if not auto-resolved
        if not auto_resolution_successful and self.agent_assist:
            assist_result = await self.agent_assist.get_assistance(
                ticket_id=ticket_id,
                issue_description=description,
                category=category or analysis.get("detected_category"),
                priority=determined_priority.value,
                customer_info=reporter
            )
            
            prescriptive_steps = [
                {
                    "step_number": i + 1,
                    "action": s.title,
                    "description": s.description,
                    "confidence": s.confidence,
                    "automated": s.suggestion_type.value in ["diagnostic_command", "script_execution"]
                }
                for i, s in enumerate(assist_result.suggestions)
            ]
        
        # Step 6: Make triage decision
        decision, reasoning = self._make_decision(
            analysis,
            auto_resolution_successful,
            determined_priority,
            correlation_result
        )
        
        # Step 7: Determine routing
        team, agent = self._determine_routing(analysis, decision, determined_priority)
        
        # Step 8: Get similar cases and KB articles
        similar_cases = []
        kb_articles = []
        if self.rag_engine:
            similar_cases = await self._get_similar_cases(description)
            kb_articles = await self._get_kb_articles(description)
        
        # Step 9: Generate customer message
        customer_message = self._generate_customer_message(
            decision,
            auto_resolution_successful,
            prescriptive_steps,
            team
        )
        
        # Step 10: Generate agent notes
        agent_notes = self._generate_agent_notes(
            analysis,
            correlation_result,
            prescriptive_steps,
            similar_cases
        )
        
        # Calculate confidence
        confidence = analysis.get("classification_confidence", 0.7)
        if auto_resolution_successful:
            confidence = 0.95
        elif len(prescriptive_steps) >= 3:
            confidence = min(confidence + 0.1, 0.9)
        
        result = TriageResult(
            ticket_id=ticket_id,
            decision=decision,
            priority=determined_priority,
            confidence=confidence,
            reasoning=reasoning,
            auto_resolution_attempted=auto_resolution_attempted,
            auto_resolution_successful=auto_resolution_successful,
            prescriptive_steps=prescriptive_steps,
            recommended_team=team,
            recommended_agent=agent,
            estimated_resolution_time=self._estimate_resolution_time(decision, determined_priority, auto_resolution_successful),
            similar_resolved_cases=similar_cases,
            knowledge_articles=kb_articles,
            customer_message=customer_message,
            agent_notes=agent_notes,
            metadata={
                "analysis": analysis,
                "correlation": correlation_result.correlations_found if correlation_result else [],
                "is_major_incident": correlation_result.is_part_of_major_incident if correlation_result else False
            }
        )
        
        self.triage_history.append(result)
        
        return result
    
    async def _analyze_ticket(
        self,
        description: str,
        category: str,
        reporter: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze ticket to understand the issue"""
        analysis = {
            "description": description,
            "detected_category": category,
            "keywords": [],
            "affected_service": None,
            "auto_resolvable": False,
            "requires_specialist": False,
            "vip_reporter": False,
            "classification_confidence": 0.7
        }
        
        desc_lower = description.lower()
        
        # Keyword extraction
        all_keywords = []
        for rule_name, rule in self.routing_rules.items():
            for kw in rule.get("keywords", []):
                if kw in desc_lower:
                    all_keywords.append(kw)
                    if not analysis["detected_category"]:
                        analysis["detected_category"] = rule_name
        
        analysis["keywords"] = list(set(all_keywords))
        
        # Detect affected service
        service_keywords = {
            "email": ["email", "outlook", "exchange"],
            "vpn": ["vpn", "remote"],
            "active_directory": ["login", "password", "account", "ad"],
            "network": ["network", "internet", "wifi"],
            "sharepoint": ["sharepoint", "teams", "onedrive"]
        }
        
        for service, keywords in service_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                analysis["affected_service"] = service
                break
        
        # Check if auto-resolvable
        auto_resolve_patterns = [
            "password reset", "forgot password", "reset my password",
            "account locked", "locked out", "unlock account",
            "vpn not connecting", "can't connect to vpn"
        ]
        
        if any(pattern in desc_lower for pattern in auto_resolve_patterns):
            analysis["auto_resolvable"] = True
            analysis["classification_confidence"] = 0.9
        
        # Check for specialist keywords
        for rule_name, rule in self.routing_rules.items():
            for kw in rule.get("critical_keywords", []):
                if kw in desc_lower:
                    analysis["requires_specialist"] = True
                    analysis["detected_category"] = rule_name
                    analysis["classification_confidence"] = 0.85
                    break
        
        # Check VIP status
        if reporter:
            vip_keywords = ["vp", "director", "manager", "executive", "ceo", "cfo", "cto"]
            title = (reporter.get("title") or "").lower()
            if any(vip in title for vip in vip_keywords):
                analysis["vip_reporter"] = True
        
        # Use LLM for better classification if available
        if self.llm_client and analysis["classification_confidence"] < 0.8:
            try:
                prompt = f"""Classify this IT support ticket:

Description: {description}

Return JSON with:
- category (network, email, access, hardware, software, security, database, server, other)
- affected_service (email, vpn, active_directory, network, sharepoint, or null)
- auto_resolvable (true/false) - can this be solved automatically?
- requires_specialist (true/false) - needs Tier 2 or above?
- confidence (0.0-1.0)
"""
                response = self.llm_client(prompt, temperature=0.0)
                try:
                    parsed = json.loads(response)
                    analysis.update({
                        "detected_category": parsed.get("category", analysis["detected_category"]),
                        "affected_service": parsed.get("affected_service", analysis["affected_service"]),
                        "auto_resolvable": parsed.get("auto_resolvable", analysis["auto_resolvable"]),
                        "requires_specialist": parsed.get("requires_specialist", analysis["requires_specialist"]),
                        "classification_confidence": parsed.get("confidence", analysis["classification_confidence"])
                    })
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.error(f"[TriageEngine] LLM classification failed: {e}")
        
        return analysis
    
    def _determine_priority(
        self,
        analysis: Dict[str, Any],
        suggested_priority: str = None
    ) -> TriagePriority:
        """Determine ticket priority"""
        
        # Start with suggested or default
        if suggested_priority:
            suggested_lower = suggested_priority.lower()
            if "1" in suggested_lower or "critical" in suggested_lower:
                return TriagePriority.P1_CRITICAL
            elif "2" in suggested_lower or "high" in suggested_lower:
                return TriagePriority.P2_HIGH
            elif "3" in suggested_lower or "medium" in suggested_lower:
                return TriagePriority.P3_MEDIUM
        
        # Default priority
        priority = TriagePriority.P3_MEDIUM
        
        # Adjust based on analysis
        desc_lower = analysis.get("description", "").lower()
        
        # Critical indicators
        critical_keywords = ["down", "outage", "all users", "production", "cannot work", "urgent", "asap"]
        if any(kw in desc_lower for kw in critical_keywords):
            priority = TriagePriority.P1_CRITICAL
        
        # High priority indicators
        elif any(kw in desc_lower for kw in ["multiple users", "team affected", "important meeting", "deadline"]):
            priority = TriagePriority.P2_HIGH
        
        # VIP gets priority boost
        if analysis.get("vip_reporter"):
            if priority == TriagePriority.P3_MEDIUM:
                priority = TriagePriority.P2_HIGH
            elif priority == TriagePriority.P4_LOW:
                priority = TriagePriority.P3_MEDIUM
        
        # Security issues are always at least P2
        if analysis.get("detected_category") == "security":
            if priority in [TriagePriority.P3_MEDIUM, TriagePriority.P4_LOW]:
                priority = TriagePriority.P2_HIGH
        
        return priority
    
    def _make_decision(
        self,
        analysis: Dict[str, Any],
        auto_resolved: bool,
        priority: TriagePriority,
        correlation_result
    ) -> Tuple[TriageDecision, str]:
        """Make triage decision"""
        
        # If auto-resolved, we're done
        if auto_resolved:
            return TriageDecision.AUTO_RESOLVE, "Issue was automatically resolved"
        
        # If part of major incident, escalate immediately
        if correlation_result and correlation_result.is_part_of_major_incident:
            return TriageDecision.ESCALATE_IMMEDIATE, f"Part of major incident: {correlation_result.major_incident_id}"
        
        # Critical priority goes to Tier 2
        if priority == TriagePriority.P1_CRITICAL:
            return TriageDecision.ESCALATE_IMMEDIATE, "Critical priority requires immediate attention"
        
        # Auto-resolvable but failed - try self-service
        if analysis.get("auto_resolvable"):
            return TriageDecision.SELF_SERVICE, "Attempting guided self-service resolution"
        
        # Requires specialist
        if analysis.get("requires_specialist"):
            return TriageDecision.ASSIGN_TIER2, f"Requires specialist attention ({analysis.get('detected_category')})"
        
        # VIP gets Tier 2
        if analysis.get("vip_reporter"):
            return TriageDecision.ASSIGN_TIER2, "VIP user requires priority handling"
        
        # Default to Tier 1
        return TriageDecision.ASSIGN_TIER1, "Standard ticket routing to Tier 1 support"
    
    def _determine_routing(
        self,
        analysis: Dict[str, Any],
        decision: TriageDecision,
        priority: TriagePriority
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determine team and agent routing"""
        
        category = analysis.get("detected_category", "general")
        rule = self.routing_rules.get(category, {})
        
        team = rule.get("team", "Service Desk")
        agent = None
        
        # Adjust team based on decision
        if decision == TriageDecision.ESCALATE_IMMEDIATE:
            if priority == TriagePriority.P1_CRITICAL:
                team = "Incident Management"
        elif decision == TriageDecision.ASSIGN_TIER2:
            # Use specialist team from rules
            pass
        elif decision == TriageDecision.ASSIGN_TIER1:
            team = "Service Desk"
        
        return team, agent
    
    async def _get_similar_cases(self, description: str) -> List[Dict[str, Any]]:
        """Get similar resolved cases"""
        if not self.rag_engine:
            return []
        
        try:
            result = self.rag_engine.retrieve(
                query=description,
                top_k=3,
                min_score=0.7
            )
            
            return [
                {
                    "ticket_id": doc.metadata.get("ticket_id", ""),
                    "description": doc.content[:200],
                    "resolution": doc.metadata.get("resolution", ""),
                    "similarity": doc.score
                }
                for doc in result.search_results
            ]
        except Exception:
            return []
    
    async def _get_kb_articles(self, description: str) -> List[Dict[str, Any]]:
        """Get relevant KB articles"""
        if not self.rag_engine:
            return []
        
        try:
            result = self.rag_engine.retrieve(
                query=description,
                top_k=3,
                min_score=0.65,
                doc_type="kb_article"
            )
            
            return [
                {
                    "article_id": doc.metadata.get("article_id", ""),
                    "title": doc.metadata.get("title", "KB Article"),
                    "url": doc.metadata.get("url", ""),
                    "relevance": doc.score
                }
                for doc in result.search_results
            ]
        except Exception:
            return []
    
    def _generate_customer_message(
        self,
        decision: TriageDecision,
        auto_resolved: bool,
        prescriptive_steps: List[Dict[str, Any]],
        team: str
    ) -> str:
        """Generate message for the customer"""
        
        if auto_resolved:
            return "Great news! Your issue has been automatically resolved. Please verify that everything is working correctly. If you still experience problems, please let us know."
        
        if decision == TriageDecision.SELF_SERVICE:
            steps_text = "\n".join([f"• {s['action']}" for s in prescriptive_steps[:3]])
            return f"I can help you resolve this. Please try these steps:\n\n{steps_text}\n\nLet me know if this resolves your issue."
        
        if decision == TriageDecision.ESCALATE_IMMEDIATE:
            return f"Your issue has been marked as high priority and escalated to our {team} team. A technician will contact you shortly."
        
        if decision in [TriageDecision.ASSIGN_TIER1, TriageDecision.ASSIGN_TIER2]:
            return f"Your ticket has been assigned to our {team}. They will begin working on your issue shortly. Expected response time: based on priority."
        
        return "Your request has been received and is being processed. We will update you shortly."
    
    def _generate_agent_notes(
        self,
        analysis: Dict[str, Any],
        correlation_result,
        prescriptive_steps: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """Generate internal notes for the agent"""
        notes = []
        
        # Analysis summary
        notes.append(f"**Category:** {analysis.get('detected_category', 'Unknown')}")
        notes.append(f"**Affected Service:** {analysis.get('affected_service', 'Unknown')}")
        notes.append(f"**Classification Confidence:** {analysis.get('classification_confidence', 0)*100:.0f}%")
        
        if analysis.get("keywords"):
            notes.append(f"**Keywords:** {', '.join(analysis['keywords'])}")
        
        if analysis.get("vip_reporter"):
            notes.append("⚠️ **VIP User** - Prioritize handling")
        
        # Correlation info
        if correlation_result and correlation_result.is_part_of_major_incident:
            notes.append(f"🚨 **PART OF MAJOR INCIDENT:** {correlation_result.major_incident_id}")
        
        # Prescriptive steps
        if prescriptive_steps:
            notes.append("\n**Recommended Resolution Steps:**")
            for step in prescriptive_steps[:5]:
                automated = "🤖" if step.get("automated") else "👤"
                notes.append(f"{automated} {step.get('step_number', '')}. {step.get('action', '')}")
        
        # Similar cases
        if similar_cases:
            notes.append("\n**Similar Resolved Cases:**")
            for case in similar_cases[:2]:
                notes.append(f"• {case.get('ticket_id', 'Unknown')}: {case.get('resolution', '')[:100]}")
        
        return "\n".join(notes)
    
    def _estimate_resolution_time(
        self,
        decision: TriageDecision,
        priority: TriagePriority,
        auto_resolved: bool
    ) -> int:
        """Estimate resolution time in minutes"""
        
        if auto_resolved:
            return 5
        
        # Base times by decision
        base_times = {
            TriageDecision.SELF_SERVICE: 15,
            TriageDecision.ASSIGN_TIER1: 30,
            TriageDecision.ASSIGN_TIER2: 60,
            TriageDecision.ASSIGN_SPECIALIST: 120,
            TriageDecision.ESCALATE_IMMEDIATE: 30,
            TriageDecision.REQUEST_INFO: 60
        }
        
        base = base_times.get(decision, 30)
        
        # Adjust by priority
        priority_multipliers = {
            TriagePriority.P1_CRITICAL: 0.5,  # Faster for critical
            TriagePriority.P2_HIGH: 0.75,
            TriagePriority.P3_MEDIUM: 1.0,
            TriagePriority.P4_LOW: 1.5
        }
        
        return int(base * priority_multipliers.get(priority, 1.0))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get triage engine metrics"""
        total = len(self.triage_history)
        auto_attempted = self.auto_resolution_stats["attempted"]
        auto_successful = self.auto_resolution_stats["successful"]
        
        decision_counts = {}
        for result in self.triage_history:
            decision = result.decision.value
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        return {
            "total_triaged": total,
            "auto_resolution_attempted": auto_attempted,
            "auto_resolution_successful": auto_successful,
            "auto_resolution_rate": f"{(auto_successful/auto_attempted*100):.1f}%" if auto_attempted > 0 else "N/A",
            "decision_breakdown": decision_counts,
            "average_confidence": sum(r.confidence for r in self.triage_history) / total if total > 0 else 0
        }


# Singleton instance
_triage_engine: Optional[TriageEngine] = None


def get_triage_engine(
    auto_resolver=None,
    agent_assist=None,
    incident_correlator=None,
    rag_engine=None,
    llm_client=None
) -> TriageEngine:
    """Get or create the triage engine singleton."""
    global _triage_engine
    if _triage_engine is None:
        _triage_engine = TriageEngine(
            auto_resolver, 
            agent_assist, 
            incident_correlator, 
            rag_engine, 
            llm_client
        )
    return _triage_engine
