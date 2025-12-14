"""
ITOps Engine API
================
FastAPI router for the Intelligent ITOps Engine.

Exposes endpoints for:
- Auto-resolution attempts
- Agent assistance
- Incident correlation
- Learning and feedback
- Triage operations
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/itops", tags=["ITOps Engine"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TriageRequest(BaseModel):
    """Request to triage a ticket"""
    ticket_id: Optional[str] = None
    description: str
    category: Optional[str] = None
    priority: Optional[str] = None
    reporter: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None


class AutoResolveRequest(BaseModel):
    """Request for auto-resolution"""
    issue_description: str
    ticket_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    dry_run: bool = False


class AgentAssistRequest(BaseModel):
    """Request for agent assistance"""
    ticket_id: str
    issue_description: str
    category: Optional[str] = None
    priority: Optional[str] = None
    customer_info: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


class CorrelationRequest(BaseModel):
    """Request for incident correlation"""
    incident_id: str
    description: str
    category: Optional[str] = None
    affected_user: Optional[str] = None
    location: Optional[str] = None
    service: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request to submit feedback"""
    ticket_id: str
    score: float = Field(..., ge=0, le=5)
    feedback_type: str = "user_feedback"  # agent_feedback, user_feedback
    comments: Optional[str] = None
    agent_id: Optional[str] = None


class ResolutionRecordRequest(BaseModel):
    """Request to record a resolution"""
    ticket_id: str
    issue_description: str
    issue_category: str
    resolution_description: str
    resolution_steps: Optional[List[str]] = None
    outcome: str = "successful"  # successful, partial, failed, workaround, escalated
    time_to_resolve: int = 0
    auto_resolved: bool = False
    runbook_used: Optional[str] = None
    agent_id: Optional[str] = None


# ============================================================================
# Lazy initialization of engines
# ============================================================================

_engines_initialized = False
_auto_resolver = None
_agent_assist = None
_incident_correlator = None
_learning_engine = None
_triage_engine = None


def _get_engines():
    """Lazy initialization of ITOps engines"""
    global _engines_initialized, _auto_resolver, _agent_assist
    global _incident_correlator, _learning_engine, _triage_engine
    
    if _engines_initialized:
        return
    
    try:
        # Try to get RAG engine if available
        rag_engine = None
        llm_client = None
        
        try:
            from rag_engine import get_rag_engine
            rag_engine = get_rag_engine()
        except ImportError:
            logger.warning("[ITOps API] RAG engine not available")
        
        # Try to get LLM client
        try:
            # Import the query function from backend
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from backend import query_azure_openai
            llm_client = lambda prompt, temperature=0.2: query_azure_openai(prompt, temperature=temperature)
        except ImportError:
            logger.warning("[ITOps API] LLM client not available")
        
        # Initialize engines
        from .auto_resolver import get_auto_resolver
        from .agent_assist import get_agent_assist
        from .incident_correlator import get_incident_correlator
        from .learning_engine import get_learning_engine
        from .triage_engine import get_triage_engine
        from .runbook_engine import get_runbook_engine
        
        runbook_engine = get_runbook_engine()
        
        _auto_resolver = get_auto_resolver(
            rag_engine=rag_engine,
            runbook_engine=runbook_engine,
            llm_client=llm_client
        )
        
        _agent_assist = get_agent_assist(
            rag_engine=rag_engine,
            llm_client=llm_client
        )
        
        _incident_correlator = get_incident_correlator(
            rag_engine=rag_engine
        )
        
        _learning_engine = get_learning_engine(
            rag_engine=rag_engine
        )
        
        _triage_engine = get_triage_engine(
            auto_resolver=_auto_resolver,
            agent_assist=_agent_assist,
            incident_correlator=_incident_correlator,
            rag_engine=rag_engine,
            llm_client=llm_client
        )
        
        _engines_initialized = True
        logger.info("[ITOps API] All engines initialized successfully")
        
    except Exception as e:
        logger.error(f"[ITOps API] Failed to initialize engines: {e}")
        _engines_initialized = False


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/status")
async def get_status():
    """Get ITOps engine status"""
    _get_engines()
    
    return {
        "status": "ok",
        "engines_initialized": _engines_initialized,
        "components": {
            "auto_resolver": _auto_resolver is not None,
            "agent_assist": _agent_assist is not None,
            "incident_correlator": _incident_correlator is not None,
            "learning_engine": _learning_engine is not None,
            "triage_engine": _triage_engine is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/metrics")
async def get_metrics():
    """Get comprehensive ITOps metrics"""
    _get_engines()
    
    metrics = {}
    
    if _auto_resolver:
        metrics["auto_resolution"] = _auto_resolver.get_metrics()
    
    if _agent_assist:
        metrics["agent_assist"] = _agent_assist.get_metrics()
    
    if _incident_correlator:
        metrics["incident_correlation"] = _incident_correlator.get_metrics()
    
    if _learning_engine:
        learning_metrics = _learning_engine.get_metrics()
        metrics["learning"] = {
            "total_resolutions_learned": learning_metrics.total_resolutions_learned,
            "successful_resolutions": learning_metrics.successful_resolutions,
            "average_feedback_score": learning_metrics.average_feedback_score,
            "knowledge_gaps_identified": learning_metrics.knowledge_gaps_identified,
            "auto_resolution_improvement": learning_metrics.auto_resolution_improvement
        }
    
    if _triage_engine:
        metrics["triage"] = _triage_engine.get_metrics()
    
    return {
        "status": "ok",
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }


# ----------------------------------------------------------------------------
# Triage Endpoints
# ----------------------------------------------------------------------------

@router.post("/triage")
async def triage_ticket(request: TriageRequest):
    """
    Intelligent ticket triage.
    
    Analyzes the ticket and determines:
    - Can it be auto-resolved?
    - What are the prescriptive steps?
    - Which team/agent should handle it?
    """
    _get_engines()
    
    if not _triage_engine:
        raise HTTPException(status_code=503, detail="Triage engine not available")
    
    try:
        result = await _triage_engine.triage_ticket(
            ticket_id=request.ticket_id or f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=request.description,
            category=request.category,
            reporter=request.reporter,
            priority=request.priority,
            session_state=request.session_state
        )
        
        return {
            "status": "ok",
            "triage_result": {
                "ticket_id": result.ticket_id,
                "decision": result.decision.value,
                "priority": result.priority.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "auto_resolution_attempted": result.auto_resolution_attempted,
                "auto_resolution_successful": result.auto_resolution_successful,
                "prescriptive_steps": result.prescriptive_steps,
                "recommended_team": result.recommended_team,
                "estimated_resolution_time": result.estimated_resolution_time,
                "similar_cases": result.similar_resolved_cases,
                "kb_articles": result.knowledge_articles,
                "customer_message": result.customer_message,
                "agent_notes": result.agent_notes
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Triage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------------
# Auto-Resolution Endpoints
# ----------------------------------------------------------------------------

@router.post("/auto-resolve")
async def auto_resolve(request: AutoResolveRequest):
    """
    Attempt automatic resolution of an issue.
    
    Analyzes the issue and attempts to resolve it using:
    - Pattern matching
    - Runbook automation
    - AI-powered diagnosis
    """
    _get_engines()
    
    if not _auto_resolver:
        raise HTTPException(status_code=503, detail="Auto-resolver not available")
    
    try:
        result = await _auto_resolver.attempt_resolution(
            issue_description=request.issue_description,
            ticket_id=request.ticket_id,
            user_context=request.user_context,
            dry_run=request.dry_run
        )
        
        return {
            "status": "ok",
            "resolution_result": {
                "ticket_id": result.ticket_id,
                "status": result.status.value,
                "confidence": result.confidence.value,
                "diagnosis": result.diagnosis,
                "recommended_actions": result.recommended_actions,
                "automated_actions_taken": result.automated_actions_taken,
                "resolution_notes": result.resolution_notes,
                "similar_cases": result.similar_cases,
                "prescriptive_steps": result.prescriptive_steps,
                "estimated_time": result.estimated_resolution_time,
                "can_auto_resolve": result.can_auto_resolve,
                "requires_human_review": result.requires_human_review,
                "escalation_recommended": result.escalation_recommended
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Auto-resolve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_issue(request: AutoResolveRequest):
    """
    Analyze an issue without attempting resolution.
    
    Returns diagnosis, recommended actions, and similar cases.
    """
    _get_engines()
    
    if not _auto_resolver:
        raise HTTPException(status_code=503, detail="Auto-resolver not available")
    
    try:
        result = await _auto_resolver.analyze_issue(
            issue_description=request.issue_description,
            ticket_id=request.ticket_id,
            user_context=request.user_context
        )
        
        return {
            "status": "ok",
            "analysis": {
                "diagnosis": result.diagnosis,
                "confidence": result.confidence.value,
                "can_auto_resolve": result.can_auto_resolve,
                "recommended_actions": result.recommended_actions,
                "prescriptive_steps": result.prescriptive_steps,
                "similar_cases": result.similar_cases,
                "estimated_time": result.estimated_resolution_time,
                "requires_human_review": result.requires_human_review
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------------
# Agent Assist Endpoints
# ----------------------------------------------------------------------------

@router.post("/agent-assist")
async def get_agent_assistance(request: AgentAssistRequest):
    """
    Get real-time assistance for an agent handling a ticket.
    
    Returns:
    - Diagnosis
    - Step-by-step resolution guidance
    - Similar resolved cases
    - Relevant KB articles
    - Quick actions
    """
    _get_engines()
    
    if not _agent_assist:
        raise HTTPException(status_code=503, detail="Agent assist not available")
    
    try:
        result = await _agent_assist.get_assistance(
            ticket_id=request.ticket_id,
            issue_description=request.issue_description,
            category=request.category,
            priority=request.priority,
            customer_info=request.customer_info,
            history=request.history
        )
        
        return {
            "status": "ok",
            "assistance": {
                "ticket_id": result.ticket_id,
                "issue_summary": result.issue_summary,
                "diagnosis": result.diagnosis,
                "suggestions": [
                    {
                        "id": s.suggestion_id,
                        "type": s.suggestion_type.value,
                        "priority": s.priority.value,
                        "title": s.title,
                        "description": s.description,
                        "action_details": s.action_details,
                        "confidence": s.confidence,
                        "estimated_time": s.estimated_time_minutes,
                        "reasoning": s.reasoning
                    }
                    for s in result.suggestions
                ],
                "similar_tickets": result.similar_tickets,
                "kb_articles": result.kb_articles,
                "quick_actions": result.quick_actions,
                "priority_assessment": result.priority_assessment,
                "estimated_resolution_time": result.estimated_resolution_time,
                "escalation_recommended": result.escalation_recommended,
                "escalation_team": result.escalation_team
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Agent assist failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostic-commands/{category}")
async def get_diagnostic_commands(category: str):
    """Get diagnostic commands for a category"""
    _get_engines()
    
    if not _agent_assist:
        raise HTTPException(status_code=503, detail="Agent assist not available")
    
    commands = _agent_assist.diagnostic_commands.get(category, [])
    
    return {
        "status": "ok",
        "category": category,
        "commands": commands
    }


@router.get("/resolution-scripts")
async def get_resolution_scripts():
    """Get available resolution scripts"""
    _get_engines()
    
    if not _agent_assist:
        raise HTTPException(status_code=503, detail="Agent assist not available")
    
    return {
        "status": "ok",
        "scripts": _agent_assist.resolution_scripts
    }


# ----------------------------------------------------------------------------
# Incident Correlation Endpoints
# ----------------------------------------------------------------------------

@router.post("/correlate")
async def correlate_incident(request: CorrelationRequest):
    """
    Correlate an incident with existing ones.
    
    Identifies:
    - Related incidents
    - Major incidents
    - Patterns
    - Root causes
    """
    _get_engines()
    
    if not _incident_correlator:
        raise HTTPException(status_code=503, detail="Incident correlator not available")
    
    try:
        result = await _incident_correlator.correlate_incident(
            incident_id=request.incident_id,
            description=request.description,
            category=request.category,
            affected_user=request.affected_user,
            location=request.location,
            service=request.service
        )
        
        return {
            "status": "ok",
            "correlation_result": {
                "incident_id": result.incident_id,
                "correlations_found": [
                    {
                        "correlation_id": c.correlation_id,
                        "related_incidents": c.related_incident_ids,
                        "correlation_types": [t.value for t in c.correlation_types],
                        "affected_users_count": c.affected_users_count,
                        "affected_services": c.affected_services,
                        "is_major_incident": c.is_major_incident,
                        "severity": c.severity.value,
                        "probable_root_cause": c.probable_root_cause,
                        "recommended_actions": c.recommended_actions
                    }
                    for c in result.correlations_found
                ],
                "patterns_matched": [
                    {
                        "pattern_id": p.pattern_id,
                        "name": p.pattern_name,
                        "description": p.description,
                        "prevention": p.prevention_recommendation
                    }
                    for p in result.patterns_matched
                ],
                "is_part_of_major_incident": result.is_part_of_major_incident,
                "major_incident_id": result.major_incident_id,
                "similar_recent_incidents": result.similar_recent_incidents,
                "trend_analysis": result.trend_analysis,
                "recommendations": result.recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Correlation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/major-incidents")
async def get_major_incidents():
    """Get active major incidents"""
    _get_engines()
    
    if not _incident_correlator:
        raise HTTPException(status_code=503, detail="Incident correlator not available")
    
    return {
        "status": "ok",
        "major_incidents": _incident_correlator.get_active_major_incidents()
    }


# ----------------------------------------------------------------------------
# Learning Engine Endpoints
# ----------------------------------------------------------------------------

@router.post("/learn/resolution")
async def record_resolution(request: ResolutionRecordRequest):
    """
    Record a ticket resolution for learning.
    
    The system learns from successful resolutions to improve
    future auto-resolution and recommendations.
    """
    _get_engines()
    
    if not _learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    try:
        from .learning_engine import ResolutionOutcome
        
        outcome_map = {
            "successful": ResolutionOutcome.SUCCESSFUL,
            "partial": ResolutionOutcome.PARTIAL,
            "failed": ResolutionOutcome.FAILED,
            "workaround": ResolutionOutcome.WORKAROUND,
            "escalated": ResolutionOutcome.ESCALATED
        }
        
        outcome = outcome_map.get(request.outcome, ResolutionOutcome.SUCCESSFUL)
        
        record = await _learning_engine.record_resolution(
            ticket_id=request.ticket_id,
            issue_description=request.issue_description,
            issue_category=request.issue_category,
            resolution_description=request.resolution_description,
            resolution_steps=request.resolution_steps,
            outcome=outcome,
            time_to_resolve=request.time_to_resolve,
            auto_resolved=request.auto_resolved,
            runbook_used=request.runbook_used,
            agent_id=request.agent_id
        )
        
        return {
            "status": "ok",
            "record_id": record.record_id,
            "message": "Resolution recorded for learning"
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Recording resolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a resolution"""
    _get_engines()
    
    if not _learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    try:
        from .learning_engine import FeedbackType
        
        feedback_type_map = {
            "agent_feedback": FeedbackType.AGENT_FEEDBACK,
            "user_feedback": FeedbackType.USER_FEEDBACK,
            "resolution_verified": FeedbackType.RESOLUTION_VERIFIED
        }
        
        fb_type = feedback_type_map.get(request.feedback_type, FeedbackType.USER_FEEDBACK)
        
        await _learning_engine.add_feedback(
            ticket_id=request.ticket_id,
            feedback_type=fb_type,
            score=request.score,
            comments=request.comments,
            agent_id=request.agent_id
        )
        
        return {
            "status": "ok",
            "message": "Feedback recorded"
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Recording feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learn/recommendations")
async def get_learning_recommendations(category: Optional[str] = None):
    """Get recommendations for improving knowledge base"""
    _get_engines()
    
    if not _learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    return {
        "status": "ok",
        "recommendations": _learning_engine.get_learning_recommendations(category)
    }


@router.get("/learn/patterns")
async def get_pattern_insights(pattern_key: Optional[str] = None):
    """Get insights for resolution patterns"""
    _get_engines()
    
    if not _learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    return {
        "status": "ok",
        "patterns": _learning_engine.get_pattern_insights(pattern_key)
    }


# ----------------------------------------------------------------------------
# Runbook Endpoints
# ----------------------------------------------------------------------------

@router.get("/runbooks")
async def list_runbooks():
    """List all available runbooks"""
    _get_engines()
    
    try:
        from .runbook_engine import get_runbook_engine
        engine = get_runbook_engine()
        
        return {
            "status": "ok",
            "runbooks": engine.list_all_runbooks()
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Listing runbooks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runbooks/{runbook_id}")
async def get_runbook(runbook_id: str):
    """Get details of a specific runbook"""
    _get_engines()
    
    try:
        from .runbook_engine import get_runbook_engine
        engine = get_runbook_engine()
        
        summary = engine.get_runbook_summary(runbook_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Runbook not found")
        
        return {
            "status": "ok",
            "runbook": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ITOps API] Getting runbook failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runbooks/{runbook_id}/execute")
async def execute_runbook(runbook_id: str, context: Dict[str, Any] = None, dry_run: bool = True):
    """Execute a runbook"""
    _get_engines()
    
    try:
        from .runbook_engine import get_runbook_engine
        engine = get_runbook_engine()
        
        result = await engine.execute_runbook(
            runbook_id=runbook_id,
            context=context or {},
            dry_run=dry_run
        )
        
        return {
            "status": "ok",
            "execution_result": {
                "runbook_id": result.runbook_id,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "error_message": result.error_message,
                "output_data": result.output_data,
                "resolution_notes": result.resolution_notes
            }
        }
        
    except Exception as e:
        logger.error(f"[ITOps API] Runbook execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
