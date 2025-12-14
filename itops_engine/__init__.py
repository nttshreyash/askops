"""
AskOps Intelligent ITOps Engine
================================
Enterprise-grade AI-powered IT Operations automation platform.

Key Capabilities:
1. Auto-Resolution Engine - Automatically resolves tickets before human assignment
2. Prescriptive Analytics - Provides step-by-step guidance to agents
3. Runbook Automation - Executes remediation scripts safely
4. Incident Correlation - Groups related incidents and detects major outages
5. Learning Loop - Continuously improves from resolutions
6. Triage Engine - Intelligent routing and prioritization

Architecture:
-------------
                    ┌─────────────────────────────────────────────────────────┐
                    │                   INCOMING TICKET                        │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │              INTELLIGENT TRIAGE ENGINE                   │
                    │  • Classify issue type & severity                       │
                    │  • Correlate with existing incidents                    │
                    │  • Check if auto-resolvable                             │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         │                                         │
           ┌─────────────▼─────────────┐             ┌────────────▼────────────┐
           │   AUTO-RESOLUTION ENGINE   │             │   AGENT ASSIST ENGINE   │
           │  • Execute safe runbooks   │             │  • Prescriptive steps   │
           │  • Validate resolution     │             │  • Similar case matches │
           │  • Auto-close if fixed     │             │  • Recommended actions  │
           └─────────────┬─────────────┘             └────────────┬────────────┘
                         │                                         │
                         │    ┌───────────────────────────┐       │
                         └────►   RESOLUTION VERIFIED?    ◄───────┘
                              └─────────────┬─────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                LEARNING ENGINE                 │
                    │  • Record successful resolutions              │
                    │  • Update knowledge base                      │
                    │  • Improve future predictions                 │
                    └───────────────────────────────────────────────┘

Usage:
------
    from itops_engine import (
        get_auto_resolver,
        get_agent_assist,
        get_incident_correlator,
        get_learning_engine,
        get_triage_engine,
        get_runbook_engine
    )
    
    # Get engines
    auto_resolver = get_auto_resolver()
    agent_assist = get_agent_assist()
    
    # Triage a ticket
    triage_result = await triage_engine.triage_ticket(
        ticket_id="INC0001234",
        description="VPN not connecting",
        category="network"
    )
    
    # Get agent assistance
    assistance = await agent_assist.get_assistance(
        ticket_id="INC0001234",
        issue_description="VPN not connecting"
    )
"""

from .auto_resolver import AutoResolver, get_auto_resolver
from .agent_assist import AgentAssist, get_agent_assist
from .runbook_engine import RunbookEngine, get_runbook_engine
from .incident_correlator import IncidentCorrelator, get_incident_correlator
from .learning_engine import LearningEngine, get_learning_engine
from .triage_engine import TriageEngine, get_triage_engine
from .itops_api import router as itops_router

__version__ = "1.0.0"
__all__ = [
    # Classes
    "AutoResolver",
    "AgentAssist", 
    "RunbookEngine",
    "IncidentCorrelator",
    "LearningEngine",
    "TriageEngine",
    # Factory functions
    "get_auto_resolver",
    "get_agent_assist",
    "get_runbook_engine",
    "get_incident_correlator",
    "get_learning_engine",
    "get_triage_engine",
    # API router
    "itops_router"
]


def initialize_itops_engine(rag_engine=None, llm_client=None):
    """
    Initialize all ITOps engine components.
    
    Args:
        rag_engine: RAG engine for knowledge retrieval
        llm_client: LLM query function
        
    Returns:
        Dict with all initialized engines
    """
    runbook_engine = get_runbook_engine()
    
    auto_resolver = get_auto_resolver(
        rag_engine=rag_engine,
        runbook_engine=runbook_engine,
        llm_client=llm_client
    )
    
    agent_assist = get_agent_assist(
        rag_engine=rag_engine,
        llm_client=llm_client
    )
    
    incident_correlator = get_incident_correlator(
        rag_engine=rag_engine
    )
    
    learning_engine = get_learning_engine(
        rag_engine=rag_engine
    )
    
    triage_engine = get_triage_engine(
        auto_resolver=auto_resolver,
        agent_assist=agent_assist,
        incident_correlator=incident_correlator,
        rag_engine=rag_engine,
        llm_client=llm_client
    )
    
    return {
        "auto_resolver": auto_resolver,
        "agent_assist": agent_assist,
        "runbook_engine": runbook_engine,
        "incident_correlator": incident_correlator,
        "learning_engine": learning_engine,
        "triage_engine": triage_engine
    }
