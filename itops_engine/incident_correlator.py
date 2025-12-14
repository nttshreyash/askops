"""
Incident Correlator Engine
==========================
Correlates related incidents to identify patterns, major incidents, and root causes.

This engine:
1. Groups related incidents together
2. Detects major incidents affecting multiple users
3. Identifies recurring issues
4. Tracks incident trends over time
5. Provides root cause analysis suggestions
"""

import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrelationType(str, Enum):
    """Types of incident correlation"""
    SAME_SERVICE = "same_service"          # Affects same service/system
    SAME_USER = "same_user"                # Same user reporting
    SAME_LOCATION = "same_location"        # Same office/location
    SAME_TIMEFRAME = "same_timeframe"      # Occurred around same time
    SAME_CATEGORY = "same_category"        # Same issue category
    SAME_ROOT_CAUSE = "same_root_cause"    # Likely same root cause
    CHILD_OF_MAJOR = "child_of_major"      # Part of major incident


class IncidentSeverity(str, Enum):
    """Incident severity levels"""
    CRITICAL = "critical"  # P1 - Major outage
    HIGH = "high"          # P2 - Significant impact
    MEDIUM = "medium"      # P3 - Moderate impact
    LOW = "low"            # P4 - Minor impact


@dataclass
class CorrelatedIncident:
    """A correlated incident group"""
    correlation_id: str
    primary_incident_id: str
    related_incident_ids: List[str]
    correlation_types: List[CorrelationType]
    affected_users_count: int
    affected_services: List[str]
    start_time: datetime
    is_major_incident: bool
    severity: IncidentSeverity
    probable_root_cause: Optional[str] = None
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentPattern:
    """A recurring incident pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    frequency: int  # Times seen
    typical_category: str
    typical_resolution: str
    prevention_recommendation: str
    last_seen: datetime
    affected_systems: List[str] = field(default_factory=list)


@dataclass
class CorrelationResult:
    """Result from incident correlation analysis"""
    incident_id: str
    correlations_found: List[CorrelatedIncident]
    patterns_matched: List[IncidentPattern]
    is_part_of_major_incident: bool
    major_incident_id: Optional[str] = None
    similar_recent_incidents: List[Dict[str, Any]] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class IncidentCorrelator:
    """
    Intelligent Incident Correlation Engine
    
    Analyzes incidents to find patterns, group related issues,
    and identify major incidents proactively.
    """
    
    def __init__(self, rag_engine=None, servicenow_client=None):
        """
        Initialize the correlator.
        
        Args:
            rag_engine: RAG engine for similarity search
            servicenow_client: ServiceNow API client
        """
        self.rag_engine = rag_engine
        self.servicenow_client = servicenow_client
        
        # Active correlations
        self.active_correlations: Dict[str, CorrelatedIncident] = {}
        
        # Incident patterns learned
        self.patterns: Dict[str, IncidentPattern] = {}
        self._load_known_patterns()
        
        # Recent incidents cache (last 24 hours)
        self.recent_incidents: List[Dict[str, Any]] = []
        
        # Service dependency map
        self.service_dependencies = self._load_service_dependencies()
        
        # Major incident thresholds
        self.major_incident_thresholds = {
            "min_affected_users": 10,
            "min_related_incidents": 5,
            "time_window_minutes": 60,
            "critical_services": ["email", "vpn", "active_directory", "network", "erp"]
        }
    
    def _load_known_patterns(self):
        """Load known incident patterns"""
        self.patterns = {
            "monday_password_reset": IncidentPattern(
                pattern_id="pat_001",
                pattern_name="Monday Password Reset Spike",
                description="Higher than normal password reset requests on Monday mornings",
                frequency=0,
                typical_category="access",
                typical_resolution="Normal self-service password reset",
                prevention_recommendation="Send reminder email on Friday about password expiry",
                last_seen=datetime.now(),
                affected_systems=["Active Directory", "SSO"]
            ),
            "vpn_peak_hours": IncidentPattern(
                pattern_id="pat_002",
                pattern_name="VPN Capacity Issues During Peak Hours",
                description="VPN connection issues during 8-10am when remote workers connect",
                frequency=0,
                typical_category="network",
                typical_resolution="Wait for capacity or restart VPN client",
                prevention_recommendation="Consider VPN infrastructure scaling",
                last_seen=datetime.now(),
                affected_systems=["VPN Gateway", "Network"]
            ),
            "post_update_issues": IncidentPattern(
                pattern_id="pat_003",
                pattern_name="Post-Update Application Issues",
                description="Application issues following Windows/software updates",
                frequency=0,
                typical_category="software",
                typical_resolution="Rollback update or repair installation",
                prevention_recommendation="Improve update testing and staged rollout",
                last_seen=datetime.now(),
                affected_systems=["Windows", "Office", "Custom Apps"]
            ),
            "quarterly_certificate_expiry": IncidentPattern(
                pattern_id="pat_004",
                pattern_name="Certificate Expiry Issues",
                description="SSL/TLS certificate expiry causing connectivity issues",
                frequency=0,
                typical_category="security",
                typical_resolution="Renew and deploy certificates",
                prevention_recommendation="Implement certificate monitoring and auto-renewal",
                last_seen=datetime.now(),
                affected_systems=["Web Services", "APIs", "VPN"]
            ),
            "network_switch_failure": IncidentPattern(
                pattern_id="pat_005",
                pattern_name="Network Switch/Router Issues",
                description="Multiple users in same location lose connectivity",
                frequency=0,
                typical_category="network",
                typical_resolution="Power cycle switch or escalate to network team",
                prevention_recommendation="Implement redundant network paths",
                last_seen=datetime.now(),
                affected_systems=["Network Infrastructure"]
            )
        }
    
    def _load_service_dependencies(self) -> Dict[str, List[str]]:
        """Load service dependency mapping"""
        return {
            "email": ["active_directory", "exchange", "network"],
            "vpn": ["active_directory", "network", "firewall"],
            "sharepoint": ["active_directory", "azure_ad", "network"],
            "teams": ["azure_ad", "network", "sharepoint"],
            "erp": ["database", "network", "active_directory"],
            "crm": ["database", "network", "active_directory"],
            "printing": ["print_server", "network", "active_directory"]
        }
    
    async def correlate_incident(
        self,
        incident_id: str,
        description: str,
        category: str,
        affected_user: str = None,
        location: str = None,
        service: str = None,
        created_at: datetime = None
    ) -> CorrelationResult:
        """
        Correlate a new incident with existing ones.
        
        Args:
            incident_id: The incident ID
            description: Incident description
            category: Issue category
            affected_user: User who reported
            location: User location/office
            service: Affected service
            created_at: When incident was created
            
        Returns:
            CorrelationResult with findings
        """
        logger.info(f"[IncidentCorrelator] Analyzing incident {incident_id}")
        
        created_at = created_at or datetime.now()
        
        # Add to recent incidents
        self._add_to_recent(incident_id, description, category, affected_user, location, service, created_at)
        
        # Find correlations
        correlations = await self._find_correlations(
            incident_id, description, category, affected_user, location, service, created_at
        )
        
        # Match against known patterns
        matched_patterns = self._match_patterns(description, category, created_at)
        
        # Check if part of major incident
        is_major, major_id = self._check_major_incident(correlations, service, category)
        
        # Find similar recent incidents
        similar = await self._find_similar_recent(description)
        
        # Analyze trends
        trends = self._analyze_trends(category, service)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            correlations, matched_patterns, is_major, trends
        )
        
        result = CorrelationResult(
            incident_id=incident_id,
            correlations_found=correlations,
            patterns_matched=matched_patterns,
            is_part_of_major_incident=is_major,
            major_incident_id=major_id,
            similar_recent_incidents=similar,
            trend_analysis=trends,
            recommendations=recommendations
        )
        
        # If this is a major incident, create correlation group
        if is_major and not major_id:
            self._create_major_incident_group(incident_id, correlations, category, service)
        
        return result
    
    def _add_to_recent(
        self,
        incident_id: str,
        description: str,
        category: str,
        affected_user: str,
        location: str,
        service: str,
        created_at: datetime
    ):
        """Add incident to recent incidents cache"""
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_incidents = [
            inc for inc in self.recent_incidents 
            if inc.get("created_at", datetime.now()) > cutoff
        ]
        
        self.recent_incidents.append({
            "incident_id": incident_id,
            "description": description,
            "category": category,
            "affected_user": affected_user,
            "location": location,
            "service": service,
            "created_at": created_at
        })
    
    async def _find_correlations(
        self,
        incident_id: str,
        description: str,
        category: str,
        affected_user: str,
        location: str,
        service: str,
        created_at: datetime
    ) -> List[CorrelatedIncident]:
        """Find correlations with other incidents"""
        correlations = []
        correlation_counter = 0
        
        # Time window for correlation (last 2 hours)
        time_window = created_at - timedelta(hours=2)
        
        # Group by different correlation types
        same_service_incidents = []
        same_location_incidents = []
        same_category_incidents = []
        
        for inc in self.recent_incidents:
            if inc["incident_id"] == incident_id:
                continue
            
            inc_time = inc.get("created_at", datetime.now())
            if inc_time < time_window:
                continue
            
            # Check correlations
            if service and inc.get("service") == service:
                same_service_incidents.append(inc)
            
            if location and inc.get("location") == location:
                same_location_incidents.append(inc)
            
            if category and inc.get("category") == category:
                same_category_incidents.append(inc)
        
        # Create correlation groups if threshold met
        threshold = 3
        
        if len(same_service_incidents) >= threshold:
            correlation_counter += 1
            affected_users = set(inc.get("affected_user") for inc in same_service_incidents if inc.get("affected_user"))
            correlations.append(CorrelatedIncident(
                correlation_id=f"corr_{correlation_counter}_{datetime.now().strftime('%Y%m%d%H%M')}",
                primary_incident_id=incident_id,
                related_incident_ids=[inc["incident_id"] for inc in same_service_incidents],
                correlation_types=[CorrelationType.SAME_SERVICE],
                affected_users_count=len(affected_users) + 1,
                affected_services=[service],
                start_time=min(inc.get("created_at", datetime.now()) for inc in same_service_incidents),
                is_major_incident=len(same_service_incidents) >= 5,
                severity=IncidentSeverity.HIGH if len(same_service_incidents) >= 5 else IncidentSeverity.MEDIUM,
                probable_root_cause=f"Service '{service}' may be experiencing issues",
                recommended_actions=[
                    f"Check {service} service health",
                    f"Review {service} logs for errors",
                    "Consider proactive communication to users"
                ]
            ))
        
        if len(same_location_incidents) >= threshold:
            correlation_counter += 1
            correlations.append(CorrelatedIncident(
                correlation_id=f"corr_{correlation_counter}_{datetime.now().strftime('%Y%m%d%H%M')}",
                primary_incident_id=incident_id,
                related_incident_ids=[inc["incident_id"] for inc in same_location_incidents],
                correlation_types=[CorrelationType.SAME_LOCATION],
                affected_users_count=len(same_location_incidents) + 1,
                affected_services=list(set(inc.get("service", "") for inc in same_location_incidents if inc.get("service"))),
                start_time=min(inc.get("created_at", datetime.now()) for inc in same_location_incidents),
                is_major_incident=False,
                severity=IncidentSeverity.MEDIUM,
                probable_root_cause=f"Location-specific issue at '{location}'",
                recommended_actions=[
                    f"Check network infrastructure at {location}",
                    "Contact local IT support",
                    "Verify no local outages"
                ]
            ))
        
        # Check for service dependencies
        if service:
            dependencies = self.service_dependencies.get(service, [])
            for dep in dependencies:
                dep_incidents = [
                    inc for inc in self.recent_incidents
                    if inc.get("service") == dep and inc_time > time_window
                ]
                if dep_incidents:
                    correlation_counter += 1
                    correlations.append(CorrelatedIncident(
                        correlation_id=f"corr_{correlation_counter}_{datetime.now().strftime('%Y%m%d%H%M')}",
                        primary_incident_id=incident_id,
                        related_incident_ids=[inc["incident_id"] for inc in dep_incidents],
                        correlation_types=[CorrelationType.SAME_ROOT_CAUSE],
                        affected_users_count=len(dep_incidents) + 1,
                        affected_services=[service, dep],
                        start_time=min(inc.get("created_at", datetime.now()) for inc in dep_incidents),
                        is_major_incident=False,
                        severity=IncidentSeverity.MEDIUM,
                        probable_root_cause=f"Dependency issue: '{dep}' affecting '{service}'",
                        recommended_actions=[
                            f"Investigate {dep} service status",
                            f"Check connectivity between {service} and {dep}",
                            "Review service dependency chain"
                        ]
                    ))
        
        return correlations
    
    def _match_patterns(
        self,
        description: str,
        category: str,
        created_at: datetime
    ) -> List[IncidentPattern]:
        """Match incident against known patterns"""
        matched = []
        
        desc_lower = description.lower()
        
        # Monday password reset pattern
        if category == "access" and "password" in desc_lower:
            if created_at.weekday() == 0 and created_at.hour < 12:
                pattern = self.patterns["monday_password_reset"]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                matched.append(pattern)
        
        # VPN peak hours pattern
        if "vpn" in desc_lower or category == "vpn":
            if 8 <= created_at.hour <= 10:
                pattern = self.patterns["vpn_peak_hours"]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                matched.append(pattern)
        
        # Post-update issues
        update_keywords = ["after update", "since update", "windows update", "office update"]
        if any(kw in desc_lower for kw in update_keywords):
            pattern = self.patterns["post_update_issues"]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            matched.append(pattern)
        
        # Certificate issues
        cert_keywords = ["certificate", "ssl", "tls", "https error", "security warning"]
        if any(kw in desc_lower for kw in cert_keywords):
            pattern = self.patterns["quarterly_certificate_expiry"]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            matched.append(pattern)
        
        return matched
    
    def _check_major_incident(
        self,
        correlations: List[CorrelatedIncident],
        service: str,
        category: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if this incident is part of a major incident"""
        
        # Check existing active correlations
        for corr_id, corr in self.active_correlations.items():
            if corr.is_major_incident:
                # Check if service matches
                if service in corr.affected_services:
                    return True, corr.correlation_id
                # Check if in related incidents
                if any(c.primary_incident_id in corr.related_incident_ids for c in correlations):
                    return True, corr.correlation_id
        
        # Check if current correlations indicate major incident
        for corr in correlations:
            if corr.is_major_incident:
                return True, None
            
            # Check thresholds
            if corr.affected_users_count >= self.major_incident_thresholds["min_affected_users"]:
                return True, None
            
            if len(corr.related_incident_ids) >= self.major_incident_thresholds["min_related_incidents"]:
                return True, None
        
        # Check if critical service
        if service in self.major_incident_thresholds["critical_services"]:
            # Count recent incidents for this service
            recent_count = sum(
                1 for inc in self.recent_incidents 
                if inc.get("service") == service
            )
            if recent_count >= 3:
                return True, None
        
        return False, None
    
    async def _find_similar_recent(self, description: str) -> List[Dict[str, Any]]:
        """Find similar recent incidents using RAG"""
        if not self.rag_engine:
            # Fallback to keyword matching
            return self._keyword_similar_search(description)
        
        try:
            result = self.rag_engine.retrieve(
                query=description,
                top_k=5,
                min_score=0.7
            )
            
            similar = []
            for doc in result.search_results:
                similar.append({
                    "incident_id": doc.metadata.get("incident_id", ""),
                    "description": doc.content[:200],
                    "resolution": doc.metadata.get("resolution", ""),
                    "similarity_score": doc.score
                })
            
            return similar
        except Exception as e:
            logger.error(f"[IncidentCorrelator] RAG search failed: {e}")
            return self._keyword_similar_search(description)
    
    def _keyword_similar_search(self, description: str) -> List[Dict[str, Any]]:
        """Simple keyword-based similarity search"""
        desc_words = set(description.lower().split())
        similar = []
        
        for inc in self.recent_incidents:
            inc_words = set(inc.get("description", "").lower().split())
            common = desc_words.intersection(inc_words)
            if len(common) >= 3:
                similarity = len(common) / max(len(desc_words), len(inc_words))
                similar.append({
                    "incident_id": inc["incident_id"],
                    "description": inc.get("description", "")[:200],
                    "similarity_score": similarity
                })
        
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar[:5]
    
    def _analyze_trends(self, category: str, service: str) -> Dict[str, Any]:
        """Analyze incident trends"""
        trends = {
            "category_trend": {},
            "service_trend": {},
            "hourly_distribution": {},
            "insight": ""
        }
        
        # Category trend (last 24 hours)
        category_counts = defaultdict(int)
        service_counts = defaultdict(int)
        hour_counts = defaultdict(int)
        
        for inc in self.recent_incidents:
            cat = inc.get("category", "unknown")
            svc = inc.get("service", "unknown")
            hr = inc.get("created_at", datetime.now()).hour
            
            category_counts[cat] += 1
            service_counts[svc] += 1
            hour_counts[hr] += 1
        
        trends["category_trend"] = dict(category_counts)
        trends["service_trend"] = dict(service_counts)
        trends["hourly_distribution"] = dict(hour_counts)
        
        # Generate insight
        if category and category_counts.get(category, 0) >= 5:
            trends["insight"] = f"Higher than usual incidents for '{category}' category today"
        elif service and service_counts.get(service, 0) >= 5:
            trends["insight"] = f"Elevated incident rate for '{service}' service"
        else:
            # Find peak hour
            if hour_counts:
                peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])
                trends["insight"] = f"Peak incident time today: {peak_hour}:00"
        
        return trends
    
    def _generate_recommendations(
        self,
        correlations: List[CorrelatedIncident],
        patterns: List[IncidentPattern],
        is_major: bool,
        trends: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Major incident recommendations
        if is_major:
            recommendations.extend([
                "⚠️ MAJOR INCIDENT: Consider activating incident management process",
                "Notify affected users proactively",
                "Assign dedicated incident commander",
                "Set up communication channel for updates"
            ])
        
        # Pattern-based recommendations
        for pattern in patterns:
            recommendations.append(f"📊 Pattern detected: {pattern.pattern_name}")
            recommendations.append(f"   Prevention: {pattern.prevention_recommendation}")
        
        # Correlation-based recommendations
        for corr in correlations:
            recommendations.extend(corr.recommended_actions)
        
        # Trend-based recommendations
        insight = trends.get("insight", "")
        if insight:
            recommendations.append(f"📈 Trend insight: {insight}")
        
        # Deduplicate
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10
    
    def _create_major_incident_group(
        self,
        incident_id: str,
        correlations: List[CorrelatedIncident],
        category: str,
        service: str
    ):
        """Create a new major incident correlation group"""
        correlation_id = f"major_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        all_related = []
        all_services = [service] if service else []
        for corr in correlations:
            all_related.extend(corr.related_incident_ids)
            all_services.extend(corr.affected_services)
        
        major_corr = CorrelatedIncident(
            correlation_id=correlation_id,
            primary_incident_id=incident_id,
            related_incident_ids=list(set(all_related)),
            correlation_types=[CorrelationType.SAME_ROOT_CAUSE],
            affected_users_count=sum(c.affected_users_count for c in correlations),
            affected_services=list(set(all_services)),
            start_time=datetime.now(),
            is_major_incident=True,
            severity=IncidentSeverity.HIGH,
            probable_root_cause=f"Major incident affecting {', '.join(set(all_services))}",
            recommended_actions=[
                "Activate incident management process",
                "Assign incident commander",
                "Begin impact assessment",
                "Prepare stakeholder communications"
            ]
        )
        
        self.active_correlations[correlation_id] = major_corr
        logger.warning(f"[IncidentCorrelator] MAJOR INCIDENT DETECTED: {correlation_id}")
    
    def get_active_major_incidents(self) -> List[Dict[str, Any]]:
        """Get list of active major incidents"""
        return [
            {
                "correlation_id": corr.correlation_id,
                "affected_services": corr.affected_services,
                "affected_users": corr.affected_users_count,
                "related_incidents": len(corr.related_incident_ids),
                "start_time": corr.start_time.isoformat(),
                "probable_cause": corr.probable_root_cause
            }
            for corr in self.active_correlations.values()
            if corr.is_major_incident
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get correlation metrics"""
        return {
            "recent_incidents_count": len(self.recent_incidents),
            "active_correlations": len(self.active_correlations),
            "active_major_incidents": sum(1 for c in self.active_correlations.values() if c.is_major_incident),
            "patterns_triggered": sum(p.frequency for p in self.patterns.values()),
            "top_patterns": [
                {"name": p.pattern_name, "frequency": p.frequency}
                for p in sorted(self.patterns.values(), key=lambda x: x.frequency, reverse=True)[:5]
            ]
        }


# Singleton instance
_incident_correlator: Optional[IncidentCorrelator] = None


def get_incident_correlator(rag_engine=None, servicenow_client=None) -> IncidentCorrelator:
    """Get or create the incident correlator singleton."""
    global _incident_correlator
    if _incident_correlator is None:
        _incident_correlator = IncidentCorrelator(rag_engine, servicenow_client)
    return _incident_correlator
