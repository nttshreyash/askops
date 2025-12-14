"""
Agent Assist Engine
===================
Provides prescriptive guidance and real-time assistance to human IT agents.

This engine:
1. Analyzes assigned tickets in real-time
2. Suggests resolution steps based on AI + knowledge base
3. Provides relevant KB articles and similar cases
4. Offers copy-paste scripts and commands
5. Tracks agent actions and learns from them
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


class SuggestionPriority(str, Enum):
    """Priority level for suggestions"""
    CRITICAL = "critical"  # Must do first
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class SuggestionType(str, Enum):
    """Types of agent suggestions"""
    DIAGNOSTIC_COMMAND = "diagnostic_command"  # Run this command
    RESOLUTION_STEP = "resolution_step"        # Do this step
    KNOWLEDGE_ARTICLE = "knowledge_article"    # Read this KB
    SIMILAR_TICKET = "similar_ticket"          # Reference this case
    SCRIPT_EXECUTION = "script_execution"      # Execute this script
    ESCALATION = "escalation"                  # Escalate to this team
    CUSTOMER_COMMUNICATION = "customer_communication"  # Send this message


@dataclass
class AgentSuggestion:
    """A single suggestion for the agent"""
    suggestion_id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    action_details: Dict[str, Any]  # Command, script, article link, etc.
    confidence: float
    estimated_time_minutes: int
    reasoning: str
    prerequisites: List[str] = field(default_factory=list)
    related_suggestions: List[str] = field(default_factory=list)


@dataclass
class AgentAssistResponse:
    """Complete response from Agent Assist"""
    ticket_id: str
    issue_summary: str
    diagnosis: str
    suggestions: List[AgentSuggestion]
    similar_tickets: List[Dict[str, Any]]
    kb_articles: List[Dict[str, Any]]
    quick_actions: List[Dict[str, Any]]  # One-click actions
    customer_context: Dict[str, Any]
    priority_assessment: str
    estimated_resolution_time: int
    escalation_recommended: bool
    escalation_team: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentAssist:
    """
    Real-time Agent Assistance Engine
    
    Provides intelligent suggestions and guidance to IT support agents,
    helping them resolve tickets faster and more accurately.
    """
    
    def __init__(self, rag_engine=None, llm_client=None, servicenow_client=None):
        """
        Initialize Agent Assist.
        
        Args:
            rag_engine: RAG engine for knowledge retrieval
            llm_client: Function to query LLM
            servicenow_client: ServiceNow API client for ticket data
        """
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        self.servicenow_client = servicenow_client
        
        # Diagnostic commands library
        self.diagnostic_commands = self._load_diagnostic_commands()
        
        # Resolution scripts library
        self.resolution_scripts = self._load_resolution_scripts()
        
        # Tracking
        self.assist_sessions: List[Dict[str, Any]] = []
        
    def _load_diagnostic_commands(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load common diagnostic commands by category"""
        return {
            "network": [
                {
                    "name": "Test Network Connectivity",
                    "command": "ping -c 4 8.8.8.8 && ping -c 4 google.com",
                    "description": "Test basic internet connectivity",
                    "platform": "linux/mac",
                    "powershell": "Test-NetConnection -ComputerName google.com -Port 443"
                },
                {
                    "name": "Check DNS Resolution",
                    "command": "nslookup google.com && nslookup company.com",
                    "description": "Verify DNS is resolving correctly",
                    "platform": "all",
                    "powershell": "Resolve-DnsName company.com"
                },
                {
                    "name": "Trace Route",
                    "command": "traceroute google.com",
                    "description": "Trace network path to identify issues",
                    "platform": "linux/mac",
                    "powershell": "Test-NetConnection -ComputerName google.com -TraceRoute"
                },
                {
                    "name": "Check IP Configuration",
                    "command": "ifconfig || ip addr",
                    "description": "View network interface configuration",
                    "platform": "linux/mac",
                    "powershell": "Get-NetIPConfiguration"
                },
                {
                    "name": "Check Network Services",
                    "command": "netstat -tuln",
                    "description": "Show listening ports and connections",
                    "platform": "linux/mac",
                    "powershell": "Get-NetTCPConnection | Where-Object State -eq 'Listen'"
                }
            ],
            "active_directory": [
                {
                    "name": "Check User Account Status",
                    "command": "Get-ADUser -Identity {username} -Properties *",
                    "description": "Get full AD user account details",
                    "platform": "powershell",
                    "requires": ["ActiveDirectory module"]
                },
                {
                    "name": "Check Account Lockout",
                    "command": "Get-ADUser -Identity {username} -Properties LockedOut,LockoutTime,BadLogonCount",
                    "description": "Check if account is locked out",
                    "platform": "powershell",
                    "requires": ["ActiveDirectory module"]
                },
                {
                    "name": "Unlock User Account",
                    "command": "Unlock-ADAccount -Identity {username}",
                    "description": "Unlock a locked AD account",
                    "platform": "powershell",
                    "requires": ["ActiveDirectory module", "Admin rights"]
                },
                {
                    "name": "Reset User Password",
                    "command": "Set-ADAccountPassword -Identity {username} -Reset -NewPassword (ConvertTo-SecureString -AsPlainText 'TempP@ssw0rd!' -Force)",
                    "description": "Reset AD user password",
                    "platform": "powershell",
                    "requires": ["ActiveDirectory module", "Admin rights"]
                },
                {
                    "name": "Check Group Membership",
                    "command": "Get-ADPrincipalGroupMembership -Identity {username} | Select-Object Name",
                    "description": "List all groups user belongs to",
                    "platform": "powershell",
                    "requires": ["ActiveDirectory module"]
                }
            ],
            "exchange": [
                {
                    "name": "Check Mailbox Status",
                    "command": "Get-Mailbox -Identity {email} | Select-Object DisplayName,PrimarySmtpAddress,Database,ProhibitSendQuota",
                    "description": "Get mailbox details and quota",
                    "platform": "powershell",
                    "requires": ["Exchange Online module"]
                },
                {
                    "name": "Check Mail Flow",
                    "command": "Get-MessageTrace -SenderAddress {email} -StartDate (Get-Date).AddDays(-2) -EndDate (Get-Date)",
                    "description": "Check recent email delivery status",
                    "platform": "powershell",
                    "requires": ["Exchange Online module"]
                },
                {
                    "name": "Test Autodiscover",
                    "command": "Test-OutlookConnectivity -Identity {email}",
                    "description": "Test Outlook autodiscover configuration",
                    "platform": "powershell",
                    "requires": ["Exchange Online module"]
                }
            ],
            "system": [
                {
                    "name": "Check System Info",
                    "command": "systeminfo",
                    "description": "Get Windows system information",
                    "platform": "windows",
                    "bash": "uname -a && cat /etc/os-release"
                },
                {
                    "name": "Check Disk Space",
                    "command": "Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{N='Used(GB)';E={[math]::Round($_.Used/1GB,2)}},@{N='Free(GB)';E={[math]::Round($_.Free/1GB,2)}}",
                    "description": "Check disk usage",
                    "platform": "powershell",
                    "bash": "df -h"
                },
                {
                    "name": "Check Running Processes",
                    "command": "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name,CPU,WorkingSet",
                    "description": "Show top 10 CPU-consuming processes",
                    "platform": "powershell",
                    "bash": "ps aux --sort=-%cpu | head -11"
                },
                {
                    "name": "Check Event Logs",
                    "command": "Get-EventLog -LogName Application -Newest 20 -EntryType Error,Warning",
                    "description": "Get recent error events",
                    "platform": "powershell",
                    "bash": "journalctl -p err -n 20"
                },
                {
                    "name": "Check Services",
                    "command": "Get-Service | Where-Object Status -eq 'Stopped' | Where-Object StartType -eq 'Automatic'",
                    "description": "Find stopped services that should be running",
                    "platform": "powershell",
                    "bash": "systemctl list-units --type=service --state=failed"
                }
            ],
            "vpn": [
                {
                    "name": "Check VPN Connection Status",
                    "command": "Get-VpnConnection | Select-Object Name,ServerAddress,ConnectionStatus",
                    "description": "Show VPN connection status",
                    "platform": "powershell"
                },
                {
                    "name": "Reset VPN Connection",
                    "command": "rasdial /disconnect && rasdial '{vpn_name}' {username} {password}",
                    "description": "Disconnect and reconnect VPN",
                    "platform": "windows"
                },
                {
                    "name": "Check VPN Routes",
                    "command": "route print",
                    "description": "Show routing table",
                    "platform": "windows",
                    "bash": "ip route"
                }
            ]
        }
    
    def _load_resolution_scripts(self) -> Dict[str, Dict[str, Any]]:
        """Load resolution script templates"""
        return {
            "clear_outlook_cache": {
                "name": "Clear Outlook Cache",
                "description": "Clears Outlook local cache to fix sync issues",
                "script_type": "powershell",
                "script": '''
# Clear Outlook local cache
$OutlookPath = "$env:LOCALAPPDATA\\Microsoft\\Outlook"
$RoamCachePath = "$OutlookPath\\RoamCache"

# Close Outlook first
Get-Process outlook -ErrorAction SilentlyContinue | Stop-Process -Force

# Wait for Outlook to close
Start-Sleep -Seconds 3

# Clear RoamCache
if (Test-Path $RoamCachePath) {
    Remove-Item "$RoamCachePath\\*" -Recurse -Force
    Write-Host "RoamCache cleared successfully"
}

Write-Host "Please restart Outlook"
''',
                "requires_admin": False,
                "estimated_time": 5
            },
            "flush_dns": {
                "name": "Flush DNS Cache",
                "description": "Clears local DNS cache to fix name resolution issues",
                "script_type": "powershell",
                "script": '''
# Flush DNS cache
ipconfig /flushdns
Clear-DnsClientCache

# Reset Winsock
netsh winsock reset

Write-Host "DNS cache flushed. Restart may be required for Winsock reset."
''',
                "requires_admin": True,
                "estimated_time": 2
            },
            "reset_network_stack": {
                "name": "Reset Network Stack",
                "description": "Full network stack reset for connectivity issues",
                "script_type": "powershell",
                "script": '''
# Requires admin privileges
# Reset network stack
netsh winsock reset
netsh int ip reset
ipconfig /release
ipconfig /flushdns
ipconfig /renew

Write-Host "Network stack reset complete. Please restart the computer."
''',
                "requires_admin": True,
                "estimated_time": 5
            },
            "clear_temp_files": {
                "name": "Clear Temporary Files",
                "description": "Clears Windows temp files to free disk space",
                "script_type": "powershell",
                "script": '''
# Clear temp files
$TempFolders = @(
    $env:TEMP,
    "$env:LOCALAPPDATA\\Temp",
    "C:\\Windows\\Temp"
)

$TotalFreed = 0

foreach ($folder in $TempFolders) {
    if (Test-Path $folder) {
        $size = (Get-ChildItem $folder -Recurse -ErrorAction SilentlyContinue | 
                 Measure-Object Length -Sum).Sum
        Remove-Item "$folder\\*" -Recurse -Force -ErrorAction SilentlyContinue
        $TotalFreed += $size
    }
}

$FreedMB = [math]::Round($TotalFreed / 1MB, 2)
Write-Host "Freed $FreedMB MB of temporary files"
''',
                "requires_admin": False,
                "estimated_time": 3
            },
            "repair_office": {
                "name": "Repair Office Installation",
                "description": "Quick repair of Microsoft Office",
                "script_type": "powershell",
                "script": '''
# Run Office Quick Repair
$OfficePath = "C:\\Program Files\\Microsoft Office\\root\\Office16"
if (Test-Path $OfficePath) {
    Start-Process "$OfficePath\\OFFICECLICKTORUN.EXE" -ArgumentList "scenario=Repair platform=x64 culture=en-us" -Wait
    Write-Host "Office repair initiated"
} else {
    Write-Host "Office installation not found at expected path"
}
''',
                "requires_admin": True,
                "estimated_time": 15
            }
        }
    
    async def get_assistance(
        self,
        ticket_id: str,
        issue_description: str,
        category: str = None,
        priority: str = None,
        customer_info: Dict[str, Any] = None,
        history: List[Dict[str, Any]] = None
    ) -> AgentAssistResponse:
        """
        Get real-time assistance for a ticket.
        
        Args:
            ticket_id: Ticket ID
            issue_description: Full issue description
            category: Issue category (network, software, etc.)
            priority: Ticket priority
            customer_info: Customer/user information
            history: Previous work notes and actions
            
        Returns:
            AgentAssistResponse with suggestions and guidance
        """
        logger.info(f"[AgentAssist] Generating assistance for ticket {ticket_id}")
        
        # Step 1: Analyze the issue
        analysis = await self._analyze_issue(issue_description, category)
        
        # Step 2: Get relevant KB articles
        kb_articles = await self._get_kb_articles(issue_description)
        
        # Step 3: Find similar resolved tickets
        similar_tickets = await self._find_similar_tickets(issue_description)
        
        # Step 4: Generate suggestions
        suggestions = self._generate_suggestions(
            analysis,
            category,
            similar_tickets,
            kb_articles,
            history
        )
        
        # Step 5: Generate quick actions
        quick_actions = self._get_quick_actions(category, analysis)
        
        # Step 6: Assess if escalation needed
        escalation_needed, escalation_team = self._assess_escalation(
            analysis, 
            priority,
            history
        )
        
        # Step 7: Estimate resolution time
        est_time = self._estimate_time(analysis, suggestions)
        
        response = AgentAssistResponse(
            ticket_id=ticket_id,
            issue_summary=self._summarize_issue(issue_description),
            diagnosis=analysis.get("diagnosis", ""),
            suggestions=suggestions,
            similar_tickets=similar_tickets,
            kb_articles=kb_articles,
            quick_actions=quick_actions,
            customer_context=customer_info or {},
            priority_assessment=self._assess_priority(analysis, priority),
            estimated_resolution_time=est_time,
            escalation_recommended=escalation_needed,
            escalation_team=escalation_team,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "analysis": analysis
            }
        )
        
        # Track this session
        self.assist_sessions.append({
            "ticket_id": ticket_id,
            "timestamp": datetime.now().isoformat(),
            "suggestions_count": len(suggestions),
            "escalation_recommended": escalation_needed
        })
        
        return response
    
    async def _analyze_issue(
        self, 
        issue_description: str,
        category: str = None
    ) -> Dict[str, Any]:
        """Analyze the issue using AI"""
        
        # Keyword-based analysis first
        analysis = {
            "keywords": self._extract_keywords(issue_description),
            "likely_category": category or self._guess_category(issue_description),
            "complexity": "low",
            "requires_remote_access": False,
            "requires_admin": False
        }
        
        # Determine complexity
        complex_keywords = ["multiple", "recurring", "intermittent", "all users", "production", "critical"]
        if any(kw in issue_description.lower() for kw in complex_keywords):
            analysis["complexity"] = "high"
        elif len(issue_description) > 500:
            analysis["complexity"] = "medium"
        
        # Check if remote access needed
        remote_keywords = ["install", "uninstall", "registry", "driver", "bios", "boot"]
        if any(kw in issue_description.lower() for kw in remote_keywords):
            analysis["requires_remote_access"] = True
        
        # Check if admin needed
        admin_keywords = ["admin", "permission", "install", "uninstall", "service", "restart server"]
        if any(kw in issue_description.lower() for kw in admin_keywords):
            analysis["requires_admin"] = True
        
        # Get AI diagnosis if available
        if self.llm_client:
            try:
                prompt = f"""Analyze this IT support ticket and provide:
1. Root cause hypothesis (1-2 sentences)
2. Key diagnostic checks needed
3. Most likely resolution path

Ticket: {issue_description}

Respond in JSON format:
{{"diagnosis": "...", "diagnostic_checks": ["..."], "resolution_path": "..."}}
"""
                response = self.llm_client(prompt, temperature=0.1)
                # Try to parse JSON from response
                try:
                    parsed = json.loads(response)
                    analysis.update(parsed)
                except json.JSONDecodeError:
                    # Extract key info even if not valid JSON
                    analysis["diagnosis"] = response[:500]
            except Exception as e:
                logger.error(f"[AgentAssist] AI analysis failed: {e}")
                analysis["diagnosis"] = "AI analysis unavailable"
        else:
            analysis["diagnosis"] = f"Issue appears to be related to {analysis['likely_category']}"
        
        return analysis
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Common IT keywords to look for
        it_keywords = [
            "password", "reset", "locked", "vpn", "email", "outlook",
            "network", "slow", "error", "crash", "install", "access",
            "permission", "printer", "wifi", "internet", "login",
            "sharepoint", "teams", "zoom", "software", "hardware"
        ]
        
        text_lower = text.lower()
        found = [kw for kw in it_keywords if kw in text_lower]
        return found
    
    def _guess_category(self, issue_description: str) -> str:
        """Guess issue category from description"""
        text = issue_description.lower()
        
        category_keywords = {
            "network": ["network", "internet", "wifi", "vpn", "connection", "firewall"],
            "email": ["email", "outlook", "mailbox", "calendar", "exchange"],
            "access": ["password", "login", "locked", "permission", "access", "reset"],
            "software": ["install", "software", "application", "crash", "error", "update"],
            "hardware": ["printer", "monitor", "keyboard", "mouse", "laptop", "computer", "slow"]
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in text for kw in keywords):
                return category
        
        return "general"
    
    async def _get_kb_articles(self, issue_description: str) -> List[Dict[str, Any]]:
        """Get relevant KB articles from RAG"""
        if not self.rag_engine:
            return []
        
        try:
            result = self.rag_engine.retrieve(
                query=issue_description,
                top_k=5,
                min_score=0.65,
                doc_type="kb_article"
            )
            
            articles = []
            for doc in result.search_results:
                articles.append({
                    "title": doc.metadata.get("title", "KB Article"),
                    "article_id": doc.metadata.get("article_id", ""),
                    "summary": doc.content[:300],
                    "url": doc.metadata.get("url", ""),
                    "relevance_score": doc.score,
                    "category": doc.metadata.get("category", "")
                })
            
            return articles
        except Exception as e:
            logger.error(f"[AgentAssist] KB search failed: {e}")
            return []
    
    async def _find_similar_tickets(self, issue_description: str) -> List[Dict[str, Any]]:
        """Find similar resolved tickets"""
        if not self.rag_engine:
            return []
        
        try:
            result = self.rag_engine.retrieve(
                query=issue_description,
                top_k=5,
                min_score=0.6,
                doc_type="resolved_ticket"
            )
            
            tickets = []
            for doc in result.search_results:
                tickets.append({
                    "ticket_number": doc.metadata.get("ticket_number", ""),
                    "short_description": doc.metadata.get("short_description", ""),
                    "resolution": doc.metadata.get("resolution", doc.content[:300]),
                    "resolution_time": doc.metadata.get("resolution_time", ""),
                    "assigned_group": doc.metadata.get("assigned_group", ""),
                    "similarity_score": doc.score
                })
            
            return tickets
        except Exception as e:
            logger.error(f"[AgentAssist] Similar ticket search failed: {e}")
            return []
    
    def _generate_suggestions(
        self,
        analysis: Dict[str, Any],
        category: str,
        similar_tickets: List[Dict[str, Any]],
        kb_articles: List[Dict[str, Any]],
        history: List[Dict[str, Any]] = None
    ) -> List[AgentSuggestion]:
        """Generate prioritized suggestions for the agent"""
        suggestions = []
        suggestion_counter = 0
        
        # 1. Diagnostic commands based on category
        cat_key = category or analysis.get("likely_category", "system")
        if cat_key in self.diagnostic_commands:
            for cmd in self.diagnostic_commands[cat_key][:3]:
                suggestion_counter += 1
                suggestions.append(AgentSuggestion(
                    suggestion_id=f"sug_{suggestion_counter}",
                    suggestion_type=SuggestionType.DIAGNOSTIC_COMMAND,
                    priority=SuggestionPriority.HIGH,
                    title=cmd["name"],
                    description=cmd["description"],
                    action_details={
                        "command": cmd.get("powershell") or cmd.get("command"),
                        "platform": cmd.get("platform", "all"),
                        "requires": cmd.get("requires", [])
                    },
                    confidence=0.85,
                    estimated_time_minutes=2,
                    reasoning="Standard diagnostic for this issue type"
                ))
        
        # 2. Resolution steps from similar tickets
        if similar_tickets:
            for ticket in similar_tickets[:2]:
                suggestion_counter += 1
                suggestions.append(AgentSuggestion(
                    suggestion_id=f"sug_{suggestion_counter}",
                    suggestion_type=SuggestionType.SIMILAR_TICKET,
                    priority=SuggestionPriority.HIGH,
                    title=f"Similar Case: {ticket.get('ticket_number', 'Unknown')}",
                    description=ticket.get('short_description', ''),
                    action_details={
                        "resolution": ticket.get('resolution', ''),
                        "ticket_number": ticket.get('ticket_number', ''),
                        "resolution_time": ticket.get('resolution_time', '')
                    },
                    confidence=ticket.get('similarity_score', 0.7),
                    estimated_time_minutes=int(ticket.get('resolution_time', 15)),
                    reasoning=f"Similarity score: {ticket.get('similarity_score', 0)*100:.0f}%"
                ))
        
        # 3. KB articles
        if kb_articles:
            for article in kb_articles[:2]:
                suggestion_counter += 1
                suggestions.append(AgentSuggestion(
                    suggestion_id=f"sug_{suggestion_counter}",
                    suggestion_type=SuggestionType.KNOWLEDGE_ARTICLE,
                    priority=SuggestionPriority.MEDIUM,
                    title=article.get('title', 'KB Article'),
                    description=article.get('summary', '')[:200],
                    action_details={
                        "article_id": article.get('article_id', ''),
                        "url": article.get('url', '')
                    },
                    confidence=article.get('relevance_score', 0.7),
                    estimated_time_minutes=5,
                    reasoning="Relevant knowledge base article"
                ))
        
        # 4. Resolution scripts
        script_keywords = {
            "clear_outlook_cache": ["outlook", "email", "sync", "cache"],
            "flush_dns": ["dns", "network", "name resolution", "can't access"],
            "reset_network_stack": ["network", "internet", "connection"],
            "clear_temp_files": ["slow", "disk", "space", "performance"],
            "repair_office": ["office", "word", "excel", "outlook", "crash"]
        }
        
        issue_text = analysis.get("diagnosis", "") + " " + " ".join(analysis.get("keywords", []))
        issue_lower = issue_text.lower()
        
        for script_id, keywords in script_keywords.items():
            if any(kw in issue_lower for kw in keywords):
                script = self.resolution_scripts.get(script_id)
                if script:
                    suggestion_counter += 1
                    suggestions.append(AgentSuggestion(
                        suggestion_id=f"sug_{suggestion_counter}",
                        suggestion_type=SuggestionType.SCRIPT_EXECUTION,
                        priority=SuggestionPriority.MEDIUM,
                        title=script["name"],
                        description=script["description"],
                        action_details={
                            "script": script["script"],
                            "script_type": script["script_type"],
                            "requires_admin": script.get("requires_admin", False)
                        },
                        confidence=0.75,
                        estimated_time_minutes=script.get("estimated_time", 5),
                        reasoning="Automated resolution script for this issue type"
                    ))
        
        # Sort by priority and confidence
        priority_order = {
            SuggestionPriority.CRITICAL: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3,
            SuggestionPriority.INFORMATIONAL: 4
        }
        
        suggestions.sort(key=lambda s: (priority_order[s.priority], -s.confidence))
        
        return suggestions
    
    def _get_quick_actions(
        self, 
        category: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get one-click quick actions for the agent"""
        actions = []
        
        # Base actions always available
        actions.append({
            "action_id": "qa_1",
            "name": "Check User Status",
            "description": "Quick check of AD account status",
            "icon": "user-check",
            "command": "Get-ADUser -Identity {username} -Properties LockedOut,Enabled,PasswordExpired",
            "category": "diagnostic"
        })
        
        # Category-specific actions
        if category in ["access", "password"]:
            actions.extend([
                {
                    "action_id": "qa_2",
                    "name": "Unlock Account",
                    "description": "Unlock AD account",
                    "icon": "unlock",
                    "command": "Unlock-ADAccount -Identity {username}",
                    "category": "resolution",
                    "requires_confirmation": True
                },
                {
                    "action_id": "qa_3",
                    "name": "Reset Password",
                    "description": "Generate and set temporary password",
                    "icon": "key",
                    "command": "Set-ADAccountPassword -Identity {username} -Reset",
                    "category": "resolution",
                    "requires_confirmation": True
                }
            ])
        
        if category in ["network", "vpn"]:
            actions.extend([
                {
                    "action_id": "qa_4",
                    "name": "Test Connectivity",
                    "description": "Run network diagnostics",
                    "icon": "wifi",
                    "command": "Test-NetConnection -ComputerName {target}",
                    "category": "diagnostic"
                },
                {
                    "action_id": "qa_5",
                    "name": "Flush DNS",
                    "description": "Clear DNS cache remotely",
                    "icon": "refresh",
                    "command": "Clear-DnsClientCache",
                    "category": "resolution"
                }
            ])
        
        if category in ["email", "software"]:
            actions.extend([
                {
                    "action_id": "qa_6",
                    "name": "Check Mailbox",
                    "description": "Get mailbox status",
                    "icon": "mail",
                    "command": "Get-Mailbox -Identity {email}",
                    "category": "diagnostic"
                },
                {
                    "action_id": "qa_7",
                    "name": "Clear Outlook Cache",
                    "description": "Clear Outlook local cache",
                    "icon": "trash",
                    "script_id": "clear_outlook_cache",
                    "category": "resolution"
                }
            ])
        
        return actions
    
    def _assess_escalation(
        self,
        analysis: Dict[str, Any],
        priority: str,
        history: List[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Assess if escalation is needed"""
        escalation_needed = False
        escalation_team = None
        
        # Check priority
        if priority in ["1 - Critical", "2 - High", "critical", "high"]:
            # High priority tickets might need escalation
            if analysis.get("complexity") == "high":
                escalation_needed = True
                escalation_team = "Tier 2 Support"
        
        # Check if requires special access
        if analysis.get("requires_admin"):
            escalation_team = "System Administration"
        
        # Check history for prolonged handling
        if history:
            # If ticket has been worked on for more than 2 hours, suggest escalation
            if len(history) > 5:
                escalation_needed = True
                escalation_team = escalation_team or "Tier 2 Support"
        
        # Category-based escalation
        category = analysis.get("likely_category", "")
        if category == "network" and "firewall" in str(analysis.get("keywords", [])):
            escalation_team = "Network Security"
        elif category == "hardware" and any(kw in str(analysis) for kw in ["server", "production"]):
            escalation_team = "Infrastructure Team"
        
        return escalation_needed, escalation_team
    
    def _estimate_time(
        self,
        analysis: Dict[str, Any],
        suggestions: List[AgentSuggestion]
    ) -> int:
        """Estimate total resolution time in minutes"""
        base_time = 10  # Base handling time
        
        # Add complexity factor
        complexity = analysis.get("complexity", "low")
        if complexity == "high":
            base_time += 30
        elif complexity == "medium":
            base_time += 15
        
        # Add time from suggestions
        if suggestions:
            # Take the minimum time from similar tickets or use suggestion times
            similar_times = [
                s.action_details.get("resolution_time", 0) 
                for s in suggestions 
                if s.suggestion_type == SuggestionType.SIMILAR_TICKET
            ]
            if similar_times and similar_times[0]:
                return max(int(similar_times[0]), 10)
            
            # Otherwise estimate from suggestion times
            suggestion_time = sum(s.estimated_time_minutes for s in suggestions[:3])
            base_time = max(base_time, suggestion_time)
        
        return min(base_time, 60)  # Cap at 60 minutes for Tier 1
    
    def _summarize_issue(self, issue_description: str) -> str:
        """Create a brief summary of the issue"""
        # First sentence or first 100 chars
        sentences = issue_description.split('.')
        if sentences:
            first = sentences[0].strip()
            if len(first) < 100:
                return first
        return issue_description[:100] + "..." if len(issue_description) > 100 else issue_description
    
    def _assess_priority(self, analysis: Dict[str, Any], current_priority: str) -> str:
        """Assess if priority should be adjusted"""
        if current_priority in ["1 - Critical", "critical"]:
            return "Correct - Critical priority appropriate"
        
        complexity = analysis.get("complexity", "low")
        
        if complexity == "high" and current_priority in ["4 - Low", "5 - Planning", "low"]:
            return "Consider Upgrading - Issue appears more complex than priority indicates"
        
        return f"Current priority ({current_priority}) appears appropriate"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Agent Assist usage metrics"""
        return {
            "total_sessions": len(self.assist_sessions),
            "escalations_recommended": sum(1 for s in self.assist_sessions if s.get("escalation_recommended")),
            "avg_suggestions_per_session": (
                sum(s.get("suggestions_count", 0) for s in self.assist_sessions) / len(self.assist_sessions)
                if self.assist_sessions else 0
            ),
            "recent_sessions": self.assist_sessions[-10:]
        }


# Singleton instance
_agent_assist: Optional[AgentAssist] = None


def get_agent_assist(rag_engine=None, llm_client=None, servicenow_client=None) -> AgentAssist:
    """Get or create the Agent Assist singleton."""
    global _agent_assist
    if _agent_assist is None:
        _agent_assist = AgentAssist(rag_engine, llm_client, servicenow_client)
    return _agent_assist
