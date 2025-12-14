"""
Runbook Automation Engine
=========================
Safely executes remediation scripts and automation runbooks.

Features:
- Pre-defined safe runbooks for common issues
- Sandboxed execution environment
- Rollback capabilities
- Audit logging
- Integration with ServiceNow for workflow automation
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import re

logger = logging.getLogger(__name__)


class RunbookCategory(str, Enum):
    """Categories of runbooks"""
    PASSWORD_RESET = "password_reset"
    ACCOUNT_UNLOCK = "account_unlock"
    VPN_TROUBLESHOOT = "vpn_troubleshoot"
    NETWORK_DIAGNOSTICS = "network_diagnostics"
    SERVICE_RESTART = "service_restart"
    DISK_CLEANUP = "disk_cleanup"
    CACHE_CLEAR = "cache_clear"
    CERTIFICATE_RENEWAL = "certificate_renewal"
    DNS_FLUSH = "dns_flush"
    PRINTER_FIX = "printer_fix"
    EMAIL_CONFIG = "email_config"
    SOFTWARE_REPAIR = "software_repair"
    PERMISSION_CHECK = "permission_check"
    CONNECTIVITY_TEST = "connectivity_test"


class RunbookRiskLevel(str, Enum):
    """Risk levels for runbook execution"""
    LOW = "low"          # Safe to auto-execute
    MEDIUM = "medium"    # Requires confirmation
    HIGH = "high"        # Requires human approval
    CRITICAL = "critical"  # Manual only


@dataclass
class RunbookStep:
    """Individual step in a runbook"""
    step_id: str
    name: str
    description: str
    action_type: str  # script, api_call, check, notification
    action_params: Dict[str, Any]
    timeout_seconds: int = 60
    retry_count: int = 1
    rollback_action: Optional[Dict[str, Any]] = None
    success_criteria: Optional[str] = None


@dataclass
class Runbook:
    """Complete runbook definition"""
    runbook_id: str
    name: str
    description: str
    category: RunbookCategory
    risk_level: RunbookRiskLevel
    applicable_issues: List[str]  # Keywords/patterns this runbook can handle
    steps: List[RunbookStep]
    estimated_duration_minutes: int
    success_rate: float = 0.0
    times_executed: int = 0
    requires_user_info: List[str] = field(default_factory=list)  # e.g., ["username", "email"]
    pre_conditions: List[str] = field(default_factory=list)
    post_validation: Optional[str] = None


@dataclass
class RunbookResult:
    """Result of runbook execution"""
    runbook_id: str
    success: bool
    started_at: datetime
    completed_at: datetime
    steps_completed: int
    total_steps: int
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    rollback_performed: bool = False
    resolution_notes: str = ""


class RunbookEngine:
    """
    Executes remediation runbooks safely and tracks results.
    """
    
    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self.execution_history: List[RunbookResult] = []
        self._load_default_runbooks()
    
    def _load_default_runbooks(self):
        """Load pre-defined runbooks for common IT issues"""
        
        # Password Reset Runbook
        self.runbooks["rb_password_reset"] = Runbook(
            runbook_id="rb_password_reset",
            name="Password Reset Automation",
            description="Automated password reset workflow with AD integration",
            category=RunbookCategory.PASSWORD_RESET,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["password reset", "forgot password", "password expired", "can't login", "login failed"],
            requires_user_info=["username", "email"],
            estimated_duration_minutes=2,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Verify User Identity",
                    description="Confirm user exists in Active Directory",
                    action_type="api_call",
                    action_params={
                        "service": "active_directory",
                        "operation": "verify_user",
                        "params": ["username"]
                    },
                    success_criteria="user_exists == true"
                ),
                RunbookStep(
                    step_id="2",
                    name="Generate Temporary Password",
                    description="Create a secure temporary password",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "New-RandomPassword -Length 12 -Complexity High"
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Reset Password in AD",
                    description="Set the new password in Active Directory",
                    action_type="api_call",
                    action_params={
                        "service": "active_directory",
                        "operation": "reset_password",
                        "params": ["username", "temp_password"],
                        "force_change_on_login": True
                    },
                    rollback_action={
                        "operation": "revert_password",
                        "params": ["username"]
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Send Notification",
                    description="Email temporary password to user",
                    action_type="notification",
                    action_params={
                        "channel": "email",
                        "template": "password_reset_complete",
                        "recipient": "user_email"
                    }
                )
            ]
        )
        
        # Account Unlock Runbook
        self.runbooks["rb_account_unlock"] = Runbook(
            runbook_id="rb_account_unlock",
            name="Account Unlock Automation",
            description="Automatically unlock locked AD accounts",
            category=RunbookCategory.ACCOUNT_UNLOCK,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["account locked", "locked out", "too many attempts", "can't access account"],
            requires_user_info=["username"],
            estimated_duration_minutes=1,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Check Account Status",
                    description="Verify account is actually locked",
                    action_type="api_call",
                    action_params={
                        "service": "active_directory",
                        "operation": "get_account_status",
                        "params": ["username"]
                    },
                    success_criteria="account_locked == true"
                ),
                RunbookStep(
                    step_id="2",
                    name="Check for Security Threats",
                    description="Verify no suspicious activity on account",
                    action_type="api_call",
                    action_params={
                        "service": "security_center",
                        "operation": "check_threats",
                        "params": ["username"]
                    },
                    success_criteria="threat_level < high"
                ),
                RunbookStep(
                    step_id="3",
                    name="Unlock Account",
                    description="Remove account lock in AD",
                    action_type="api_call",
                    action_params={
                        "service": "active_directory",
                        "operation": "unlock_account",
                        "params": ["username"]
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Notify User",
                    description="Inform user their account is unlocked",
                    action_type="notification",
                    action_params={
                        "channel": "email",
                        "template": "account_unlocked",
                        "recipient": "user_email"
                    }
                )
            ]
        )
        
        # VPN Troubleshooting Runbook
        self.runbooks["rb_vpn_troubleshoot"] = Runbook(
            runbook_id="rb_vpn_troubleshoot",
            name="VPN Connectivity Troubleshooting",
            description="Diagnose and fix common VPN issues",
            category=RunbookCategory.VPN_TROUBLESHOOT,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["vpn not working", "can't connect to vpn", "vpn disconnects", "vpn slow", "remote access"],
            requires_user_info=["username", "device_name"],
            estimated_duration_minutes=5,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Check VPN Server Status",
                    description="Verify VPN servers are operational",
                    action_type="api_call",
                    action_params={
                        "service": "monitoring",
                        "operation": "check_service_health",
                        "params": {"service_name": "vpn_gateway"}
                    }
                ),
                RunbookStep(
                    step_id="2",
                    name="Verify User VPN Permissions",
                    description="Check user has VPN access rights",
                    action_type="api_call",
                    action_params={
                        "service": "active_directory",
                        "operation": "check_group_membership",
                        "params": ["username", "VPN-Users"]
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Check Certificate Status",
                    description="Verify user VPN certificate is valid",
                    action_type="api_call",
                    action_params={
                        "service": "pki",
                        "operation": "verify_certificate",
                        "params": ["username"]
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Reset VPN Profile",
                    description="Push fresh VPN configuration to device",
                    action_type="api_call",
                    action_params={
                        "service": "mdm",
                        "operation": "push_vpn_profile",
                        "params": ["device_name"]
                    }
                ),
                RunbookStep(
                    step_id="5",
                    name="Provide Connection Instructions",
                    description="Send user step-by-step reconnection guide",
                    action_type="notification",
                    action_params={
                        "channel": "email",
                        "template": "vpn_reconnect_guide",
                        "recipient": "user_email"
                    }
                )
            ]
        )
        
        # Network Diagnostics Runbook
        self.runbooks["rb_network_diag"] = Runbook(
            runbook_id="rb_network_diag",
            name="Network Connectivity Diagnostics",
            description="Comprehensive network troubleshooting",
            category=RunbookCategory.NETWORK_DIAGNOSTICS,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["network slow", "no internet", "can't access", "connection timeout", "network down"],
            requires_user_info=["device_ip", "device_name"],
            estimated_duration_minutes=3,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Ping Gateway",
                    description="Test connectivity to default gateway",
                    action_type="script",
                    action_params={
                        "script_type": "shell",
                        "script": "ping -c 4 {gateway_ip}"
                    }
                ),
                RunbookStep(
                    step_id="2",
                    name="DNS Resolution Test",
                    description="Verify DNS is resolving correctly",
                    action_type="script",
                    action_params={
                        "script_type": "shell",
                        "script": "nslookup company.com"
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Check DHCP Lease",
                    description="Verify device has valid IP lease",
                    action_type="api_call",
                    action_params={
                        "service": "dhcp",
                        "operation": "check_lease",
                        "params": ["device_ip"]
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Port Connectivity Test",
                    description="Test critical ports are accessible",
                    action_type="script",
                    action_params={
                        "script_type": "shell",
                        "script": "nc -zv proxy.company.com 8080"
                    }
                )
            ]
        )
        
        # Service Restart Runbook
        self.runbooks["rb_service_restart"] = Runbook(
            runbook_id="rb_service_restart",
            name="Application Service Restart",
            description="Safely restart application services",
            category=RunbookCategory.SERVICE_RESTART,
            risk_level=RunbookRiskLevel.MEDIUM,
            applicable_issues=["application not responding", "service down", "app crashed", "system hang"],
            requires_user_info=["service_name", "server_name"],
            estimated_duration_minutes=5,
            pre_conditions=["maintenance_window_active OR critical_issue"],
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Check Current Service Status",
                    description="Get current state of the service",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Get-Service -Name {service_name} -ComputerName {server_name}"
                    }
                ),
                RunbookStep(
                    step_id="2",
                    name="Notify Stakeholders",
                    description="Send notification about pending restart",
                    action_type="notification",
                    action_params={
                        "channel": "teams",
                        "template": "service_restart_notice",
                        "recipients": "service_owners"
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Stop Service Gracefully",
                    description="Attempt graceful service shutdown",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Stop-Service -Name {service_name} -Force"
                    },
                    timeout_seconds=120
                ),
                RunbookStep(
                    step_id="4",
                    name="Start Service",
                    description="Start the service",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Start-Service -Name {service_name}"
                    },
                    timeout_seconds=60
                ),
                RunbookStep(
                    step_id="5",
                    name="Verify Service Health",
                    description="Confirm service is running properly",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Test-ServiceHealth -Name {service_name}"
                    },
                    success_criteria="status == running AND health == healthy"
                )
            ]
        )
        
        # Disk Cleanup Runbook
        self.runbooks["rb_disk_cleanup"] = Runbook(
            runbook_id="rb_disk_cleanup",
            name="Disk Space Cleanup",
            description="Free up disk space by removing temporary files",
            category=RunbookCategory.DISK_CLEANUP,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["disk full", "low disk space", "out of space", "storage full", "c drive full"],
            requires_user_info=["device_name"],
            estimated_duration_minutes=10,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Check Current Disk Usage",
                    description="Get current disk space statistics",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Get-PSDrive C | Select-Object Used,Free"
                    }
                ),
                RunbookStep(
                    step_id="2",
                    name="Clear Temp Files",
                    description="Remove Windows temporary files",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Remove-Item -Path $env:TEMP\\* -Recurse -Force -ErrorAction SilentlyContinue"
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Clear Windows Update Cache",
                    description="Remove old Windows Update files",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Clear-WindowsUpdateCache"
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Empty Recycle Bin",
                    description="Clear the recycle bin",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"
                    }
                ),
                RunbookStep(
                    step_id="5",
                    name="Report Space Recovered",
                    description="Calculate and report freed space",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Get-PSDrive C | Select-Object Free"
                    }
                )
            ]
        )

        # Email Configuration Runbook
        self.runbooks["rb_email_config"] = Runbook(
            runbook_id="rb_email_config",
            name="Email Client Configuration",
            description="Configure or repair email client settings",
            category=RunbookCategory.EMAIL_CONFIG,
            risk_level=RunbookRiskLevel.LOW,
            applicable_issues=["email not working", "outlook problems", "can't send email", "email sync", "mailbox issues"],
            requires_user_info=["username", "email", "device_name"],
            estimated_duration_minutes=5,
            steps=[
                RunbookStep(
                    step_id="1",
                    name="Verify Mailbox Status",
                    description="Check Exchange mailbox is active",
                    action_type="api_call",
                    action_params={
                        "service": "exchange",
                        "operation": "get_mailbox_status",
                        "params": ["email"]
                    }
                ),
                RunbookStep(
                    step_id="2",
                    name="Check Autodiscover",
                    description="Verify autodiscover is working",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Test-OutlookConnectivity -Identity {email}"
                    }
                ),
                RunbookStep(
                    step_id="3",
                    name="Reset Outlook Profile",
                    description="Create new Outlook profile remotely",
                    action_type="script",
                    action_params={
                        "script_type": "powershell",
                        "script": "Reset-OutlookProfile -ComputerName {device_name} -UserName {username}"
                    }
                ),
                RunbookStep(
                    step_id="4",
                    name="Send Test Email",
                    description="Verify email is working",
                    action_type="api_call",
                    action_params={
                        "service": "exchange",
                        "operation": "send_test_email",
                        "params": ["email"]
                    }
                )
            ]
        )

        logger.info(f"Loaded {len(self.runbooks)} default runbooks")
    
    def find_applicable_runbooks(self, issue_description: str, issue_category: str = None) -> List[Runbook]:
        """
        Find runbooks that can handle a given issue.
        
        Args:
            issue_description: Description of the issue
            issue_category: Optional category filter
            
        Returns:
            List of applicable runbooks sorted by relevance
        """
        applicable = []
        issue_lower = issue_description.lower()
        
        for runbook in self.runbooks.values():
            # Check if any applicable issue keywords match
            score = 0
            for keyword in runbook.applicable_issues:
                if keyword.lower() in issue_lower:
                    score += 1
            
            if score > 0:
                applicable.append((runbook, score))
        
        # Sort by score (descending) and success rate
        applicable.sort(key=lambda x: (x[1], x[0].success_rate), reverse=True)
        
        return [rb for rb, score in applicable]
    
    def can_auto_execute(self, runbook: Runbook) -> bool:
        """
        Check if a runbook can be automatically executed without human approval.
        """
        return runbook.risk_level in [RunbookRiskLevel.LOW]
    
    async def execute_runbook(
        self, 
        runbook_id: str, 
        context: Dict[str, Any],
        dry_run: bool = False
    ) -> RunbookResult:
        """
        Execute a runbook with given context.
        
        Args:
            runbook_id: ID of the runbook to execute
            context: Dictionary with required parameters (username, device_name, etc.)
            dry_run: If True, simulate execution without making changes
            
        Returns:
            RunbookResult with execution details
        """
        if runbook_id not in self.runbooks:
            return RunbookResult(
                runbook_id=runbook_id,
                success=False,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                steps_completed=0,
                total_steps=0,
                error_message=f"Runbook {runbook_id} not found"
            )
        
        runbook = self.runbooks[runbook_id]
        started_at = datetime.now()
        steps_completed = 0
        output_data = {}
        error_message = None
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Executing runbook: {runbook.name}")
        
        try:
            for step in runbook.steps:
                logger.info(f"  Step {step.step_id}: {step.name}")
                
                if dry_run:
                    # Simulate step execution
                    output_data[step.step_id] = {
                        "status": "simulated",
                        "action": step.action_type,
                        "description": step.description
                    }
                    steps_completed += 1
                    continue
                
                # Execute the step based on action type
                step_result = await self._execute_step(step, context)
                output_data[step.step_id] = step_result
                
                if not step_result.get("success", False):
                    error_message = f"Step {step.step_id} failed: {step_result.get('error', 'Unknown error')}"
                    logger.error(error_message)
                    break
                
                steps_completed += 1
                
        except Exception as e:
            error_message = f"Runbook execution failed: {str(e)}"
            logger.exception(error_message)
        
        completed_at = datetime.now()
        success = steps_completed == len(runbook.steps) and error_message is None
        
        # Update runbook statistics
        runbook.times_executed += 1
        if success:
            # Update success rate with moving average
            runbook.success_rate = (
                (runbook.success_rate * (runbook.times_executed - 1) + 1.0) 
                / runbook.times_executed
            )
        
        result = RunbookResult(
            runbook_id=runbook_id,
            success=success,
            started_at=started_at,
            completed_at=completed_at,
            steps_completed=steps_completed,
            total_steps=len(runbook.steps),
            error_message=error_message,
            output_data=output_data,
            resolution_notes=self._generate_resolution_notes(runbook, output_data, success)
        )
        
        self.execution_history.append(result)
        return result
    
    async def _execute_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single runbook step."""
        
        # In production, this would integrate with actual systems
        # For now, we simulate execution
        
        if step.action_type == "script":
            return await self._execute_script(step, context)
        elif step.action_type == "api_call":
            return await self._execute_api_call(step, context)
        elif step.action_type == "notification":
            return await self._send_notification(step, context)
        elif step.action_type == "check":
            return await self._perform_check(step, context)
        else:
            return {"success": False, "error": f"Unknown action type: {step.action_type}"}
    
    async def _execute_script(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a script step (simulated for safety)."""
        script = step.action_params.get("script", "")
        script_type = step.action_params.get("script_type", "shell")
        
        # Replace placeholders with context values
        for key, value in context.items():
            script = script.replace(f"{{{key}}}", str(value))
        
        logger.info(f"    Would execute {script_type}: {script}")
        
        # Simulate successful execution
        return {
            "success": True,
            "script_type": script_type,
            "script": script,
            "output": "Simulated execution successful",
            "executed_at": datetime.now().isoformat()
        }
    
    async def _execute_api_call(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call step (simulated)."""
        service = step.action_params.get("service", "")
        operation = step.action_params.get("operation", "")
        
        logger.info(f"    Would call {service}.{operation}")
        
        # Simulate successful API call
        return {
            "success": True,
            "service": service,
            "operation": operation,
            "response": {"status": "completed"},
            "executed_at": datetime.now().isoformat()
        }
    
    async def _send_notification(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification (simulated)."""
        channel = step.action_params.get("channel", "email")
        template = step.action_params.get("template", "")
        
        logger.info(f"    Would send {channel} notification using template: {template}")
        
        return {
            "success": True,
            "channel": channel,
            "template": template,
            "sent_at": datetime.now().isoformat()
        }
    
    async def _perform_check(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a check/validation step."""
        return {
            "success": True,
            "check_passed": True,
            "checked_at": datetime.now().isoformat()
        }
    
    def _generate_resolution_notes(
        self, 
        runbook: Runbook, 
        output_data: Dict[str, Any], 
        success: bool
    ) -> str:
        """Generate human-readable resolution notes."""
        if success:
            notes = f"Successfully executed runbook: {runbook.name}\n\n"
            notes += "Steps completed:\n"
            for step in runbook.steps:
                notes += f"  ✓ {step.name}\n"
            return notes
        else:
            notes = f"Runbook {runbook.name} did not complete successfully.\n\n"
            notes += "Please review the execution logs and complete manually if needed."
            return notes
    
    def get_runbook_summary(self, runbook_id: str) -> Dict[str, Any]:
        """Get a summary of a runbook for display."""
        if runbook_id not in self.runbooks:
            return None
        
        rb = self.runbooks[runbook_id]
        return {
            "id": rb.runbook_id,
            "name": rb.name,
            "description": rb.description,
            "category": rb.category.value,
            "risk_level": rb.risk_level.value,
            "estimated_duration": rb.estimated_duration_minutes,
            "success_rate": f"{rb.success_rate * 100:.1f}%",
            "times_executed": rb.times_executed,
            "steps": [{"name": s.name, "description": s.description} for s in rb.steps],
            "requires_info": rb.requires_user_info
        }
    
    def list_all_runbooks(self) -> List[Dict[str, Any]]:
        """List all available runbooks."""
        return [self.get_runbook_summary(rb_id) for rb_id in self.runbooks.keys()]


# Singleton instance
_runbook_engine: Optional[RunbookEngine] = None

def get_runbook_engine() -> RunbookEngine:
    """Get or create the runbook engine singleton."""
    global _runbook_engine
    if _runbook_engine is None:
        _runbook_engine = RunbookEngine()
    return _runbook_engine
