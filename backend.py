# --- RAG Engine Integration ---
# Import RAG components for knowledge-enhanced responses
try:
    from rag_engine import RAGEngine, get_rag_engine, get_knowledge_learner
    from rag_engine.api import router as rag_router
    from rag_engine.vector_store import DocumentType
    RAG_ENABLED = True
    print("[RAG] RAG Engine loaded successfully")
except ImportError as e:
    print(f"[RAG] RAG Engine not available: {e}")
    RAG_ENABLED = False
    rag_router = None

# --- Intelligent ITOps Engine Integration ---
# Import ITOps components for auto-resolution, prescriptive analytics, and agent assist
try:
    from itops_engine import (
        itops_router,
        get_auto_resolver,
        get_agent_assist,
        get_triage_engine,
        get_incident_correlator,
        get_learning_engine,
        get_runbook_engine
    )
    ITOPS_ENABLED = True
    print("[ITOps] Intelligent ITOps Engine loaded successfully")
except ImportError as e:
    print(f"[ITOps] ITOps Engine not available: {e}")
    ITOPS_ENABLED = False
    itops_router = None

# --- Agentic LLM Orchestration ---
# Master system prompt for all agents (keeps enterprise constraints, safety, and expected behavior)
MASTER_PROMPT = """
You are AskGen, an enterprise ITSM assistant. Follow these global rules for every agent call:
- Be professional, concise, and friendly.
- Always prioritize troubleshooting-first. Provide at most 1-2 actionable steps a user can perform.
- Never suggest actions that require admin privileges, hardware replacement, or access the user does not have.
- If an issue cannot be resolved by the user, respond EXACTLY with ESCALATE where the agent expects it.
- Preserve user privacy: do not include or expose secrets, credentials, or PII in responses.
- When returning structured output (classification, ticket payloads), respond in strict JSON only as requested.
- Use session_state and chat_history context to avoid repeating suggestions and to include troubleshooting history when escalating.
- If asked to create tickets, call ServiceNow and return the ticket id/number and status in a structured format.
- Important: If the user mentions access requests (SharePoint, file/folder access, permissions), software installation, or provisioning, treat the intent as a 'request' by default. Ask specific clarifying questions (which folder or URL, which software & version, who needs access) before creating a ticket. Do NOT default to incident for access/provisioning intents.
"""

import os
import subprocess
import shutil
from typing import Tuple
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import traceback
import socket
import ssl as _ssl
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time
import urllib3
import enum


# --- Small compatibility helpers (safe, minimal implementations) ---
def get_original_user_problem(session_state: dict, message: str) -> str:
    """Return the original user problem text stored in session_state or the current message as a fallback."""
    try:
        return session_state.get("original_problem") or (message or "")
    except Exception:
        return message or ""


def is_non_it_query(message: str) -> bool:
    """Lightweight heuristic: returns True for obvious non-IT queries (weather, general knowledge).
    This is a conservative helper; if unsure, it returns False so LLM classifier can decide.
    """
    if not message:
        return False
    msg = message.lower()
    non_it_triggers = ["weather", "who is", "what is", "define", "capital of", "translate"]
    return any(t in msg for t in non_it_triggers)


def get_request_language(request, data) -> str:
    """Determine language preference from request JSON or Accept-Language header. Returns a short code like 'en'."""
    try:
        lang = (data.get("language") if isinstance(data, dict) else None) or request.headers.get("Accept-Language") or "en"
        # take primary subtag
        if isinstance(lang, str) and lang:
            return lang.split(",")[0].split("-")[0]
    except Exception:
        pass
    return "en"

def agentic_self_service(user_message, session_state, allow_troubleshooting=True):
    """
    LLM agent for self-service troubleshooting. Returns response string.
    Uses MASTER_PROMPT as system message and includes a short agent instruction + compact context.
    
    When RAG is enabled, first searches the knowledge base for relevant articles/tickets
    and includes them in the context for better responses.
    """
    chat_history = session_state.get("chat_history", [])
    
    # --- RAG Enhancement: Search knowledge base for relevant context ---
    rag_context = ""
    rag_sources = []
    if RAG_ENABLED and allow_troubleshooting:
        try:
            rag_engine = get_rag_engine()
            # Search for similar issues and KB articles
            rag_result = rag_engine.retrieve(
                query=user_message,
                top_k=3,
                min_score=0.65
            )
            if rag_result.has_context:
                rag_context = f"\n\n## Relevant Knowledge Base Context:\n{rag_result.context_text}"
                rag_sources = rag_result.sources
                # Store sources in session_state for potential citation
                session_state['rag_sources'] = rag_sources
                session_state['rag_confidence'] = rag_result.confidence.value
                print(f"[RAG] Found {len(rag_result.search_results)} relevant documents (top score: {rag_result.top_score:.2f})")
        except Exception as e:
            print(f"[RAG] Knowledge retrieval failed: {e}")
    
    # Short agent instruction scopes the master prompt
    agent_instr = (
        "Agent: Troubleshooting — Provide 1-2 clear, actionable steps the user can perform now. "
        "If the problem cannot be fixed by the user, reply EXACTLY with ESCALATE. Do NOT suggest admin-only actions."
    )
    
    # Enhanced instruction when RAG context is available
    if rag_context:
        agent_instr = (
            "Agent: Troubleshooting (Knowledge-Enhanced) — You have been provided with relevant knowledge base articles and past resolved tickets. "
            "Use this context to provide accurate, specific troubleshooting steps. "
            "Provide 1-2 clear, actionable steps the user can perform now. "
            "If the context contains a direct solution, guide the user through it. "
            "If the problem cannot be fixed by the user, reply EXACTLY with ESCALATE. "
            "Do NOT suggest admin-only actions. Cite sources when applicable (e.g., 'According to KB article...')."
        )
    
    # Compact context to help the LLM avoid repetition
    context = f"SESSION_STATE: {session_state}; RECENT_HISTORY: {chat_history[-6:]}{rag_context}"
    if allow_troubleshooting:
        prompt = f"{agent_instr}\nCONTEXT: {context}\nUser: {user_message}"
    else:
        # If troubleshooting not allowed, instruct the agent to escalate
        agent_instr_no = (
            "Agent: Troubleshooting (disabled) — Do NOT provide troubleshooting steps. "
            "Respond with: 'Your issue will be forwarded to IT support. Please wait for assistance.'"
        )
        prompt = f"{agent_instr_no}\nCONTEXT: {context}\nUser: {user_message}"
    response = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.2)
    # Defensive sanitization: if the LLM returned a JSON object that contains
    # ticket-like fields (ticket_id, number, sys_id, request_number, INC/REQ/RITM patterns),
    # do NOT surface a fabricated ticket id from the Troubleshooting agent. Convert such
    # outputs to the ESCALATE sentinel so the orchestrator/ticketing node handles creation.
    try:
        # Try to extract a JSON object from the model output
        m = re.search(r"\{[\s\S]*?\}", response)
        if m:
            try:
                parsed = json.loads(m.group(0))
                # Lowercase keys for heuristic checks
                keys = [k.lower() for k in (parsed.keys() if isinstance(parsed, dict) else [])]
                # Check keys or values that suggest a ticket was fabricated
                tickety_keys = any(k for k in keys if any(tok in k for tok in ("ticket", "ticket_id", "number", "sys_id", "request")))
                tickety_vals = False
                for v in (parsed.values() if isinstance(parsed, dict) else []):
                    try:
                        sval = str(v)
                        if re.search(r"\b(INC|REQ|RITM)\d+", sval, re.IGNORECASE):
                            tickety_vals = True
                            break
                    except Exception:
                        continue
                if tickety_keys or tickety_vals:
                    # Replace with sentinel; do not return the JSON containing ticket identifiers
                    return "ESCALATE"
            except Exception:
                # If JSON parse fails, continue with original response
                pass
    except Exception:
        pass
    return response


def agentic_classification(user_problem):
    """
    LLM agent for ITSM ticket classification. Handles greetings/small talk gracefully.
    Returns dict with category, subcategory, urgency, type (request|incident), and optionally a 'greeting' flag.
    Uses MASTER_PROMPT as system message for consistent global rules.
    """
    # First, use LLM to check if the message is a greeting/small talk
    greeting_check_instr = (
        "Agent: Greeting-Detector — Reply ONLY with YES if the message is a greeting/small talk (hi/hello/thanks/etc.). "
        "Reply ONLY with NO if it is a real IT issue/request."
    )
    greeting_context = f"RECENT_HISTORY: {user_problem[:200]}"
    greeting_check_prompt = f"{greeting_check_instr}\nCONTEXT: {greeting_context}\nUser: {user_problem}"
    greeting_result = query_azure_openai(greeting_check_prompt, system=MASTER_PROMPT, temperature=0.0).strip().upper()
    if greeting_result.startswith("YES"):
        return {"greeting": True}

    # Quick check: is this an IT-related support request? If not, return a not_it flag so orchestrator can short-circuit.
    it_check_instr = (
        "Agent: IT-Related-Detector — Reply ONLY with YES if the message is an IT support request/question (software, hardware, network, access, permissions, VPN, email, account, provisioning). "
        "Reply ONLY with NO if it's general knowledge, personal questions, or anything unrelated to IT support."
    )
    it_check_prompt = f"{it_check_instr}\nCONTEXT: {greeting_context}\nUser: {user_problem}"
    try:
        it_result = query_azure_openai(it_check_prompt, system=MASTER_PROMPT, temperature=0.0).strip().upper()
        if it_result.startswith("NO"):
            return {"not_it": True}
    except Exception as e:
        print(f"[WARN] IT-related detector failed: {e}")

    # Otherwise, proceed with normal classification
    class_instr = (
        "Agent: Classification — Return a JSON object with keys: type (request|incident), category (hardware, software, access, network, other), "
        "subcategory, urgency (low, medium, high, critical). Respond ONLY with the JSON."
    )
    class_context = f"RECENT_HISTORY: {user_problem[:400]}"
    prompt = f"{class_instr}\nCONTEXT: {class_context}\nUser issue: {user_problem}"
    result = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
    print("[DEBUG] LLM classify_issue raw output:", result)
    try:
        match = re.search(r"{[\s\S]*?}", result)
        if not match:
            raise ValueError("No JSON object found in LLM output")
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Parsing classification JSON failed: {e}\nRaw output: {result}")
        return {}

# Top-level helpers for Outlook troubleshooting flow execution
def _build_outlook_troubleshoot_flow() -> list:
    """Return a default troubleshooting flow for Outlook issues.
    The flow is a list of step dicts: {id, title, script, on_success, on_fail}.
    Script paths are relative to BASE_DIR.
    """
    return [
        {"id": "check_internet", "title": "Checking Internet Connectivity", "script": os.path.join("scripts", "check_internet.ps1"), "on_success": "test_outlook_port", "on_fail": "network_reset"},
        {"id": "test_outlook_port", "title": "Testing Outlook service connectivity (outlook.office365.com:443)", "script": os.path.join("scripts", "test_outlook_port.ps1"), "on_success": "clear_roamcache", "on_fail": "network_reset"},
        {"id": "clear_roamcache", "title": "Backup and clear Outlook RoamCache", "script": os.path.join("scripts", "clear_roamcache.ps1"), "on_success": "restart_outlook", "on_fail": "restart_outlook"},
        {"id": "restart_outlook", "title": "Restarting Outlook", "script": os.path.join("scripts", "restart_outlook.ps1"), "on_success": "resolved", "on_fail": "network_reset"},
    ]


def _run_powershell_script(script_abs_path: str) -> Tuple[int, str, str]:
    """Execute a PowerShell script and return (returncode, stdout, stderr).
    Uses powershell.exe -ExecutionPolicy Bypass -File <script>.
    """
    pwsh = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not pwsh:
        return (127, "", "powershell executable not found on PATH")
    cmd = [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_abs_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired:
        return (124, "", "script timed out")
    except Exception as e:
        return (1, "", str(e))


def execute_troubleshoot_flow(session_state: dict) -> Tuple[bool, str]:
    """Execute the troubleshoot flow stored in session_state['troubleshoot_flow'].
    Returns (resolved_bool, summary_text).
    This will only run if the flow is present and the user has given explicit consent.
    NOTE: This runs scripts on the machine where this Python process is running. If AskGen
    is running as a server elsewhere rather than on the user's local machine, this will
    execute locally on the server instead of the user's PC. The caller must ensure this
    is appropriate.
    """
    flow = session_state.get("troubleshoot_flow") or []
    if not flow:
        return (False, "No troubleshooting flow configured.")

    history = session_state.get("troubleshoot_history", [])
    resolved = False
    step_map = {s["id"]: s for s in flow}
    # start with first step
    current = flow[0]["id"]
    while current and current not in (None, "resolved"):
        step = step_map.get(current)
        if not step:
            history.append({"id": current, "status": "missing_step"})
            break
        script_rel = step.get("script")
        script_abs = os.path.join(BASE_DIR, script_rel) if script_rel else None
        if not script_abs or not os.path.isfile(script_abs):
            history.append({"id": step["id"], "status": "missing_script", "script": script_rel})
            # follow on_fail if configured
            current = step.get("on_fail")
            continue
        # run the script
        rc, out, err = _run_powershell_script(script_abs)
        ok = (rc == 0)
        history.append({"id": step["id"], "script": script_rel, "rc": rc, "stdout": out.strip(), "stderr": err.strip(), "ok": ok})
        if ok:
            nxt = step.get("on_success")
        else:
            nxt = step.get("on_fail")
        if nxt == "resolved":
            resolved = True
            current = None
            break
        current = nxt

    session_state["troubleshoot_history"] = history
    # Build user-friendly summary
    lines = []
    for h in history:
        if h.get("status") == "missing_script":
            lines.append(f"Step {h.get('id')}: script not found: {h.get('script')}")
        elif h.get("status") == "missing_step":
            lines.append(f"Step {h.get('id')}: configuration missing")
        else:
            lines.append(f"Step {h.get('id')}: return code={h.get('rc')}, ok={h.get('ok')}. stdout: {h.get('stdout')[:240]}")
    if resolved:
        lines.append("Troubleshooting completed: outlook issue appears resolved.")
    else:
        lines.append("Troubleshooting completed: issue not resolved. Please contact IT support or run the suggested network reset.")
    return (resolved, "\n".join(lines))

def translate_text(text: str, target_lang: str) -> str:
    """Translate text into target_lang using Azure OpenAI translation via a simple system instruction.
    This is a best-effort translation for short texts (ticket short_description/description or LLM replies).
    If target_lang is 'en' or text is empty, returns text unchanged.
    """
    if not text or not target_lang or target_lang.lower().startswith('en'):
        return text
    try:
        instr = f"Translate the following text into {target_lang}. Return only the translated text, no explanation."
        prompt = f"{instr}\n\nText:\n" + text
        out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
        if isinstance(out, str) and out.strip():
            # Strip any JSON or extraneous text; keep raw translation
            return out.strip()
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
    return text
# --- LangGraph imports ---
from langgraph.graph import StateGraph, END
# Use Pydantic BaseModel for state if LangGraph State is unavailable
from pydantic import BaseModel
# --- LangGraph State and Nodes ---
class ITSMState(BaseModel):
    user_id: int
    message: str
    session_state: dict
    troubleshooting_response: str = None
    classification: dict = None
    ticket_info: dict = None
    escalated: bool = False
    solved: bool = False
    response: str = None
    agent_stage: str = None

def node_greeting(state: ITSMState):
    """Early greeting detector node. If greeting, respond and end, else continue to classification."""
    state.agent_stage = "Greeting Agent"
    msg = (state.message or "").strip()
    # quick heuristic for greetings/small talk
    if not msg:
        return state
    if re.search(r"^(hi|hello|hey|good\s(morning|afternoon|evening)|thanks|thank you)\b", msg, re.IGNORECASE) or (len(msg.split()) <= 3 and any(w in msg.lower() for w in ["hi","hello","thanks","thank"])):
        state.response = "Hello! How can I assist you with your IT needs today?"
        state.solved = True
        # Use a transient flag so we don't permanently mark the session as a 'greeting' which would block later flows
        state.session_state["last_was_greeting"] = True
        # persist to chat history
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        return state
    return state


def llm_ask_followup(classification, user_problem, session_state):
    """Ask LLM to produce a clarifying question and required slots for a request-type ticket."""
    instr = (
        "Agent: Clarifier — Given the ticket classification and context, return a JSON object with keys: question (a single clarifying question), "
        "required_slots (an array of short slot names e.g. [\"software_name\"]). Respond ONLY with the JSON."
    )
    # If escalation was confirmed, hint to the LLM to prepare an incident payload
    ctx = {"classification": classification, "session_state": session_state, "user_problem": user_problem}
    if session_state.get("escalation_confirmed") or session_state.get("type") == "incident":
        ctx["force_type"] = "incident"
    context = ctx
    prompt = f"{instr}\nCONTEXT: {json.dumps(context)}\nUser issue: {user_problem}"
    out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
    try:
        m = re.search(r"{[\s\S]*?}", out)
        if not m:
            raise ValueError("No JSON found")
        j = json.loads(m.group(0))
        return j
    except Exception as e:
        print(f"[ERROR] Clarifier LLM failed: {e}\nRaw: {out}")
        return None


def node_clarify(state: ITSMState):
    """Node to ask clarifying question(s) for request-type tickets before creating sc_request."""
    state.agent_stage = "Clarify Agent"
    classification = state.classification or {}
    user_problem = get_original_user_problem(state.session_state, state.message)
    # If there is a pending_slots entry, consume the user's latest message as the answer for the first slot
    pending_slots = state.session_state.get("pending_slots", [])
    if pending_slots and state.message:
        slots = state.session_state.get("slots", {})
        slot_name = pending_slots.pop(0)
        slots[slot_name] = state.message.strip()
        state.session_state["slots"] = slots
        state.session_state["pending_slots"] = pending_slots
        # If no more pending slots, mark clarified
        if not pending_slots:
            state.session_state["clarified"] = True
            state.session_state.pop("pending_question", None)
            # done collecting slots
            state.session_state.pop("awaiting_slot_collection", None)
            state.response = "Thanks — I've captured that information. I'll proceed to create the request."
            # persist bot reply
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            state.solved = False
            return state
        else:
            # still more slots to collect
            # set next pending_question based on slot_questions mapping
            slot_questions = state.session_state.get("slot_questions", {})
            next_slot = pending_slots[0]
            next_q = slot_questions.get(next_slot) if slot_questions else None
            state.session_state["pending_question"] = next_q
            state.response = next_q or "Please provide the requested information."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            state.solved = False
            return state
    # If already clarified, nothing to do
    if state.session_state.get("clarified"):
        return state
    # Otherwise, ask LLM for a follow-up question for request-type only when no catalog suggestion handled it.
    # If a catalog_suggestion exists in session_state but the user has not confirmed it yet, we should not call the LLM clarifier here.
    if state.session_state.get("catalog_suggestion") and state.session_state.get("awaiting_catalog_confirmation"):
        # Wait for user's confirmation handled in orchestrator
        return state
    # Otherwise, ask LLM for a follow-up question for request-type
    clar = llm_ask_followup(classification, user_problem, state.session_state)
    if clar and isinstance(clar, dict) and clar.get("question") and clar.get("required_slots"):
        q = clar.get("question")
        slots = clar.get("required_slots")
        state.session_state["pending_slots"] = slots
        state.session_state["pending_question"] = q
        state.response = q
        # persist bot question
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        state.solved = False
        return state
    # fallback: no clarifier available, continue
    return state

def node_troubleshooting(state: ITSMState):
    state.agent_stage = "Troubleshooting Agent"
    # Defensive: ensure troubleshooting node never exposes or persists ticket-like identifiers
    try:
        # remove any accidental ticket-like keys that might have been set elsewhere
        for k in list(state.session_state.keys()):
            lk = k.lower()
            if any(tok in lk for tok in ("ticket", "ticket_id", "request_number", "sys_id", "incident")):
                # keep ticket creation history but avoid fabricated ticket identifiers in session during troubleshooting
                if k not in ("created_tickets", "last_ticket_created"):
                    state.session_state.pop(k, None)
    except Exception:
        pass
    # If the user was asked to consent to run a troubleshoot flow and replied, handle it here as well
    if state.session_state.get("awaiting_troubleshoot_consent"):
        reply = (state.message or "").strip()
        # Use whole-word regex to avoid accidental substring matches
        if re.search(r"\b(yes|y|sure|please do|go ahead|run|okay|ok)\b", reply, re.IGNORECASE):
            # clear awaiting flag
            state.session_state.pop("awaiting_troubleshoot_consent", None)
            try:
                resolved, summary = execute_troubleshoot_flow(state.session_state)
            except Exception as e:
                state.response = f"I attempted to run the troubleshooting flow but an error occurred: {e}"
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                # Record a deterministic anchor index so frontend can attach the troubleshooting
                # panel to this exact assistant message (avoids brittle string matching).
                try:
                    state.session_state["chat_history"] = ch
                    state.session_state["troubleshoot_anchor"] = len(ch) - 1
                except Exception:
                    state.session_state["chat_history"] = ch
                return state
            # Persist and return summary to user
            # If the flow reported 'resolved', ask the user to confirm resolution; otherwise offer to create a ticket
            if resolved:
                # Provide concise confirmation and point to the troubleshooting panel for details
                state.response = "I've run the steps and they report success. Please check the troubleshooting panel below for detailed output. Did this resolve your issue? Reply 'yes' or 'no'."
                ch = state.session_state.get("chat_history", [])
                # If last bot message looks like the consent prompt, replace it so the troubleshooting panel
                    # attaches to this follow-up message. Otherwise append.
                if ch and isinstance(ch[-1], (list, tuple)) and str(ch[-1][0]).lower() in ("bot", "assistant"):
                    last_msg = str(ch[-1][1] or "").lower()
                    if "i can run a short, safe troubleshooting run now" in last_msg or "reply 'yes' to run these on this machine" in last_msg or "reply 'yes' to proceed on this machine" in last_msg:
                        ch[-1] = ("Bot", state.response)
                        # Update anchor to point at the replaced message
                        try:
                            state.session_state["chat_history"] = ch
                            state.session_state["troubleshoot_anchor"] = len(ch) - 1
                        except Exception:
                            state.session_state["chat_history"] = ch
                        print("[DEBUG] Replaced consent prompt with resolution follow-up in chat_history")
                    else:
                        ch.append(("Bot", state.response))
                        try:
                            state.session_state["chat_history"] = ch
                            state.session_state["troubleshoot_anchor"] = len(ch) - 1
                        except Exception:
                            state.session_state["chat_history"] = ch
                else:
                    ch.append(("Bot", state.response))
                    try:
                        state.session_state["chat_history"] = ch
                        state.session_state["troubleshoot_anchor"] = len(ch) - 1
                    except Exception:
                        state.session_state["chat_history"] = ch
                # Ask the user to explicitly confirm resolution
                state.session_state["awaiting_resolution_confirmation"] = True
                state.session_state["awaiting_feedback"] = False
                state.solved = False
                return state
            else:
                # Short apology and escalation offer; details are available in the troubleshooting panel
                state.response = "I ran the steps but they didn't resolve the problem. I can create a support incident with the logs — would you like me to do that? Reply 'yes' to create a ticket or 'no' to try more troubleshooting."
                ch = state.session_state.get("chat_history", [])
                # If last bot message looks like the consent prompt, replace it so the troubleshooting panel
                # attaches to the escalation prompt bubble. Otherwise append normally.
                if ch and isinstance(ch[-1], (list, tuple)) and str(ch[-1][0]).lower() in ("bot", "assistant"):
                    last_msg = str(ch[-1][1] or "").lower()
                    if "i can run a short, safe troubleshooting run now" in last_msg or "reply 'yes' to run these on this machine" in last_msg or "reply 'yes' to proceed on this machine" in last_msg:
                        ch[-1] = ("Bot", state.response)
                        try:
                            state.session_state["chat_history"] = ch
                            state.session_state["troubleshoot_anchor"] = len(ch) - 1
                        except Exception:
                            state.session_state["chat_history"] = ch
                        print("[DEBUG] Replaced consent prompt with escalation follow-up in chat_history")
                    else:
                        ch.append(("Bot", state.response))
                        try:
                            state.session_state["chat_history"] = ch
                            state.session_state["troubleshoot_anchor"] = len(ch) - 1
                        except Exception:
                            state.session_state["chat_history"] = ch
                else:
                    ch.append(("Bot", state.response))
                    try:
                        state.session_state["chat_history"] = ch
                        state.session_state["troubleshoot_anchor"] = len(ch) - 1
                    except Exception:
                        state.session_state["chat_history"] = ch
                # Ask the user whether to escalate
                state.session_state["awaiting_escalation_confirmation"] = True
                state.session_state["awaiting_feedback"] = False
                state.solved = False
                return state
        elif re.search(r"\b(no|n|not now|don't|dont|nope)\b", reply, re.IGNORECASE):
            state.session_state.pop("awaiting_troubleshoot_consent", None)
            state.response = "Okay — I won't run any scripts. I can provide the steps for you to run manually or try other troubleshooting approaches. What would you like to do next?"
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
        else:
            state.response = "Please reply 'yes' to allow me to run the troubleshooting steps on this machine, or 'no' to decline."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
    # If awaiting_feedback is set, this is a feedback message
    awaiting_feedback = state.session_state.get("awaiting_feedback", False)
    # If we are currently collecting catalog variables, do not run troubleshooting; defer to clarify
    if state.session_state.get("awaiting_slot_collection") or (state.session_state.get('pending_slots') and not state.session_state.get('clarified')):
        # Let clarify node handle slot collection
        return state
    negative_keywords = [
        "not working", "didn't work", "did not work", "no", "still", "problem persists", "issue remains", "unsolved", "doesn't help", "failed", "unable", "escalate", "need help", "need support", "not helping", "not resolved", "not fixed", "not solved", "frustrated", "waste of time", "nothing works", "useless", "unable to solve", "unable to fix", "not able to solve", "not able to fix", "not able to resolve", "unable to resolve"
    ]
    # If we are awaiting explicit feedback to a troubleshooting step
    # Handle the special resolution confirmation flow first (user asked to confirm resolved after successful run)
    if state.session_state.get("awaiting_resolution_confirmation"):
        reply = (state.message or "").strip().lower()
        yes_vals = ["yes", "y", "fixed", "resolved", "works", "working", "solved", "that fixed it"]
        no_vals = ["no", "n", "still", "not", "not working", "didn't work", "did not work", "still not working"]
        if any(v in reply for v in yes_vals):
            state.response = "Great — glad that resolved your issue. If you need anything else, let me know."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            state.session_state["awaiting_resolution_confirmation"] = False
            state.session_state["awaiting_feedback"] = False
            state.solved = True
            return state
        elif any(v in reply for v in no_vals):
            # User reports still not resolved -> ask whether to create a ticket
            state.session_state["awaiting_resolution_confirmation"] = False
            state.session_state["awaiting_escalation_confirmation"] = True
            state.response = "Sorry that didn't fix it. Would you like me to create a support incident/ticket for this issue? Reply 'yes' to create the ticket or 'no' to try more troubleshooting."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            state.solved = False
            return state
        else:
            state.response = "Please reply 'yes' if the issue is resolved, or 'no' if it's still not working and you'd like me to create a ticket."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state

    if awaiting_feedback:
        # If the user replied with a non-IT/general question, treat it as out-of-scope and inform them.
        try:
            if is_non_it_query(state.message):
                state.response = "Sorry — I can only help with IT-related support requests. I don't have information on that topic."
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                state.session_state["chat_history"] = ch
                # Clear awaiting_feedback because this reply wasn't feedback for the troubleshooting steps
                state.session_state["awaiting_feedback"] = False
                state.solved = True
                return state
        except Exception:
            pass

        negative_feedback = any(word in (state.message or "").lower() for word in negative_keywords)
        if negative_feedback:
            state.session_state["negative_feedback_count"] = state.session_state.get("negative_feedback_count", 0) + 1
            state.solved = False
            # If user explicitly says it didn't work, auto-confirm escalation and route to ticketing
            # Set explicit incident type so downstream ticketing creates an incident
            state.session_state["type"] = "incident"
            # Record escalation reason for operator context
            state.session_state["escalation_reason"] = state.troubleshooting_response or "User reported troubleshooting steps did not resolve the issue."
            # Mark escalation as confirmed so the ticketing node performs the deterministic creation
            state.session_state["escalation_confirmed"] = True
            # Mark clarified so the orchestrator routes directly to ticketing
            state.session_state["clarified"] = True
            # Clear awaiting_feedback since we've handled the negative reply
            state.session_state["awaiting_feedback"] = False
            # Inform the user we will create the ticket now (ticketing node will add details)
            state.response = "I will create a support incident now and include the troubleshooting logs. One moment while I create the ticket."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
        else:
            state.session_state["negative_feedback_count"] = 0
            state.session_state["escalated"] = False
            state.solved = True  # Only solved if no negative feedback
            state.session_state["awaiting_feedback"] = False
            # Do not generate another troubleshooting response here
            state.troubleshooting_response = ""
            state.response = "Glad that helped — let me know if you need anything else."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
    else:
        # First troubleshooting step, force feedback
        state.session_state["awaiting_feedback"] = True
        state.solved = False  # Not solved yet, waiting for feedback, must be bool for Pydantic
        allow_troubleshooting = state.session_state.get('escalated', False) is False
        troubleshooting_response = agentic_self_service(state.message, state.session_state, allow_troubleshooting=allow_troubleshooting)
        state.troubleshooting_response = troubleshooting_response
        # Defensive sanitization: if the LLM returned structured JSON containing ticket-like keys
        # (e.g., ticket, ticket_id, number, sys_id, INC/REQ/RITM patterns), do not surface it —
        # convert to an explicit ESCALATE flow so the ticketing node creates the record.
        try:
            if isinstance(troubleshooting_response, str):
                txt = troubleshooting_response.strip()
                # If the response already used the ESCALATE sentinel, follow normal path
                if txt.upper() == "ESCALATE":
                    state.session_state["awaiting_escalation_confirmation"] = True
                    if not state.session_state.get("type"):
                        state.session_state["type"] = "incident"
                    state.session_state["escalation_reason"] = "Automated troubleshooting determined the issue requires escalation."
                    state.response = "I couldn't resolve this with self-service steps. Would you like me to create a support incident/ticket for you? Reply 'yes' to create the ticket or 'no' to try more troubleshooting."
                    ch = state.session_state.get("chat_history", [])
                    ch.append(("Bot", state.response))
                    state.session_state["chat_history"] = ch
                    state.session_state["awaiting_feedback"] = False
                    return state
                # Try to extract JSON object from text and inspect keys/values
                m = re.search(r"\{[\s\S]*?\}", txt)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        keys = [k.lower() for k in (parsed.keys() if isinstance(parsed, dict) else [])]
                        tickety_keys = any(k for k in keys if any(tok in k for tok in ("ticket", "ticket_id", "number", "sys_id", "request")))
                        tickety_vals = False
                        for v in (parsed.values() if isinstance(parsed, dict) else []):
                            try:
                                sval = str(v)
                                if re.search(r"\b(INC|REQ|RITM)\d+", sval, re.IGNORECASE):
                                    tickety_vals = True
                                    break
                            except Exception:
                                continue
                        if tickety_keys or tickety_vals:
                            # Replace with sentinel; do not return the JSON containing ticket identifiers
                            state.session_state["awaiting_escalation_confirmation"] = True
                            if not state.session_state.get("type"):
                                state.session_state["type"] = "incident"
                            state.session_state["escalation_reason"] = "Automated troubleshooting determined the issue requires escalation."
                            state.response = "I couldn't resolve this with self-service steps. Would you like me to create a support incident/ticket for you? Reply 'yes' to create the ticket or 'no' to try more troubleshooting."
                            ch = state.session_state.get("chat_history", [])
                            ch.append(("Bot", state.response))
                            state.session_state["chat_history"] = ch
                            state.session_state["awaiting_feedback"] = False
                            return state
                    except Exception:
                        # If parsing fails, continue with original response
                        pass
        except Exception:
            pass
        # Normal successful troubleshooting reply (safe to surface)
        state.response = troubleshooting_response
        # persist bot reply in session_state chat_history
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", troubleshooting_response))
        state.session_state["chat_history"] = ch
        return state


def node_classification(state: ITSMState):
    state.agent_stage = "Classification Agent"
    # Defensive guard: if we are awaiting explicit consent to run a troubleshoot flow,
    # do not run classification here; let the Troubleshooting node handle the user's reply first.
    if state.session_state.get("awaiting_troubleshoot_consent"):
        # Re-prompt the user to reply to the consent question if needed
        # (the orchestrator/troubleshooting nodes are responsible for the consent flow)
        return state
    # If we're waiting for the user to confirm resolution, do not re-run classification
    if state.session_state.get("awaiting_resolution_confirmation"):
        return state
    user_problem = get_original_user_problem(state.session_state, state.message)
    lp = (user_problem or "").lower()
    # Lightweight keyword hinting to prefer request path for software/install/access intents
    request_keywords = ["install", "installation", "install software", "software install", "software installation", "sharepoint", "access", "request", "provision", "permission", "rights"]

    # Detect explicit user intent to raise/open/create a request and force request type immediately
    explicit_request_regex = r"\b(raise (a )?request|i want to raise( a)? request|i want to request|please create (a )?request|open (a )?request|create request|raise request|request access|i need access)\b"
    if re.search(explicit_request_regex, lp, re.IGNORECASE):
        state.session_state["type_hint"] = "request"
        # Set explicit type so downstream logic doesn't get overwritten by LLM
        state.session_state["type"] = "request"

    # Also apply keyword hints
    if any(k in lp for k in request_keywords):
        state.session_state["type_hint"] = "request"

    # Detect explicit status-check requests and route to status_check
    ticket_id = detect_status_check(user_problem)
    if ticket_id:
        state.session_state["ticket_check_number"] = ticket_id
        return state

    classification = agentic_classification(user_problem)
    state.classification = classification

    # If the LLM determined this is not an IT-related query, short-circuit with a polite reply.
    if classification.get("not_it"):
        state.response = "Sorry — I can only help with IT-related support requests. I don't have information on that topic."
        state.solved = True
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        return state

    # Handle greeting/small talk: respond and END
    if classification.get("greeting"):
        state.response = "Hello! How can I assist you with your IT needs today?"
        state.solved = True
        # transient greeting flag
        state.session_state["last_was_greeting"] = True
        return state

    # If LLM returned an explicit type, honor it only if we don't already have an explicit session-level type
    if classification.get("type") in ["request", "incident"]:
        if not state.session_state.get("type"):
            state.session_state["type"] = classification.get("type")
    else:
        # If frontend or earlier logic hinted request, respect it
        if state.session_state.get("type_hint") == "request" or state.session_state.get("type") == "request":
            state.session_state["type"] = "request"
        else:
            category = classification.get("category", "").lower()
            urgency = classification.get("urgency", "").lower()
            # Simple heuristic: treat 'access', 'installation', 'request' as request, others as incident
            if category in ["access", "installation", "request"] or urgency in ["low", "medium"]:
                state.session_state["type"] = "request"
            else:
                state.session_state["type"] = "incident"
    return state

def node_ticketing(state: ITSMState):
    state.agent_stage = "Ticketing Agent"
    # Safety guard: if we're awaiting consent for running troubleshooting steps, delay ticket creation
    if state.session_state.get("awaiting_troubleshoot_consent"):
        # Build a friendly consent prompt that lists the planned steps and short descriptions
        flow = state.session_state.get("troubleshoot_flow") or []
        if flow:
            # Short descriptions by known step id (fallback to empty string)
            short_desc = {
                "check_internet": "Quick network reachability test (non-destructive).",
                "test_outlook_port": "TCP connectivity check to outlook.office365.com:443.",
                "clear_roamcache": "Back up and clear Outlook RoamCache (safe, non-destructive).",
                "restart_outlook": "Safely stop and restart Outlook processes in your user session."
            }
            lines = ["I can run the following safe troubleshooting steps on your PC:"]
            for i, s in enumerate(flow, start=1):
                title = s.get("title") or s.get("id")
                desc = short_desc.get(s.get("id"), "")
                lines.append(f"{i}) {title} — {desc}")
            lines.append("")
            lines.append("Reply 'yes' to run these steps on your PC or 'no' to decline. These scripts run locally and will only act on processes owned by your user account. We will record your consent for auditing.")
            state.response = "\n".join(lines)
        else:
            state.response = "I can run a few safe diagnostic steps on your PC (network check and restarting the app). Reply 'yes' to run them or 'no' to decline. These scripts run locally and only affect processes owned by your user account."
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        return state
    # If we're currently waiting for the user's explicit escalation confirmation, do not create a ticket here.
    # The orchestrator/troubleshooting nodes will set `escalation_confirmed` and route to ticketing once the user replies.
    if state.session_state.get("awaiting_escalation_confirmation"):
        return state
    # If waiting for the user to confirm resolution after troubleshooting, do not create tickets yet
    if state.session_state.get("awaiting_resolution_confirmation"):
        return state
    classification = state.classification or {}
    user_id = state.user_id
    user_problem = get_original_user_problem(state.session_state, state.message)
    # If this ticketing call follows an explicit escalation confirmation from troubleshooting,
    # ensure we create an incident regardless of any LLM classification that may suggest a request.
    if state.session_state.get("escalation_confirmed"):
        state.session_state["type"] = "incident"
    # Determine explicit table from session_state/type or classification to avoid relying on LLM payloads
    type_ = state.session_state.get("type", classification.get("type", "request"))
    table = "sc_request" if type_ == "request" else "incident"
    # Prepare ticket payload via LLM (allows richer capture of attempted solutions and context)
    prepared_payload = llm_prepare_ticket(classification, user_problem, state.session_state)
    # If escalation was explicitly confirmed, force incident creation and ensure payload marks it as an incident
    if state.session_state.get("escalation_confirmed"):
        table = "incident"
        try:
            state.session_state["type"] = "incident"
            if isinstance(prepared_payload, dict):
                prepared_payload["type"] = "incident"
                # Append escalation reason for operator context
                reason = state.session_state.get("escalation_reason")
                if reason:
                    existing_desc = prepared_payload.get("description") or ""
                    prepared_payload["description"] = (existing_desc + "\n\nEscalation reason: " + str(reason)).strip()
        except Exception:
            # best-effort; if mutation fails, continue and rely on explicit table variable
            pass
    # Create ticket in ServiceNow using prepared payload when available; pass explicit table
    ticket = create_ticket(None, user_id, user_problem, classification, None, payload_override=prepared_payload, session_state=state.session_state, table=table)
    assignee_name = getattr(ticket, "assigned_to_name", None) or 'Unassigned'
    ticket_info = {
        "id": getattr(ticket, "id", None),
        "number": getattr(ticket, "number", None),
        "status": getattr(ticket, "status", None),
        "assigned_to": assignee_name,
        "priority": getattr(ticket, "priority", None),
        "category": getattr(ticket, "category", None),
        "subcategory": getattr(ticket, "subcategory", None),
        "urgency": getattr(ticket, "urgency", None),
        "description": getattr(ticket, "description", None),
        "created_at": getattr(ticket, "created_at", None)
    }
    state.ticket_info = ticket_info
    # persist bot reply and created ticket info
    ch = state.session_state.get("chat_history", [])
    if table == "incident":
        state.response = f"Your issue has been categorized as an incident and a ticket has been created. Our IT team will assist you soon. Your ticket number is {ticket_info.get('number') or ticket_info.get('id')}."
    else:
        state.response = f"Your request has been routed and a ticket has been created. Our IT team will assist you soon. Your ticket number is {ticket_info.get('number') or ticket_info.get('id')}."
    ch.append(("Bot", state.response))
    # also store created ticket in session_state
    st = state.session_state.get("created_tickets", [])
    st.append({"number": ticket_info.get("number"), "id": ticket_info.get("id"), "created_at": ticket_info.get("created_at")})
    state.session_state["created_tickets"] = st
    state.session_state["chat_history"] = ch
    state.solved = False
    # Clear the explicit 'type' after ticket creation so new user input is re-evaluated by the orchestrator
    try:
        state.session_state.pop("type", None)
        # record timestamp for observability
        state.session_state["last_ticket_created"] = datetime.utcnow().isoformat() + "Z"
    except Exception:
        pass
    return state

def node_already_escalated(state: ITSMState):
    state.agent_stage = "Already Escalated Agent"
    state.response = "Your issue has already been escalated to IT support. Please wait for assistance or check your ticket status."
    state.solved = False
    return state

def node_status_check(state: ITSMState):
    """Node that performs ticket status lookup (incident or sc_request) and returns a human-friendly response."""
    state.agent_stage = "Status Check Agent"
    ticket_number = state.session_state.get("ticket_check_number") or detect_status_check(state.message)
    if not ticket_number:
        state.response = "I couldn't find a ticket number in your message. Please provide the ticket ID (e.g. INC0012345)."
        state.solved = True
        return state
    # Query ServiceNow via existing helper
    try:
        t = get_ticket(ticket_number)
        if t.get("error"):
            state.response = f"I couldn't find a ticket with ID {ticket_number}. Please check the number and try again."
            state.ticket_info = None
            state.solved = True
            return state
        # Human-friendly response
        assigned = t.get("assigned_to") or "Unassigned"
        status = t.get("status") or "Unknown"
        priority = t.get("priority") or t.get("urgency") or "N/A"
        created = t.get("created_at") or "N/A"
        desc = (t.get("description") or "").strip()
        summary = f"Ticket {t.get('number') or t.get('id')} is currently '{status}'. Assigned to: {assigned}. Priority: {priority}. Created: {created}."
        if desc:
            summary += f" Description: {desc[:240]}{'...' if len(desc)>240 else ''}"
        state.response = summary
        state.ticket_info = t
        state.solved = True
        return state
    except Exception as e:
        print(f"[ERROR] Status check failed: {e}")
        state.response = "There was an error checking the ticket status. Please try again later."
        state.solved = False
        return state

def decide_after_classification(state: ITSMState):
    # If a ticket_check_number exists, go to status_check
    if state.session_state.get("ticket_check_number"):
        return "status_check"
    type_ = state.session_state.get("type", state.classification.get("type") if state.classification else "request")
    # If request and clarification not done, go to clarify
    if type_ == "request" and not state.session_state.get("clarified", False):
        return "clarify"
    if type_ == "request":
        return "ticketing"
    else:
        return "troubleshooting"

def decide_after_troubleshooting(state: ITSMState):
    # If awaiting_feedback is True, stop and wait for user input
    if state.session_state.get("awaiting_feedback", False):
        return END
    # If solved (user satisfied), end. If not solved (negative feedback), go to routing (re-classification) and then ticketing.
    if state.solved is True:
        return END
    elif state.solved is False:
        return "routing"
    else:
        return END

def decide_after_routing(state: ITSMState):
    return "ticketing"

def decide_after_greeting(state: ITSMState):
    """If the previous turn was a greeting, only short-circuit when the current message is also a greeting.
    Otherwise clear the transient flag and continue to classification."""
    try:
        msg = (state.message or "").strip()
        # If last turn was a greeting, only end if this turn is also a greeting/small-talk
        if state.session_state.get("last_was_greeting"):
            if re.search(r"^(hi|hello|hey|good\s(morning|afternoon|evening)|thanks|thank you)\b", msg, re.IGNORECASE) or (len(msg.split()) <= 3 and any(w in msg.lower() for w in ["hi","hello","thanks","thank"])):
                return END
            # Not a greeting this turn -> clear the transient flag and continue
            state.session_state.pop("last_was_greeting", None)
            return "classification"
        # Fallback: if the state was solved and message looks like greeting, end
        if getattr(state, "solved", False) and re.search(r"^(hi|hello|hey|good\s(morning|afternoon|evening)|thanks|thank you)\b", msg, re.IGNORECASE):
            return END
    except Exception:
        pass
    return "classification"

def node_orchestrator(state: ITSMState):
    """Central LLM orchestrator node. Decides which agent should run next based on message + session_state.
    It performs lightweight checks (greeting, status-check, explicit request phrases) then calls the classifier
    to populate `state.classification` if needed. It does not perform the downstream action itself.
    """
    state.agent_stage = "Orchestrator"
    msg = (state.message or "").strip()
    lp = msg.lower()
    # If we are currently awaiting the user's explicit consent to run a troubleshoot flow,
    # do not continue with classification or ticketing logic in this orchestrator turn.
    # The Troubleshooting node will handle the user's yes/no reply and run the flow as appropriate.
    if state.session_state.get("awaiting_troubleshoot_consent"):
        return state
    # If we are awaiting resolution confirmation, avoid calling LLM classifiers on short replies
    # — let the Troubleshooting node handle them. We DO NOT short-circuit on awaiting_escalation_confirmation
    # here because the orchestrator contains deterministic handling for escalation confirmations
    # and should be allowed to interpret simple yes/no replies and route to ticketing.
    if state.session_state.get("awaiting_resolution_confirmation"):
        return state
    # Quick deterministic check: if the message is a general-knowledge / non-IT query, respond directly.
    try:
        if is_non_it_query(msg):
            state.response = "Sorry — I can only help with IT-related support requests. I don't have information on that topic."
            state.solved = True
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
    except Exception:
        pass
    # Persist chat_history already handled by API caller
    # 1) Status check detection (user asked for ticket status)
    ticket_id = detect_status_check(msg)
    if ticket_id:
        state.session_state["ticket_check_number"] = ticket_id
        return state
    # Detect common Outlook issues and proactively offer a guided troubleshooting flow
    try:
        # Broaden detection to capture variants like 'outlook not', 'outlook issue', 'outlook problem'
        outlook_kw_matches = ["not working", "can't open", "cannot open", "won't open", "crash", "keeps crashing", "stopped working", "not syncing", "sync issues", "not", "problem", "issue", "broken"]
        if "outlook" in lp and any(kw in lp for kw in outlook_kw_matches):
            # Avoid prompting while collecting catalog or slots
            if not state.session_state.get('awaiting_slot_collection') and not state.session_state.get('awaiting_catalog_confirmation'):
                # Prepare a default Outlook troubleshooting flow and ask user for consent to run it locally
                flow = _build_outlook_troubleshoot_flow()
                state.session_state["troubleshoot_flow"] = flow
                # also provide a structured array of step titles for the frontend consent UI
                try:
                    state.session_state["troubleshoot_consent_steps"] = [s.get("title") or s.get("id") for s in flow]
                except Exception:
                    state.session_state["troubleshoot_consent_steps"] = []
                state.session_state["awaiting_troubleshoot_consent"] = True
                # mark that we've prompted so we don't keep re-prompting verbatim
                state.session_state["troubleshoot_consent_prompted"] = True
                # present the steps concisely to the user and ask for explicit consent in an agentic tone
                steps = "\n".join([f"- {s['title']}" for s in flow])
                state.response = f"I can run a short, safe troubleshooting run now:\n{steps}\n\nReply 'yes' to run these on this machine or 'no' to decline. I will only act on processes owned by your account and will record your consent."
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                state.session_state["chat_history"] = ch
                print("[DEBUG] Prompted user for Outlook troubleshoot consent")
                return state
    except Exception:
        pass

    # Fallback: if a troubleshoot_flow exists but for some reason no consent flag is set, re-prompt the user
    try:
        # If a troubleshoot_flow exists but for some reason no consent flag is set, re-prompt once
        if state.session_state.get('troubleshoot_flow') and not state.session_state.get('awaiting_troubleshoot_consent') and not state.session_state.get('awaiting_slot_collection') and not state.session_state.get('troubleshoot_consent_prompted'):
            flow = state.session_state.get('troubleshoot_flow')
            try:
                state.session_state['troubleshoot_consent_steps'] = [s.get('title') or s.get('id') for s in flow]
            except Exception:
                state.session_state['troubleshoot_consent_steps'] = []
            steps = "\n".join([f"- {s['title']}" for s in flow])
            state.session_state['awaiting_troubleshoot_consent'] = True
            state.session_state['troubleshoot_consent_prompted'] = True
            state.response = f"I can run a short, safe troubleshooting run now:\n{steps}\n\nReply 'yes' to run these on this machine or 'no' to decline. I will only act on processes owned by your account and will record your consent."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            print("[DEBUG] Re-prompted for existing troubleshoot_flow consent")
            return state
    except Exception:
        pass
    # If we're waiting for the user's yes/no to run a troubleshoot flow, pause here and allow the
    # Troubleshooting node to handle the user's reply and execute the flow. This prevents the
    # orchestrator and the troubleshooting node from both acting on the same message.
    if state.session_state.get("awaiting_troubleshoot_consent"):
        return state
    # If we are awaiting escalation confirmation, interpret user's reply
    if state.session_state.get("awaiting_escalation_confirmation"):
        reply = lp
        yes_vals = ["yes", "y", "sure", "please do", "create ticket", "create", "confirm"]
        no_vals = ["no", "n", "not now", "try more", "try again", "don't"]
        if any(v in reply for v in yes_vals):
            # user confirmed escalation -> proceed to ticketing as incident unless explicit request set
            state.session_state["awaiting_escalation_confirmation"] = False
            # When the user explicitly confirms escalation after troubleshooting, create an incident.
            # Override any prior 'request' classification so we open an incident for IT investigation.
            state.session_state["type"] = "incident"
            # mark clarified so orchestrator routes directly to ticketing
            state.session_state["clarified"] = True
            state.session_state["escalation_confirmed"] = True
            return state
        elif any(v in reply for v in no_vals):
            # user declined escalation -> continue troubleshooting
            state.session_state["awaiting_escalation_confirmation"] = False
            state.session_state["awaiting_feedback"] = True
            # respond with a friendly follow-up
            state.response = "Okay — let's try another troubleshooting step. Please describe any additional details or say 'still not working' after trying the steps."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
        # if unclear, ask again
        state.response = "Please reply 'yes' to create a ticket or 'no' to continue troubleshooting."
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        return state
    # If we are awaiting catalog confirmation (user was offered a catalog item), interpret reply
    if state.session_state.get("awaiting_catalog_confirmation"):
        reply = lp
        yes_vals = ["yes", "y", "sure", "please do", "create ticket", "create", "confirm"]
        no_vals = ["no", "n", "not now", "try more", "try again", "don't", "dont"]
        # User confirmed they want to create a request from the suggested catalog item
        if any(v in reply for v in yes_vals):
            # clear awaiting flag now that user replied
            state.session_state.pop("awaiting_catalog_confirmation", None)
            item = state.session_state.get("catalog_suggestion")
            if not item or not item.get("sys_id"):
                state.response = "I couldn't retrieve the catalog item details. Please provide more information about what you need."
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                state.session_state["chat_history"] = ch
                return state
            # Fetch variables (item_option_new) using the richer field set used by /catalog/item
            try:
                vparams = {
                    "sysparm_query": f"cat_item={item['sys_id']}",
                    "sysparm_fields": "sys_id,question,question_text,variable_name,type,order,mandatory,default_value,reference,choice",
                    "sysparm_limit": 300
                }
                vars_list = _sn_table_get('item_option_new', vparams)
                pending_slots = []
                slot_questions = {}
                slot_meta = {}
                for v in vars_list:
                    # variable_name is canonical when present, fallback to question text
                    var_name = (v.get('variable_name') or v.get('name') or v.get('question') or v.get('question_text'))
                    if not var_name:
                        continue
                    # Prefer the explicit question_text if present, else use question or variable_name
                    q = (v.get('question_text') or v.get('question') or var_name).strip()
                    # Normalize slot key to a safe identifier (no spaces)
                    key = str(var_name).strip().replace(' ', '_')
                    # Determine field type and choices/reference info
                    v_type = (v.get('type') or '').lower()
                    choices = None
                    if v_type in ("choice", "select") and v.get('choice'):
                        raw = v.get('choice')
                        if isinstance(raw, str):
                            choices = [{'value': c.strip(), 'label': c.strip()} for c in raw.split(',') if c.strip()]
                    reference_table = v.get('reference')
                    mandatory = bool(v.get('mandatory'))
                    # Build a more instructive question for choice/reference fields
                    instr_q = q
                    if choices:
                        opts = ', '.join([c.get('label') for c in choices])
                        instr_q = f"{instr_q} Choices: {opts}. Please pick one."
                    if reference_table:
                        instr_q = f"{instr_q} This is a reference field. Type a name and I will search for the matching record to select."
                    if mandatory:
                        instr_q = f"{instr_q} (required)"
                    pending_slots.append(key)
                    slot_questions[key] = instr_q
                    slot_meta[key] = {'type': v_type or 'string', 'choices': choices, 'reference_table': reference_table, 'mandatory': mandatory}
                # Ensure this flow is treated as a request and mark that we're collecting slots
                state.session_state["type"] = "request"
                state.session_state["awaiting_slot_collection"] = True
                # store metadata so clarifier/slot-collecting nodes can perform lookups or present choices
                state.session_state['slot_meta'] = slot_meta
                # If no variables found, let the user know and proceed to create the request
                if not pending_slots:
                    state.response = f"The catalog item '{item.get('name')}' has no user-fillable variables. I'll create the request now."
                    ch = state.session_state.get("chat_history", [])
                    ch.append(("Bot", state.response))
                    state.session_state["chat_history"] = ch
                    # mark clarified so flow moves to ticketing
                    state.session_state["clarified"] = True
                    # ensure we are not left in slot-collection mode
                    state.session_state.pop("awaiting_slot_collection", None)
                    return state
                # Populate session_state for slot collection
                state.session_state["pending_slots"] = pending_slots
                state.session_state["slots"] = {}
                state.session_state["slot_questions"] = slot_questions
                # Ask the first required question
                first_slot = pending_slots[0]
                state.session_state["pending_question"] = slot_questions.get(first_slot) or f"Please provide {first_slot}."
                state.response = state.session_state["pending_question"]
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                state.session_state["chat_history"] = ch
                # do not mark clarified; node_clarify will mark when slots exhausted
                return state
            except Exception as e:
                print(f"[WARN] Failed to fetch catalog variables: {e}")
                state.response = "I couldn't fetch the catalog item's fields. Please provide the details you want in the request." 
                ch = state.session_state.get("chat_history", [])
                ch.append(("Bot", state.response))
                state.session_state["chat_history"] = ch
                # drop suggestion so user can continue
                state.session_state.pop("catalog_suggestion", None)
                state.session_state.pop("awaiting_slot_collection", None)
                return state
        elif any(v in reply for v in no_vals):
            # User declined the suggested catalog item -> clear suggestion and continue with normal clarify flow
            state.session_state.pop("awaiting_catalog_confirmation", None)
            state.session_state.pop("catalog_suggestion", None)
            state.response = "Okay — let's collect some details so I can create the appropriate request."
            ch = state.session_state.get("chat_history", [])
            ch.append(("Bot", state.response))
            state.session_state["chat_history"] = ch
            return state
    # 2) Greeting quick short-circuit (transient)
    if re.search(r"^(hi|hello|hey|good\s(morning|afternoon|evening)|thanks|thank you)\b", msg, re.IGNORECASE) or (len(msg.split()) <= 3 and any(w in lp for w in ["hi","hello","thanks","thank"])):
        state.response = "Hello! How can I assist you with your IT needs today?"
        state.solved = True
        state.session_state["last_was_greeting"] = True
        ch = state.session_state.get("chat_history", [])
        ch.append(("Bot", state.response))
        state.session_state["chat_history"] = ch
        return state
    # 3) If user explicitly asked to raise/create a request, set type immediately
    explicit_request_regex = r"\b(raise (a )?request|i want to raise( a)? request|i want to request|please create (a )?request|open (a )?request|create request|raise request|request access|i need access|i want to raise a request)\b"
    if re.search(explicit_request_regex, lp, re.IGNORECASE):
        state.session_state["type"] = "request"
    # 3b) Keyword-based detection for access/sharepoint/software intents -> prefer request
    if ("sharepoint" in lp or "share point" in lp) and any(k in lp for k in ["access", "permission", "folder", "site"]):
        state.session_state["type"] = "request"
    if any(kw in lp for kw in ["install", "installation", "provision", "software", "deploy"]) and any(kw in lp for kw in ["install", "request", "need"]):
        state.session_state["type"] = "request"
    # 4) If we already have an explicit type in session_state, honor it
    if state.session_state.get("type") in ["request", "incident"]:
        # leave classification to downstream nodes but ensure state.classification exists
        if not state.classification:
            state.classification = {"type": state.session_state.get("type")}
        return state
    # 5) Otherwise call LLM-based classifier to get richer classification
    try:
        classification = agentic_classification(get_original_user_problem(state.session_state, msg))
        state.classification = classification
    except Exception as e:
        print(f"[ERROR] Orchestrator classification failed: {e}")
        # fallback
        state.classification = {"type": "incident", "category": "other"}
    # Proactive catalog lookup: if this looks like a request and we haven't collected slots yet,
    # try extracting compact keywords and search the ServiceNow catalog. If we find a candidate,
    # set session_state['catalog_suggestion'] and ['awaiting_catalog_confirmation'] and prompt the user.
    try:
        type_ = state.session_state.get('type') or (state.classification.get('type') if isinstance(state.classification, dict) else None)
        pending = state.session_state.get('pending_slots') or []
        # Only run proactive catalog selection for requests with no pending slot collection and no existing suggestion
        if type_ == 'request' and not state.session_state.get('clarified') and not pending and not state.session_state.get('catalog_suggestion'):
            user_msg = get_original_user_problem(state.session_state, state.message) or state.message or ''
            # Fetch active catalog items (limited) from ServiceNow
            try:
                params = {"sysparm_query": "active=true", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 200}
                candidates = _sn_table_get('sc_cat_item', params)
            except Exception as e:
                print(f"[WARN] Failed to fetch catalog list: {e}")
                candidates = []
            # If we have candidates, ask the LLM to pick the best match from the small list
            if candidates:
                # Build a compact numbered list for the LLM
                compact = []
                for i, it in enumerate(candidates):
                    compact.append({"idx": i, "sys_id": it.get('sys_id'), "name": it.get('name'), "short_description": it.get('short_description')})
                pick_instr = (
                    "Agent: Catalog-Selector — You will be given a user's request and a JSON array of catalog items (idx, sys_id, name, short_description). "
                    "Return ONLY a JSON object: {\"pick_idx\": <index> } where pick_idx is the index of the best matching item (0-based). If none match, return {\"pick_idx\": -1 }."
                )
                prompt = f"{pick_instr}\nUSER_REQUEST: {user_msg}\nCATALOG_ITEMS: {json.dumps(compact[:80])}"
                try:
                    sel_out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
                    m = re.search(r"\{[\s\S]*?\}", sel_out)
                    pick_idx = -1
                    if m:
                        try:
                            j = json.loads(m.group(0))
                            pick_idx = int(j.get('pick_idx', -1))
                        except Exception:
                            pick_idx = -1
                    # Fallback: try to parse an integer from plain output
                    if pick_idx == -1:
                        m2 = re.search(r"(-?\d+)", sel_out)
                        if m2:
                            try:
                                pick_idx = int(m2.group(1))
                            except Exception:
                                pick_idx = -1
                except Exception as e:
                    print(f"[WARN] Catalog selector LLM failed: {e}")
                    pick_idx = -1
                # If a valid pick, attach suggestion and ask user to confirm
                if isinstance(pick_idx, int) and pick_idx >= 0 and pick_idx < len(candidates):
                    chosen = candidates[pick_idx]
                    state.session_state['catalog_suggestion'] = {'sys_id': chosen.get('sys_id'), 'name': chosen.get('name'), 'short_description': chosen.get('short_description')}
                    state.session_state['awaiting_catalog_confirmation'] = True
                    state.response = f"I found a catalog item that may match your request: {chosen.get('name')}. Reply 'yes' to use it or 'no' to continue."
                    ch = state.session_state.get('chat_history', [])
                    ch.append(('Bot', state.response))
                    state.session_state['chat_history'] = ch
                    return state
            # If no candidates or no good pick, fall back to keyword-based lookup
            kws = _llm_extract_catalog_keywords(user_msg, max_keywords=3)
            suggestion = None
            for k in kws:
                if not k:
                    continue
                suggestion = _sn_find_catalog_item(k)
                if suggestion:
                    break
            if not suggestion:
                suggestion = _sn_find_catalog_item(user_msg)
            if suggestion:
                state.session_state['catalog_suggestion'] = suggestion
                state.session_state['awaiting_catalog_confirmation'] = True
                state.response = f"I found a catalog item that may match your request: {suggestion.get('name')}. Reply 'yes' to use it or 'no' to continue." 
                ch = state.session_state.get('chat_history', [])
                ch.append(('Bot', state.response))
                state.session_state['chat_history'] = ch
                return state
    except Exception as e:
        # Special: start conversational catalog slot collection when message begins with START_CONVERSATIONAL_CATALOG
        if msg.startswith('START_CONVERSATIONAL_CATALOG'):
            parts = msg.split()
            if len(parts) >= 2:
                item_sys_id = parts[1]
                # fetch item variables
                try:
                    vars_resp = _sn_table_get('item_option_new', params={'sysparm_query': f'cat_item={item_sys_id}', 'sysparm_limit': 200})
                    variables = []
                    for v in (vars_resp.get('result') or []):
                        # normalize variable metadata
                        variables.append({
                            'id': v.get('sys_id'),
                            'name': v.get('element'),
                            'label': v.get('question_text') or v.get('name') or v.get('element'),
                            'type': v.get('type') or 'string',
                            'mandatory': v.get('mandatory') == 'true' or v.get('mandatory') == True,
                            'default': v.get('default_value'),
                            'reference_table': v.get('reference'),
                            'choices': [],
                            'lookup': bool(v.get('reference'))
                        })
                    # build slot questions (simple phrasing)
                    slot_questions = { sv['name']: sv['label'] for sv in variables }
                    pending_slots = [ sv['name'] for sv in variables if sv.get('mandatory') or True ]
                    # store in session_state
                    state.session_state['pending_slots'] = pending_slots
                    state.session_state['slot_questions'] = slot_questions
                    state.session_state['pending_question'] = slot_questions[pending_slots[0]] if pending_slots else None
                    state.session_state['awaiting_slot_collection'] = True
                    state.session_state['catalog_suggestion'] = {'item': {'sys_id': item_sys_id}}
                    # respond with first question
                    state.response = state.session_state['pending_question'] or 'I will ask you a few questions about this request.'
                    return state
                except Exception as e:
                    state.response = 'I could not fetch the catalog item fields. Please try the inline form.'
                    return state
        print(f"[WARN] proactive catalog lookup failed: {e}")
    return state


def decide_after_orchestrator(state: ITSMState):
    """Route from orchestrator to the appropriate node based on session_state and classification."""
    # If orchestrator already set a response and marked solved, END immediately
    # This handles greetings and other short-circuit cases properly
    if state.solved and state.response:
        return END
    # status check explicit
    if state.session_state.get("ticket_check_number"):
        return "status_check"
    # greeting short-circuit (backup check)
    if state.session_state.get("last_was_greeting"):
        msg = (state.message or "").strip()
        if re.search(r"^(hi|hello|hey|good\s(morning|afternoon|evening)|thanks|thank you)\b", msg, re.IGNORECASE) or (len(msg.split()) <= 3 and any(w in msg.lower() for w in ["hi","hello","thanks","thank"])):
            return END
        # clear transient and continue
        state.session_state.pop("last_was_greeting", None)
    # If explicit request type or classification suggests request, go to clarify (to gather slots) or ticketing
    type_ = state.session_state.get("type") or (state.classification.get("type") if isinstance(state.classification, dict) else None)
    # If we're waiting for the user's yes/no to a catalog suggestion, pause here until they reply
    if state.session_state.get("awaiting_catalog_confirmation"):
        return END
    # If we're waiting for the user's yes/no to run a troubleshoot flow, pause here until they reply
    if state.session_state.get("awaiting_troubleshoot_consent"):
        # Route to the troubleshooting node so that it can interpret the user's yes/no reply
        return "troubleshooting"
    # If we're waiting for explicit feedback about troubleshooting steps, route to troubleshooting
    # This ensures short replies like "not working" are handled by the Troubleshooting node
    # instead of being re-classified or routed to ticketing prematurely.
    if state.session_state.get("awaiting_feedback"):
        return "troubleshooting"
    # If we're waiting for resolution or escalation confirmation, route to troubleshooting
    if state.session_state.get("awaiting_resolution_confirmation") or state.session_state.get("awaiting_escalation_confirmation"):
        return "troubleshooting"
    # If escalation was explicitly confirmed (user replied 'yes'), route directly to ticketing
    # This ensures incidents created via an explicit escalation flow reach the ticketing node.
    if state.session_state.get("escalation_confirmed"):
        return "ticketing"
    # Safely extract category
    category = ""
    if isinstance(state.classification, dict):
        category = (state.classification.get("category") or "").lower()
    # If session state indicates request or classification suggests request
    if type_ == "request" or category in ["access", "installation", "request"]:
        # If we have pending_slots or awaiting_slot_collection, prioritize clarify
        if state.session_state.get('pending_slots') or state.session_state.get('awaiting_slot_collection'):
            return "clarify"
        # If not clarified, ask clarifying questions first
        if not state.session_state.get("clarified"):
            return "clarify"
        return "ticketing"
    # Else, treat as troubleshooting/incident
    return "troubleshooting"

graph = StateGraph(ITSMState)
graph.add_node("greeting", node_greeting)
graph.add_node("orchestrator", node_orchestrator)
graph.add_node("classification", node_classification)
graph.add_node("clarify", node_clarify)
graph.add_node("troubleshooting", node_troubleshooting)
graph.add_node("routing", node_classification)  # reuse classification for routing/categorization
graph.add_node("ticketing", node_ticketing)
graph.add_node("already_escalated", node_already_escalated)
graph.add_node("status_check", node_status_check)
# Set orchestrator as the new entry point
graph.set_entry_point("orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    decide_after_orchestrator,
    {"status_check": "status_check", "clarify": "clarify", "ticketing": "ticketing", "troubleshooting": "troubleshooting", END: END}
)
# Keep existing greeting edge for compatibility (orchestrator handles greeting but keep node available)
graph.add_conditional_edges(
    "greeting",
    decide_after_greeting,
    {END: END, "classification": "classification"}
)
graph.add_conditional_edges(
    "classification",
    decide_after_classification,
    {
        "ticketing": "ticketing",
        "troubleshooting": "troubleshooting",
        "status_check": "status_check",
        "clarify": "clarify"
    }
)
graph.add_conditional_edges(
    "troubleshooting",
    decide_after_troubleshooting,
    {
        END: END,
        "routing": "routing"
    }
)
graph.add_conditional_edges(
    "routing",
    decide_after_routing,
    {
        "ticketing": "ticketing"
    }
)
graph.add_edge("ticketing", END)
graph.add_edge("already_escalated", END)
itsm_workflow = graph.compile()


# --- ServiceNow Catalog helpers & API endpoints ---
# Ensure the FastAPI app is created before any route decorators are evaluated.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # Development-friendly permissive CORS. For production, lock this down to explicit origins.
    allow_origins=["*"],
    # When allow_origins is set to ['*'] it's safer to disable credentials; enable if you need cookies/auth.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include RAG API routes if available
if RAG_ENABLED and rag_router:
    app.include_router(rag_router)
    print("[RAG] RAG API endpoints registered at /api/rag/*")

# Include ITOps API routes if available
if ITOPS_ENABLED and itops_router:
    app.include_router(itops_router)
    print("[ITOps] ITOps API endpoints registered at /api/itops/*")

# Serve built frontend if available. The Dockerfile copies the frontend build into ./static
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
ALT_DIST = os.path.join(BASE_DIR, "askgen-react", "dist")
STATIC_SOURCE = None
if os.path.isdir(STATIC_DIR):
    STATIC_SOURCE = STATIC_DIR
elif os.path.isdir(ALT_DIST):
    STATIC_SOURCE = ALT_DIST

if STATIC_SOURCE:
    # Mount the static files under /static to avoid shadowing API routes like /api
    app.mount("/static", StaticFiles(directory=STATIC_SOURCE), name="static")


@app.get("/",
         include_in_schema=False)
def root_index():
    """Serve the frontend index if present, otherwise simple status JSON."""
    if STATIC_SOURCE:
        index_path = os.path.join(STATIC_SOURCE, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
    return JSONResponse({"status": "ok", "frontend_available": bool(STATIC_SOURCE)})


@app.get('/api/status')
def api_status():
    """Simple health endpoint used by container healthchecks and platforms.
    Returns 200 when the app is healthy.
    """
    return JSONResponse({"status": "ok"})


@app.get('/api/integrations/status')
def integrations_status():
    """Check connection status for all integrations."""
    from datetime import datetime
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "integrations": {}
    }
    
    # Test ServiceNow connection
    try:
        url = f"{SERVICE_NOW_BASE}/sys_properties?sysparm_limit=1"
        resp = SESSION.get(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), 
                          headers={"Accept": "application/json"}, timeout=10, verify=VERIFY_TLS)
        if resp.status_code == 200:
            results["integrations"]["servicenow"] = {
                "status": "connected",
                "instance": SERVICE_NOW_INSTANCE,
                "message": "ServiceNow connection successful",
                "response_time_ms": int(resp.elapsed.total_seconds() * 1000)
            }
        else:
            results["integrations"]["servicenow"] = {
                "status": "error",
                "instance": SERVICE_NOW_INSTANCE,
                "message": f"ServiceNow returned status {resp.status_code}",
                "error_code": resp.status_code
            }
    except Exception as e:
        results["integrations"]["servicenow"] = {
            "status": "disconnected",
            "instance": SERVICE_NOW_INSTANCE,
            "message": f"Connection failed: {str(e)}",
            "error": str(type(e).__name__)
        }
    
    # Test Azure OpenAI connection
    try:
        test_resp = query_azure_openai("Say 'ok'", system="You are a test. Reply with only 'ok'.", temperature=0)
        if test_resp and "ok" in test_resp.lower():
            results["integrations"]["azure_openai"] = {
                "status": "connected",
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "Not configured"),
                "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "Not configured"),
                "message": "Azure OpenAI connection successful"
            }
        else:
            results["integrations"]["azure_openai"] = {
                "status": "warning",
                "message": "Azure OpenAI responded but may have issues"
            }
    except Exception as e:
        results["integrations"]["azure_openai"] = {
            "status": "disconnected",
            "message": f"Connection failed: {str(e)}",
            "error": str(type(e).__name__)
        }
    
    # Test RAG Engine
    try:
        if RAG_ENABLED:
            rag = get_rag_engine()
            results["integrations"]["rag_engine"] = {
                "status": "connected",
                "message": "RAG Engine is active and ready"
            }
        else:
            results["integrations"]["rag_engine"] = {
                "status": "disconnected",
                "message": "RAG Engine not available"
            }
    except Exception as e:
        results["integrations"]["rag_engine"] = {
            "status": "error",
            "message": f"RAG Engine error: {str(e)}"
        }
    
    # Test ITOps Engine
    try:
        if ITOPS_ENABLED:
            auto_resolver = get_auto_resolver()
            results["integrations"]["itops_engine"] = {
                "status": "connected",
                "message": "ITOps Engine is active with all components",
                "components": {
                    "auto_resolver": True,
                    "agent_assist": True,
                    "triage_engine": True,
                    "incident_correlator": True,
                    "learning_engine": True
                }
            }
        else:
            results["integrations"]["itops_engine"] = {
                "status": "disconnected",
                "message": "ITOps Engine not available"
            }
    except Exception as e:
        results["integrations"]["itops_engine"] = {
            "status": "error",
            "message": f"ITOps Engine error: {str(e)}"
        }
    
    return JSONResponse(results)


# Fallback for client-side routing is declared at the end of this file (moved there so explicit API routes
# are registered first). See bottom-of-file for implementation.

def _sn_table_get(table: str, params: dict | None = None):
    """Helper to query ServiceNow table API and return result list."""
    url = f"{SERVICE_NOW_BASE}/{table}"
    try:
        resp = SESSION.get(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), params=params or {}, headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
        resp.raise_for_status()
        return resp.json().get('result', [])
    except Exception:
        raise


def _sn_find_catalog_item(query: str):
    """Find a single best-matching catalog item by name/description. Returns compact item or None."""
    try:
        if not query:
            return None
        q = f"active=true^NQnameLIKE{query}^ORshort_descriptionLIKE{query}"
        params = {"sysparm_query": q, "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 1}
        items = _sn_table_get('sc_cat_item', params)
        if not items:
            return None
        it = items[0]
        return {'sys_id': it.get('sys_id'), 'name': it.get('name'), 'short_description': it.get('short_description')}
    except Exception:
        return None


def _llm_extract_catalog_keywords(user_message: str, max_keywords: int = 3):
    """Ask the LLM to produce 1-3 short keywords suitable for searching the ServiceNow catalog.
    Returns a list of keywords (strings) or an empty list on failure.
    """
    if not user_message or not user_message.strip():
        return []
    instr = (
        "Agent: Catalog-Keyword-Extractor — Given a user's request, return a JSON array of 1 to "
        f"{max_keywords} short search keywords (single words or short phrases) that best match catalog item names. "
        "Respond ONLY with the JSON array, no prose."
    )
    prompt = f"{instr}\nUser request: {user_message}"
    try:
        out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
        # Prefer strict JSON array; try to extract using regex
        m = re.search(r"\[[\s\S]*?\]", out)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                # coerce to short strings and limit size
                kws = [str(x).strip() for x in arr if str(x).strip()]
                return kws[:max_keywords]
        # Fallback: comma-split if the model returned a plain comma-separated string
        parts = [p.strip() for p in re.split(r"[,;]\s*|\s{2,}", out) if p.strip()]
        if parts:
            return parts[:max_keywords]
    except Exception as e:
        print(f"[WARN] Keyword extractor failed: {e}\nRaw output: {out if 'out' in locals() else ''}")
    return []


def _llm_pick_catalog_index(items: list, user_message: str) -> int:
    """Ask the LLM to pick the best-matching catalog item index from a provided list.
    Returns 0-based index, or -1 if no good match.
    """
    if not items:
        return -1
    # Prepare a compact JSON list for the LLM with only name and short_description
    compact = []
    for it in items[:200]:
        compact.append({
            "sys_id": it.get("sys_id"),
            "name": (it.get("name") or "")[:180],
            "short_description": (it.get("short_description") or "")[:300]
        })
    instr = (
        "Agent: Catalog-Selector — Given an array of catalog items (index, name, short_description) and a user's request, "
        "return ONLY the 0-based integer index of the single best-matching item, or -1 if none match. "
        "Do NOT include any extra text or explanation."
    )
    prompt = f"{instr}\nITEMS: {json.dumps(compact)}\nUser request: {user_message}"
    try:
        out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
        # Try to extract an integer from the response
        m = re.search(r"(-?\d+)", out)
        if m:
            idx = int(m.group(1))
            if idx < 0:
                return -1
            if idx >= len(items):
                return -1
            return idx
    except Exception as e:
        print(f"[WARN] _llm_pick_catalog_index failed: {e}")
    return -1


@app.get('/catalog/items')
def catalog_items(query: str = '', page: int = 1, limit: int = 50):
    """List catalog items (sc_cat_item) with optional search query."""
    try:
        q = "active=true"
        if query:
            # search name or short_description
            q += f"^NQnameLIKE{query}^ORshort_descriptionLIKE{query}"
        params = {"sysparm_query": q, "sysparm_fields": "sys_id,name,short_description,category", "sysparm_limit": limit, "sysparm_offset": (page - 1) * limit}
        items = _sn_table_get('sc_cat_item', params)
        out = []
        for it in items:
            out.append({
                "sys_id": it.get('sys_id'),
                "name": it.get('name'),
                "short_description": it.get('short_description'),
                "category": it.get('category')
            })
        return {'items': out, 'total': len(out), 'page': page}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/catalog/item/{item_id}')
def catalog_item(item_id: str):
    """Return a catalog item metadata and its variables (item_option_new)."""
    try:
        params = {"sysparm_query": f"sys_id={item_id}", "sysparm_exclude_reference_link": True}
        items = _sn_table_get('sc_cat_item', params)
        if not items:
            raise HTTPException(status_code=404, detail='Catalog item not found')
        item = items[0]
        # fetch variables for this catalog item
        # include common ServiceNow variable identifier fields (variable_name, element, name)
        vparams = {"sysparm_query": f"cat_item={item_id}", "sysparm_fields": "sys_id,question,question_text,variable_name,element,name,type,order,mandatory,default_value,reference,choice", "sysparm_limit": 300}
        vars = _sn_table_get('item_option_new', vparams)
        # Some instances attach variables via variable-sets mapped through sc_item_option_mtom.
        # If item_option_new returns few or no variables, try the mapping table and fetch referenced variables.
        try:
            if (not vars) or (isinstance(vars, list) and len(vars) < 4):
                mtom_params = {"sysparm_query": f"cat_item={item_id}", "sysparm_fields": "item_option,item_option_new,variable", "sysparm_limit": 300}
                mtom = _sn_table_get('sc_item_option_mtom', mtom_params)
                ids = []
                for m in mtom:
                    # mapping records may store the linked variable under different keys; try common ones
                    for k in ('item_option', 'item_option_new', 'variable'):
                        vref = m.get(k)
                        if isinstance(vref, dict):
                            # sometimes ServiceNow returns an object with value/display_value
                            val = vref.get('value') or vref.get('sys_id') or vref.get('id')
                        else:
                            val = vref
                        if val:
                            ids.append(str(val))
                if ids:
                    # uniq and attempt to resolve sc_item_option -> item_option_new mapping
                    ids = list(dict.fromkeys(ids))
                    # First try to fetch item_option_new directly by those ids
                    q = 'sys_idIN' + ','.join(ids)
                    candidate_vars = _sn_table_get('item_option_new', {"sysparm_query": q, "sysparm_fields": vparams['sysparm_fields'], "sysparm_limit": 300})
                    sc_map = {}
                    if candidate_vars and len(candidate_vars) > 0:
                        vars = candidate_vars
                    else:
                        # If not found, these ids may be sc_item_option records; fetch them and extract linked variable/item_option_new ids
                        try:
                            sc_rows = _sn_table_get('sc_item_option', {"sysparm_query": 'sys_idIN' + ','.join(ids), "sysparm_fields": "sys_id,item_option_new,variable,question,element,variable_name,question_text", "sysparm_limit": 300})
                            linked = []
                            for sr in sc_rows:
                                # attempt to map item_option_new/variable -> sc_row for later merging
                                for fk in ('item_option_new', 'variable'):
                                    vref = sr.get(fk)
                                    if isinstance(vref, dict):
                                        val = vref.get('value') or vref.get('sys_id') or vref.get('id')
                                    else:
                                        val = vref
                                    if val:
                                        linked.append(str(val))
                                        # record mapping so we can merge sc_row fields into the final variable metadata
                                        sc_map[str(val)] = sr
                                # Also consider if the mapping directly contains a variable_name/element we can use
                                for fk in ('variable_name', 'element', 'question', 'question_text'):
                                    if sr.get(fk) and not sc_map.get(sr.get('item_option_new') if isinstance(sr.get('item_option_new'), str) else None):
                                        pass
                            if linked:
                                linked = list(dict.fromkeys(linked))
                                q2 = 'sys_idIN' + ','.join(linked)
                                vars = _sn_table_get('item_option_new', {"sysparm_query": q2, "sysparm_fields": vparams['sysparm_fields'], "sysparm_limit": 300})
                        except Exception:
                            pass
        except Exception:
            # non-fatal: keep original vars if mapping lookup fails
            pass
        variables = []
        for v in vars:
            raw_type = v.get('type')
            choices = None
            # parse comma-separated choices when present
            if v.get('choice'):
                raw = v.get('choice')
                if isinstance(raw, str):
                    choices = [{'value': c.strip(), 'label': c.strip()} for c in raw.split(',') if c.strip()]
            # If no inline choice string and the variable type code indicates a choice (ServiceNow type '3'),
            # try to fetch choice rows from the item_option_new_choice table.
            if not choices:
                try:
                    rt = str(raw_type or '').strip().lower()
                    if rt == '3' or rt in ('choice', 'select', 'radio'):
                        # attempt a few possible foreign-key fields used by different instances
                        qopts = [f"question={v.get('sys_id')}", f"variable={v.get('sys_id')}", f"item_option_new={v.get('sys_id')}"]
                        choice_rows = []
                        for q in qopts:
                            try:
                                cr = _sn_table_get('item_option_new_choice', {"sysparm_query": q, "sysparm_fields": "sys_id,label,value,display_value", "sysparm_limit": 300})
                                if cr:
                                    choice_rows = cr
                                    break
                            except Exception:
                                continue
                        if choice_rows:
                            choices = []
                            for cr in choice_rows:
                                # Normalize possible object values returned by ServiceNow (e.g., {value, display_value})
                                raw_val = cr.get('value')
                                if isinstance(raw_val, dict):
                                    val = raw_val.get('value') or raw_val.get('sys_id') or raw_val.get('display_value') or ''
                                    dv = raw_val.get('display_value') or ''
                                else:
                                    val = raw_val or cr.get('sys_id') or ''
                                    dv = ''
                                lab = cr.get('label') or cr.get('display_value') or cr.get('text') or dv or val
                                choices.append({'value': str(val), 'label': str(lab)})
                        # Fallback: some instances store choices in sys_choice table
                            if not choices:
                                try:
                                    elem = v.get('element') or v.get('variable_name') or var_name
                                    if elem:
                                        scq = f"element={elem}"
                                        cr2 = _sn_table_get('sys_choice', {"sysparm_query": scq, "sysparm_fields": "label,value,display_value", "sysparm_limit": 200})
                                        if cr2:
                                            choices = []
                                            for sc in cr2:
                                                raw_val2 = sc.get('value')
                                                if isinstance(raw_val2, dict):
                                                    val2 = raw_val2.get('value') or raw_val2.get('display_value') or ''
                                                    dv2 = raw_val2.get('display_value') or ''
                                                else:
                                                    val2 = raw_val2 or ''
                                                    dv2 = ''
                                                lab2 = sc.get('label') or sc.get('display_value') or dv2 or val2
                                                choices.append({'value': str(val2), 'label': str(lab2)})
                                except Exception:
                                    pass
                except Exception:
                    pass
            reference_table = v.get('reference') or v.get('reference_table')
            # Friendly type: choice > reference > raw type > string
            if choices:
                v_type = 'choice'
            elif reference_table:
                v_type = 'reference'
            else:
                v_type = str(raw_type) if raw_type is not None else 'string'
            # Try to merge any sc_item_option mapping for richer labels/names
            sc_row = sc_map.get(v.get('sys_id')) if 'sc_map' in locals() else None
            # Stable name synthesis: prefer explicit identifiers then question text, otherwise fallback to sc_row values or sys_id
            var_name = v.get('variable_name') or v.get('element') or v.get('name') or v.get('question') or v.get('question_text')
            if not var_name and sc_row:
                var_name = sc_row.get('variable_name') or sc_row.get('element') or sc_row.get('question') or sc_row.get('question_text')
            if not var_name:
                var_name = v.get('sys_id')
            label = (v.get('question_text') or v.get('question') or (sc_row.get('question_text') if sc_row else None) or str(var_name))
            variables.append({
                'id': v.get('sys_id'),
                'name': str(var_name),
                'label': label,
                'type': v_type,
                'mandatory': bool(v.get('mandatory')),
                'default': v.get('default_value') or v.get('default') or '',
                'reference_table': reference_table,
                'choices': choices,
                'lookup': True if reference_table else False
            })
        # Special-case: include known variable-set variables for this catalog item if present
        try:
            # If this catalog item specifically uses the 'it_to_it' variable set, include its variables
            # (sys_id provided from debugging: b682f39a4f334200086eeed18110c791)
            if item.get('sys_id') == '67e2f2da4fff0200086eeed18110c7dd':
                try:
                    vs_id = 'b682f39a4f334200086eeed18110c791'
                    vparams2 = {"sysparm_query": f"variable_set={vs_id}", "sysparm_fields": "sys_id,variable_name,element,name,question,question_text,type,mandatory,default_value,reference,choice", "sysparm_limit": 200}
                    vs_vars = _sn_table_get('item_option_new', vparams2)
                    for v in vs_vars:
                            # normalize like above, but prefer variable_set's question_text/name
                            raw_type = v.get('type')
                            choices = None
                            if v.get('choice'):
                                raw = v.get('choice')
                                if isinstance(raw, str):
                                    choices = [{'value': c.strip(), 'label': c.strip()} for c in raw.split(',') if c.strip()]
                            reference_table = v.get('reference') or v.get('reference_table')
                            # Determine type: interpret ServiceNow numeric code '3' as choice
                            rt = str(raw_type or '').strip().lower()
                            if choices:
                                v_type = 'choice'
                            elif reference_table:
                                v_type = 'reference'
                            elif rt == '3' or rt in ('choice', 'select', 'radio'):
                                v_type = 'choice'
                            else:
                                v_type = str(raw_type) if raw_type is not None else 'string'

                            # If this is a choice type but we don't yet have choices, try loading item_option_new_choice / sys_choice
                            if v_type == 'choice' and not choices:
                                try:
                                    vid = v.get('sys_id')
                                    qopts = [f"question={vid}", f"variable={vid}", f"item_option_new={vid}"]
                                    choice_rows = []
                                    for q in qopts:
                                        try:
                                            cr = _sn_table_get('item_option_new_choice', {"sysparm_query": q, "sysparm_fields": "sys_id,label,value,display_value", "sysparm_limit": 300})
                                            if cr:
                                                choice_rows = cr
                                                break
                                        except Exception:
                                            continue
                                    if choice_rows:
                                        choices = []
                                        for cr in choice_rows:
                                            raw_val = cr.get('value')
                                            if isinstance(raw_val, dict):
                                                val = raw_val.get('value') or raw_val.get('sys_id') or raw_val.get('display_value') or ''
                                                dv = raw_val.get('display_value') or ''
                                            else:
                                                val = raw_val or cr.get('sys_id') or ''
                                                dv = ''
                                            lab = cr.get('label') or cr.get('display_value') or cr.get('text') or dv or val
                                            choices.append({'value': str(val), 'label': str(lab)})
                                        # fallback: try question_choice table (some instances store choices there)
                                        if not choices:
                                            try:
                                                qch = _sn_table_get('question_choice', {"sysparm_query": f"question={vid}^active=true", "sysparm_fields": "value,text,sequence,active", "sysparm_display_value": True, "sysparm_limit": 500})
                                                if qch:
                                                    choices = []
                                                    for r in qch:
                                                        raw_val = r.get('value')
                                                        if isinstance(raw_val, dict):
                                                            val = raw_val.get('value') or raw_val.get('sys_id') or raw_val.get('display_value') or ''
                                                            dv = raw_val.get('display_value') or ''
                                                        else:
                                                            val = raw_val or r.get('sys_id') or ''
                                                            dv = ''
                                                        lab = r.get('text') or r.get('display_value') or dv or val
                                                        seq = r.get('sequence') or r.get('order') or 0
                                                        try:
                                                            seq_num = int(seq) if seq is not None and str(seq).isdigit() else 0
                                                        except Exception:
                                                            seq_num = 0
                                                        active_flag = False
                                                        a = r.get('active')
                                                        if isinstance(a, bool):
                                                            active_flag = a
                                                        else:
                                                            active_flag = str(a).lower() in ('true','1','yes')
                                                        choices.append({'value': str(val), 'label': str(lab), 'sequence': seq_num, 'active': bool(active_flag)})
                                                    choices = sorted(choices, key=lambda x: x.get('sequence', 0))
                                            except Exception:
                                                pass
                                    # fallback to sys_choice
                                    if not choices:
                                        try:
                                            elem = v.get('element') or v.get('variable_name')
                                            if elem:
                                                scq = f"element={elem}"
                                                sc_rows = _sn_table_get('sys_choice', {"sysparm_query": scq, "sysparm_fields": "label,value,display_value", "sysparm_limit": 200})
                                                if sc_rows:
                                                    choices = []
                                                    for sc in sc_rows:
                                                        raw_val2 = sc.get('value')
                                                        if isinstance(raw_val2, dict):
                                                            val2 = raw_val2.get('value') or raw_val2.get('display_value') or ''
                                                            dv2 = raw_val2.get('display_value') or ''
                                                        else:
                                                            val2 = raw_val2 or ''
                                                            dv2 = ''
                                                        lab2 = sc.get('label') or sc.get('display_value') or dv2 or val2
                                                        choices.append({'value': str(val2), 'label': str(lab2)})
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            var_name = v.get('variable_name') or v.get('element') or v.get('name') or v.get('question') or v.get('question_text') or v.get('sys_id')
                            if not var_name:
                                var_name = v.get('sys_id')
                            label = (v.get('question_text') or v.get('question') or str(var_name))
                            # If choices still empty for choice-type variables, try resolve_variable_choices
                            if v_type == 'choice' and not choices:
                                try:
                                    resp = resolve_variable_choices(variable_sys_id=v.get('sys_id'))
                                    if isinstance(resp, dict):
                                        fetched = resp.get('choices') or []
                                        if fetched:
                                            norm = []
                                            for c in fetched:
                                                if isinstance(c, dict):
                                                    val = c.get('value') or c.get('label') or ''
                                                    lab = c.get('label') or c.get('value') or ''
                                                else:
                                                    # primitive value returned
                                                    val = str(c)
                                                    lab = str(c)
                                                norm.append({'value': str(val), 'label': str(lab)})
                                            choices = norm
                                except Exception:
                                    pass

                            # If still no choices, mark dynamic; otherwise include normalized choices
                            if v_type == 'choice' and not choices:
                                variables.append({
                                    'id': v.get('sys_id'),
                                    'name': str(var_name),
                                    'label': label,
                                    'type': v_type,
                                    'mandatory': bool(v.get('mandatory')),
                                    'default': v.get('default_value') or v.get('default') or '',
                                    'reference_table': reference_table,
                                    'choices': None,
                                    'lookup': True if reference_table else False,
                                    'source': 'variable_set',
                                    'dynamic': True,
                                    'sys_id': v.get('sys_id')
                                })
                            else:
                                variables.append({
                                    'id': v.get('sys_id'),
                                    'name': str(var_name),
                                    'label': label,
                                    'type': v_type,
                                    'mandatory': bool(v.get('mandatory')),
                                    'default': v.get('default_value') or v.get('default') or '',
                                    'reference_table': reference_table,
                                    'choices': choices,
                                    'lookup': True if reference_table else False,
                                    'source': 'variable_set',
                                    'sys_id': v.get('sys_id')
                                })
                except Exception:
                    pass
        except Exception:
            pass
        return {'item': {'sys_id': item.get('sys_id'), 'name': item.get('name'), 'short_description': item.get('short_description')}, 'variables': variables}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# Simple in-memory cache for catalog variable payloads: { catItemId: { 'ts': epoch, 'data': [...] } }
CATALOG_VARIABLES_CACHE: dict = {}
CATALOG_VARIABLES_TTL = 300  # seconds (5 minutes)


@app.get('/catalog/{cat_item_id}/variables')
def catalog_variables(cat_item_id: str):
    """Return normalized variables for a catalog item.
    Primary source: Service Catalog API /api/sn_sc/servicecatalog/items/{catItemId}/variables
    Fallback: item_option_new + question_choice table queries.
    Caches results for a short period.
    """
    try:
        # Check cache
        now = time.time()
        cached = CATALOG_VARIABLES_CACHE.get(cat_item_id)
        if cached and now - cached.get('ts', 0) < CATALOG_VARIABLES_TTL:
            return {'item': {'sys_id': cat_item_id}, 'variables': cached['data'], 'cached': True}

        # 1) Try Service Catalog variables API
        sc_url = f"https://{SERVICE_NOW_INSTANCE}/api/sn_sc/servicecatalog/items/{cat_item_id}/variables"
        try:
            resp = SESSION.get(sc_url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
            resp.raise_for_status()
            sc_vars = resp.json().get('result', [])
        except Exception:
            sc_vars = []

        variables_out = []
        if sc_vars:
            # Normalize service catalog variables
            for v in sc_vars:
                name = v.get('name') or v.get('element') or v.get('variable_name')
                label = v.get('question_text') or v.get('question') or v.get('label') or name
                raw_type = str(v.get('type') or '').lower()
                if raw_type == '3' or raw_type in ('choice', 'select', 'radio'):
                    v_type = 'choice'
                elif raw_type in ('reference',):
                    v_type = 'reference'
                else:
                    v_type = 'string'

                choices = v.get('choices') or None
                # Normalize choices if present
                if choices and isinstance(choices, list):
                    norm_choices = []
                    for c in choices:
                        val = c.get('value') if isinstance(c, dict) else c
                        lab = c.get('label') if isinstance(c, dict) else c
                        norm_choices.append({'value': val, 'label': lab})
                    choices = sorted(norm_choices, key=lambda x: x.get('order') or 0)
                else:
                    choices = None

                var_obj = {
                    'name': name,
                    'label': label,
                    'type': v_type,
                    'mandatory': bool(v.get('mandatory')),
                }
                if v_type == 'choice' and choices:
                    var_obj['choices'] = choices
                if v_type == 'choice':
                    choices = []
                    try:
                        q = f"question={v.get('sys_id')}^active=true^ORDERBYsequence"
                        cr = _sn_table_get('question_choice', {"sysparm_query": q, "sysparm_fields": "label,value,sequence,active", "sysparm_display_value": True, "sysparm_limit": 500})
                        for r in (cr or []):
                            # Normalize possible object/value shapes
                            raw_val = r.get('value')
                            if isinstance(raw_val, dict):
                                val = raw_val.get('value') or raw_val.get('sys_id') or raw_val.get('display_value') or ''
                                dv = raw_val.get('display_value') or ''
                            else:
                                val = raw_val or r.get('sys_id') or ''
                                dv = ''
                            lab = r.get('label') or r.get('display_value') or dv or val
                            seq = r.get('sequence') or r.get('order') or 0
                            try:
                                seq_num = int(seq) if seq is not None and str(seq).isdigit() else 0
                            except Exception:
                                seq_num = 0
                            a = r.get('active')
                            if isinstance(a, bool):
                                active_flag = a
                            else:
                                active_flag = str(a).lower() in ('true','1','yes')
                            choices.append({'value': str(val), 'label': str(lab), 'order': seq_num, 'default': False, 'active': bool(active_flag)})
                    except Exception:
                        choices = []
                    # sort
                    choices = sorted(choices, key=lambda x: x.get('order') or 0)
                    if choices:
                        var_obj['choices'] = choices
                    else:
                        # mark dynamic if no static choices found
                        var_obj['dynamic'] = True
                        var_obj['sys_id'] = v.get('sys_id')

                # append the normalized var_obj (choices may have been populated above)
                variables_out.append(var_obj)

            # Special-case: include known variable-set variables for this catalog item if present
            try:
                if cat_item_id == '67e2f2da4fff0200086eeed18110c7dd':
                    try:
                        vs_id = 'b682f39a4f334200086eeed18110c791'
                        vparams2 = {"sysparm_query": f"variable_set={vs_id}", "sysparm_fields": "sys_id,variable_name,element,name,question,question_text,type,mandatory,default_value,reference,choice", "sysparm_limit": 200}
                        vs_vars = _sn_table_get('item_option_new', vparams2)
                        for v in vs_vars:
                            raw_type = v.get('type')
                            choices = None
                            if v.get('choice'):
                                raw = v.get('choice')
                                if isinstance(raw, str):
                                    choices = [{'value': c.strip(), 'label': c.strip()} for c in raw.split(',') if c.strip()]
                            reference_table = v.get('reference') or v.get('reference_table')
                            rt = str(raw_type or '').strip().lower()
                            if choices:
                                v_type = 'choice'
                            elif reference_table:
                                v_type = 'reference'
                            elif rt == '3' or rt in ('choice', 'select', 'radio'):
                                v_type = 'choice'
                            else:
                                v_type = str(raw_type) if raw_type is not None else 'string'

                            # Load choice rows when needed
                            if v_type == 'choice' and not choices:
                                try:
                                    vid = v.get('sys_id')
                                    qopts = [f"question={vid}", f"variable={vid}", f"item_option_new={vid}"]
                                    choice_rows = []
                                    for q in qopts:
                                        try:
                                            cr = _sn_table_get('item_option_new_choice', {"sysparm_query": q, "sysparm_fields": "sys_id,label,value,display_value", "sysparm_limit": 300})
                                            if cr:
                                                choice_rows = cr
                                                break
                                        except Exception:
                                            continue
                                    if choice_rows:
                                        choices = []
                                        for cr in choice_rows:
                                            raw_val = cr.get('value')
                                            if isinstance(raw_val, dict):
                                                val = raw_val.get('value') or raw_val.get('sys_id') or raw_val.get('display_value') or ''
                                                dv = raw_val.get('display_value') or ''
                                            else:
                                                val = raw_val or cr.get('sys_id') or ''
                                                dv = ''
                                            lab = cr.get('label') or cr.get('display_value') or cr.get('text') or dv or val
                                            choices.append({'value': str(val), 'label': str(lab)})
                                    if not choices:
                                        try:
                                            elem = v.get('element') or v.get('variable_name')
                                            if elem:
                                                scq = f"element={elem}"
                                                sc_rows = _sn_table_get('sys_choice', {"sysparm_query": scq, "sysparm_fields": "label,value,display_value", "sysparm_limit": 200})
                                                if sc_rows:
                                                    choices = []
                                                    for sc in sc_rows:
                                                        raw_val2 = sc.get('value')
                                                        if isinstance(raw_val2, dict):
                                                            val2 = raw_val2.get('value') or raw_val2.get('display_value') or ''
                                                            dv2 = raw_val2.get('display_value') or ''
                                                        else:
                                                            val2 = raw_val2 or ''
                                                            dv2 = ''
                                                        lab2 = sc.get('label') or sc.get('display_value') or dv2 or val2
                                                        choices.append({'value': str(val2), 'label': str(lab2)})
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            var_name = v.get('variable_name') or v.get('element') or v.get('name') or v.get('question') or v.get('question_text') or v.get('sys_id')
                            if not var_name:
                                var_name = v.get('sys_id')
                            label = (v.get('question_text') or v.get('question') or str(var_name))
                            # If choices still empty for a choice-type variable, try resolve_variable_choices helper
                            if v_type == 'choice' and not choices:
                                try:
                                    # Call the resolver which queries question_choice with retries and normalizes rows
                                    resp = resolve_variable_choices(variable_sys_id=v.get('sys_id'))
                                    if isinstance(resp, dict):
                                        fetched = resp.get('choices') or []
                                        if fetched:
                                            # Normalize to {value,label} shape
                                            norm = []
                                            for c in fetched:
                                                val = c.get('value') if isinstance(c, dict) else (c.get('value') if isinstance(c, dict) else str(c))
                                                lab = c.get('label') if isinstance(c, dict) else (c.get('label') if isinstance(c, dict) else str(c))
                                                # support resolver returning primitive objects
                                                if isinstance(c, dict):
                                                    val = c.get('value') or c.get('label') or ''
                                                    lab = c.get('label') or c.get('value') or ''
                                                norm.append({'value': str(val), 'label': str(lab)})
                                            choices = norm
                                except Exception:
                                    pass

                            # If no choices after resolution attempt, mark dynamic
                            if v_type == 'choice' and not choices:
                                var_obj = {
                                    'name': str(var_name),
                                    'label': label,
                                    'type': v_type,
                                    'mandatory': bool(v.get('mandatory')),
                                    'reference': reference_table,
                                    'choices': None,
                                    'sys_id': v.get('sys_id'),
                                    'dynamic': True
                                }
                            else:
                                var_obj = {
                                    'name': str(var_name),
                                    'label': label,
                                    'type': v_type,
                                    'mandatory': bool(v.get('mandatory')),
                                    'reference': reference_table,
                                    'choices': choices,
                                    'sys_id': v.get('sys_id')
                                }
                            variables_out.append(var_obj)
                    except Exception:
                        pass
            except Exception:
                pass

        # cache and return
        CATALOG_VARIABLES_CACHE[cat_item_id] = {'ts': now, 'data': variables_out}
        return {'item': {'sys_id': cat_item_id}, 'variables': variables_out, 'cached': False}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/catalog/{cat_item_id}/variables/{var_name}/choices')
def catalog_variable_choices(cat_item_id: str, var_name: str, body: dict | None = None):
    """Compute choices for a variable given form context. Uses a server-side dynamic helper if configured.
    Body: { "context": { "parentVar": "value", ... } }
    """
    try:
        ctx = (body or {}).get('context') or {}
        # Find the item_option_new definition for this variable
        try:
            q = f"cat_item={cat_item_id}^NQname={var_name}^ORelement={var_name}^ORvariable_name={var_name}"
            defs = _sn_table_get('item_option_new', {"sysparm_query": q, "sysparm_fields": "sys_id,name,element,variable_name,type,question_text,reference", "sysparm_limit": 5})
            def_row = defs[0] if defs else None
        except Exception:
            def_row = None

        if not def_row:
            return JSONResponse(status_code=404, content={'error': 'variable definition not found'})

        var_sys_id = def_row.get('sys_id')

        # If a dynamic helper endpoint is configured, call it
        helper_url = os.getenv('SERVICE_NOW_DYNAMIC_HELPER')
        if not helper_url:
            return JSONResponse(status_code=501, content={'error': 'dynamic helper not configured', 'note': 'set SERVICE_NOW_DYNAMIC_HELPER env var to a Scripted REST URL that accepts {cat_item, variable_sys_id, context}'} )

        payload = {'cat_item': cat_item_id, 'variable_sys_id': var_sys_id, 'var_name': var_name, 'context': ctx}
        try:
            hresp = SESSION.post(helper_url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json", "Content-Type": "application/json"}, json=payload, timeout=30, verify=VERIFY_TLS)
            hresp.raise_for_status()
            data = hresp.json().get('result') if isinstance(hresp.json(), dict) else hresp.json()
            # Expecting { choices: [ {value,label,order,default}, ... ] }
            return {'choices': data.get('choices') if isinstance(data, dict) and data.get('choices') else data}
        except Exception as he:
            return JSONResponse(status_code=502, content={'error': 'dynamic helper call failed', 'detail': str(he)})

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/catalog/lookup')
def catalog_lookup(table: str, q: str, limit: int = 20):
    """Lookup records in a reference table (e.g., cmdb_ci) for typeahead fields."""
    try:
        # Prefer searching display_name or name
        params = {"sysparm_query": f"nameLIKE{q}^active=true", "sysparm_fields": "sys_id,name,display_name", "sysparm_limit": limit}
        res = _sn_table_get(table, params)
        out = []
        for r in res:
            label = r.get('display_name') or r.get('name') or r.get('value')
            out.append({'value': r.get('sys_id') or r.get('value'), 'label': label})
        return {'results': out}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.post('/catalog/order')
def catalog_order(body: dict):
    """Place a catalog order for an item_id with supplied variables. Attempts servicecatalog order API then falls back to creating a sc_request."""
    item_id = body.get('item_id')
    variables = body.get('variables', {})
    requested_for = body.get('requested_for')
    comment = body.get('comment')
    if not item_id:
        return JSONResponse(status_code=400, content={'error': 'missing item_id'})
    # First, attempt to use the Service Catalog order API which runs the catalog workflow
    # and creates sc_request + sc_req_item + RITMs automatically.
    try:
        # Use the order_now endpoint which runs the Service Catalog engine and creates sc_item_option rows.
        url = f"https://{SERVICE_NOW_INSTANCE}/api/sn_sc/servicecatalog/items/{item_id}/order_now"
        # Build payload per recommended shape: variables as plain values where reference fields are sys_id strings.
        # Some instances accept array-style variables; we'll try dict-style then array-style.
        qty = int(body.get('quantity', 1) or 1)

        # Variant 1: dict-style variables (preferred). Send requested_for as sys_id string when provided.
        vars_dict = {}
        for k, v in (variables or {}).items():
            # For reference variables pass the sys_id string; for others pass the raw string value.
            vars_dict[k] = v
        payload_a = {'sysparm_quantity': qty, 'variables': vars_dict}
        if requested_for:
            # requested_for as plain sys_id string
            payload_a['requested_for'] = requested_for if isinstance(requested_for, str) else (requested_for.get('value') or requested_for.get('sys_id') if isinstance(requested_for, dict) else requested_for)
        if comment:
            payload_a['comments'] = comment

        # Variant 2: array-style variables [{name, value}, ...]
        vars_array = []
        for k, v in (variables or {}).items():
            vars_array.append({'name': k, 'value': v})
        payload_b = {'sysparm_quantity': qty, 'variables': vars_array}
        if requested_for:
            payload_b['requested_for'] = payload_a.get('requested_for')
        if comment:
            payload_b['comments'] = comment

        attempts = [('dict', payload_a), ('array', payload_b)]
        last_error = None
        for label, payload in attempts:
            try:
                sc_resp = SESSION.post(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json", "Content-Type": "application/json"}, json=payload, timeout=30, verify=VERIFY_TLS)
                sc_resp.raise_for_status()
                sc_data = sc_resp.json().get('result', {})
                # Build a richer confirmation payload including request and RITM(s) when possible
                req_num = sc_data.get('number') or sc_data.get('request_number') or sc_data.get('request') or sc_data.get('request_id') or sc_data.get('sys_id')
                # Attempt to extract the sc_request sys_id from the response
                req_sys = None
                # possible fields: 'request' may be a dict or string
                rf = sc_data.get('request') or sc_data.get('request_sys_id') or sc_data.get('request_id')
                if isinstance(rf, dict):
                    req_sys = rf.get('value') or rf.get('sys_id') or rf.get('id')
                else:
                    req_sys = rf
                # If we still don't have the request sys_id but have the number, try to look it up
                if not req_sys and req_num:
                    try:
                        sr = _sn_table_get('sc_request', { 'sysparm_query': f"number={req_num}", 'sysparm_fields': 'sys_id,number', 'sysparm_limit': 1 })
                        if sr:
                            req_sys = sr[0].get('sys_id') or sr[0].get('value')
                    except Exception:
                        req_sys = None

                result_payload = {'success': True, 'message': 'Order placed successfully'}
                if req_num:
                    result_payload['request_number'] = req_num
                if req_sys:
                    result_payload['request_sys_id'] = req_sys
                    # build instance links when SERVICE_NOW_INSTANCE available
                    try:
                        base = f"https://{SERVICE_NOW_INSTANCE}" if SERVICE_NOW_INSTANCE else None
                        if base:
                            result_payload['request_url'] = f"{base}/nav_to.do?uri=sc_request.do?sys_id={req_sys}"
                    except Exception:
                        pass
                    # fetch any sc_req_item rows (RITMs) created under this request
                    try:
                        ritms = _sn_table_get('sc_req_item', { 'sysparm_query': f"request={req_sys}", 'sysparm_fields': 'sys_id,number,short_description', 'sysparm_limit': 50 })
                        rlist = []
                        if ritms:
                            base = f"https://{SERVICE_NOW_INSTANCE}" if SERVICE_NOW_INSTANCE else None
                            for r in ritms:
                                rsys = r.get('sys_id')
                                rnum = r.get('number') or r.get('request_number')
                                rd = {'sys_id': rsys, 'number': rnum, 'short_description': r.get('short_description')}
                                if base and rsys:
                                    rd['url'] = f"{base}/nav_to.do?uri=sc_req_item.do?sys_id={rsys}"
                                rlist.append(rd)
                        if rlist:
                            result_payload['ritms'] = rlist
                    except Exception:
                        # non-fatal
                        pass

                return result_payload
            except Exception as sce:
                # capture body for debugging and try next variant
                sc_error_body = None
                try:
                    sc_error_body = sc_resp.json()
                except Exception:
                    try:
                        sc_error_body = sc_resp.text
                    except Exception:
                        sc_error_body = str(sce)
                last_error = {'variant': label, 'status_code': getattr(sc_resp, 'status_code', None), 'body': sc_error_body, 'exception': str(sce)}
                continue
        raise Exception(f"ServiceCatalog order_now failed for all payload variants: {last_error}")
    except Exception as e:
        # ServiceCatalog order failed or is unavailable. Fall back to creating sc_request then sc_req_item
        try:
            # Create sc_request first
            req_payload = {
                'short_description': f'Catalog order: {item_id}',
                'description': comment or 'Catalog order via AskGen',
                'u_item_id': item_id,
                'requested_for': requested_for,
                # Preserve variables both as structured 'variables' and as a JSON backup in 'u_variables'
                'variables': variables,
                'u_variables': json.dumps(variables),
                'u_display_values': json.dumps(body.get('display_values') or {})
            }
            r = SESSION.post(f"{SERVICE_NOW_BASE}/sc_request", auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json"}, json=req_payload, timeout=30, verify=VERIFY_TLS)
            r.raise_for_status()
            rdata = r.json().get('result', {})
            # Now create an sc_req_item linked to the created request so RITM-like item exists
            try:
                sc_request_sys_id = rdata.get('sys_id')
                req_item_payload = {
                    # link to parent request
                    'request': sc_request_sys_id,
                    # preserve the catalog item id so owners can map it
                    'u_item_id': item_id,
                    # link the catalog item reference so the RITM shows the Item field
                    'cat_item': item_id,
                    'item': item_id,
                    'short_description': f'Catalog order item: {item_id}',
                    'description': comment or 'Catalog order via AskGen',
                    # include structured variables so the created RITM/sc_req_item contains the filled values
                    'variables': variables,
                    'u_variables': json.dumps(variables),
                    'u_display_values': json.dumps(body.get('display_values') or {}),
                    'requested_for': requested_for
                }
                ri = SESSION.post(f"{SERVICE_NOW_BASE}/sc_req_item", auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json"}, json=req_item_payload, timeout=30, verify=VERIFY_TLS)
                ri.raise_for_status()
                ridata = ri.json().get('result', {})
                # Friendly confirmation when fallback request/item were created
                fr_req_num = rdata.get('number') or rdata.get('request_number') or rdata.get('request_id') or rdata.get('sys_id')
                fr_req_sys = rdata.get('sys_id') or None
                payload_out = {'success': True, 'message': 'Order placed successfully (fallback created)'}
                if fr_req_num:
                    payload_out['request_number'] = fr_req_num
                if fr_req_sys:
                    payload_out['request_sys_id'] = fr_req_sys
                    try:
                        base = f"https://{SERVICE_NOW_INSTANCE}" if SERVICE_NOW_INSTANCE else None
                        if base:
                            payload_out['request_url'] = f"{base}/nav_to.do?uri=sc_request.do?sys_id={fr_req_sys}"
                    except Exception:
                        pass
                # include the created RITM (sc_req_item) info from ridata when available
                try:
                    ritm_sys = ridata.get('sys_id') or ridata.get('result', {}).get('sys_id')
                    ritm_num = ridata.get('number') or ridata.get('request_number')
                    if ritm_sys or ritm_num:
                        rinfo = {'sys_id': ritm_sys, 'number': ritm_num, 'short_description': ridata.get('short_description')}
                        try:
                            base = f"https://{SERVICE_NOW_INSTANCE}" if SERVICE_NOW_INSTANCE else None
                            if base and ritm_sys:
                                rinfo['url'] = f"{base}/nav_to.do?uri=sc_req_item.do?sys_id={ritm_sys}"
                        except Exception:
                            pass
                        payload_out['ritms'] = [rinfo]
                except Exception:
                    pass
                return payload_out
            except Exception as e2:
                # Created sc_request but failed to create sc_req_item; still confirm the request creation
                fr_req_num = rdata.get('number') or rdata.get('request_number') or rdata.get('request_id') or rdata.get('sys_id')
                return {'success': True, 'request_number': fr_req_num, 'message': 'Order submitted but RITM creation failed (see server logs)'}
        except Exception as e3:
            return JSONResponse(status_code=500, content={'error': str(e), 'fallback_error': str(e3), 'sent_payload': payload})

# TLS verification settings
# By default, verify certificates for external services. ServiceNow and other
# integrations should use VERIFY_TLS=True in production. Per your request,
# Azure OpenAI calls will use AZURE_VERIFY_TLS=False so only Azure uses
# verify=False.
VERIFY_TLS = True
AZURE_VERIFY_TLS = False

# Use a dedicated requests Session that ignores environment proxy variables.
# This prevents `requests` from automatically using HTTPS_PROXY/HTTP_PROXY which
# can cause TLS handshake resets or interception by corporate proxies. We still
# respect VERIFY_TLS for certificate verification control.
SESSION = requests.Session()
SESSION.trust_env = False

# --- Azure OpenAI config ---
# Load from environment variables (set these in .env or system environment)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

def query_azure_openai(prompt, system=None, temperature=0.2):
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }
    print("[DEBUG] Sending prompt to Azure OpenAI:", prompt)
    try:
        # Use AZURE_VERIFY_TLS here so only Azure calls can have verification disabled
        response = SESSION.post(url, headers=headers, json=payload, timeout=60, verify=AZURE_VERIFY_TLS)
        response.raise_for_status()
        data = response.json()
        # Defensive: check for expected keys
        if "choices" in data and data["choices"] and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
            txt = data["choices"][0]["message"]["content"]
            print("[DEBUG] Azure OpenAI response:", txt)
            return txt.strip()
        else:
            print(f"[ERROR] Unexpected Azure OpenAI response format: {data}")
            return "ESCALATE"
    except Exception as e:
        print(f"[ERROR] Azure OpenAI query failed: {e}")
        return "ESCALATE"

# --- Azure OpenAI LLM wrapper ---
# (see function above)

def llm_troubleshooting(user_message, session_state):
    prompt_instr = (
        "Agent: Troubleshooting — Provide 1-2 clear, actionable steps the user can try now. "
        "If you cannot resolve, reply EXACTLY with ESCALATE."
    )
    context = f"SESSION_STATE: {session_state}"
    prompt = f"{prompt_instr}\nCONTEXT: {context}\nUser: {user_message}"
    response = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.2)
    if response.strip().upper() == "ESCALATE":
        return response
    if any(word in response.lower() for word in ["cannot help", "not sure", "contact it", "escalate", "admin rights", "hardware", "support team", "unable to resolve"]):
        return "ESCALATE"
    return response

import json
import re

def llm_classify_issue(user_message):
    instr = (
        "Agent: Classification — Return a JSON object with keys: category (hardware, software, access, network, other), "
        "subcategory, urgency (low, medium, high, critical). Respond ONLY with the JSON."
    )
    prompt = f"{instr}\nUser issue: {user_message}"
    result = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
    print("[DEBUG] LLM classify_issue raw output:", result)
    try:
        match = re.search(r"{[\s\S]*?}", result)
        if not match:
            raise ValueError("No JSON object found in LLM output")
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Parsing LLM output: {e}\nRaw output: {result}")
        return {"category": "other", "subcategory": "general", "urgency": "low"}
# --- ServiceNow configuration ---
from requests.auth import HTTPBasicAuth

# ServiceNow configuration (use environment variables in production)
SERVICE_NOW_INSTANCE = os.getenv("SERVICE_NOW_INSTANCE", "dev300144.service-now.com")
SERVICE_NOW_USER = os.getenv("SERVICE_NOW_USER", "admin")
SERVICE_NOW_PASS = os.getenv("SERVICE_NOW_PASS", "Pli5m*=vWX1S")
SERVICE_NOW_BASE = f"https://{SERVICE_NOW_INSTANCE}/api/now/table"

# Compatibility stub for routing (previously used DB to find best support)
def find_best_support(db, classification):
    return None

# ServiceNow-backed ticket creation
def create_ticket(db, user_id, description, classification, assignee, payload_override=None, session_state=None, table=None):
    """
    Create ticket in ServiceNow. If payload_override is provided, use it directly as the POST body (allows LLM-prepared payloads).
    session_state can be provided to select request vs incident explicitly.
    """
    # Build payload: prefer payload_override body but always respect explicit `table` arg when provided
    if payload_override is not None and isinstance(payload_override, dict):
        payload = payload_override
    else:
        # Prefer classification['priority'] but accept legacy 'urgency'
        derived_priority = classification.get("priority") or classification.get("urgency")
        payload = {
            "short_description": description if len(description) <= 160 else description[:157] + "...",
            "description": description,
            # ServiceNow standard priority field; keep u_urgency for backward compatibility if needed
            "priority": derived_priority,
            "u_urgency": derived_priority,
            "u_category": classification.get("category"),
            "u_subcategory": classification.get("subcategory")
        }
    # Determine table from explicit parameter, then session_state type, then classification heuristic
    if table is None:
        if session_state and session_state.get("type"):
            is_request = session_state.get("type") == "request"
        else:
            category = (classification.get("category") or "").lower()
            is_request = category in ["access", "installation", "request"]
        table = "sc_request" if is_request else "incident"
    url = f"{SERVICE_NOW_BASE}/{table}?sysparm_exclude_reference_link=True"
    try:
        # If a language is specified in session_state or payload, translate the description fields
        try:
            target_lang = None
            if session_state and isinstance(session_state, dict):
                target_lang = session_state.get('language')
            if not target_lang and isinstance(payload, dict):
                target_lang = payload.get('language') or payload.get('lang')
            if target_lang and isinstance(target_lang, str) and not target_lang.lower().startswith('en'):
                # translate short_description and description before sending
                try:
                    payload['short_description'] = translate_text(payload.get('short_description',''), target_lang)
                    payload['description'] = translate_text(payload.get('description',''), target_lang)
                except Exception as e:
                    print(f"[WARN] Ticket translation failed: {e}")
        except Exception:
            pass
        # Minimal debug: mark that ticket creation was attempted and record non-sensitive metadata
        try:
            if session_state is not None and isinstance(session_state, dict):
                session_state['ticket_creation_attempted'] = True
                session_state['ticket_creation_table'] = table
                # avoid storing full payloads; only store top-level keys for observability
                if isinstance(payload, dict):
                    session_state['ticket_creation_payload_keys'] = list(payload.keys())[:20]
        except Exception:
            pass
        resp = SESSION.post(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                            headers={"Accept": "application/json"}, json=payload, timeout=30, verify=VERIFY_TLS)
        resp.raise_for_status()
        data = resp.json().get("result", {})
        class SNOWTicket:
            pass
        t = SNOWTicket()
        t.id = data.get("sys_id")
        t.number = data.get("number") or data.get("request_number") or t.id
        t.status = data.get("state") or data.get("status") or "open"
        # Normalize priority: prefer explicit ServiceNow 'priority' field, fall back to classification
        t.priority = data.get("priority") or classification.get("priority") or classification.get("urgency", "medium")
        assigned = data.get("assigned_to")
        if isinstance(assigned, dict):
            t.assigned_to = assigned.get("value")
            t.assigned_to_name = assigned.get("display_value")
        else:
            t.assigned_to = assigned
            t.assigned_to_name = None
        t.category = data.get("u_category") or classification.get("category")
        t.subcategory = data.get("u_subcategory") or classification.get("subcategory")
        # preserve any legacy urgency field but keep priority as primary
        t.urgency = data.get("u_urgency") or classification.get("urgency")
        t.description = data.get("description") or description
        t.created_at = data.get("sys_created_on")
        # persist created ticket in session_state if provided
        if session_state is not None:
            st = session_state.get("created_tickets", [])
            st.append({"number": t.number, "id": t.id, "created_at": t.created_at, "table": table})
            session_state["created_tickets"] = st
        return t
    except Exception as e:
        # Record minimal error info for debugging without printing sensitive payloads.
        try:
            # If resp exists and has content, try to extract status and body
            err_status = None
            err_body = None
            if 'resp' in locals() and hasattr(resp, 'status_code'):
                err_status = getattr(resp, 'status_code', None)
                try:
                    err_body = resp.json()
                except Exception:
                    try:
                        err_body = resp.text
                    except Exception:
                        err_body = None
            else:
                # If no resp object, capture exception text
                err_body = str(e)
        except Exception:
            err_status = None
            err_body = str(e)
        print(f"[ERROR] ServiceNow ticket creation failed: {e}")
        class FallbackTicket:
            pass
        t = FallbackTicket()
        t.id = None
        t.number = None
        t.status = "open"
        t.priority = classification.get("urgency", "medium")
        t.assigned_to = None
        t.assigned_to_name = None
        t.category = classification.get("category")
        t.subcategory = classification.get("subcategory")
        t.urgency = classification.get("urgency")
        t.description = payload.get("description") if isinstance(payload, dict) else description
        t.created_at = None
        # Persist minimal ticket creation error into session_state for observability (non-verbose)
        try:
            if session_state is not None and isinstance(session_state, dict):
                session_state["ticket_creation_error"] = {"status": err_status, "body": err_body}
        except Exception:
            pass
        return t


# --- New LLM-backed ticket payload preparer ---
def llm_prepare_ticket(classification, user_problem, session_state):
    """
    Use the MASTER_PROMPT as system message and ask the LLM to produce a strict JSON payload
    suitable for ServiceNow ticket creation. Returns a dict payload or None on failure.
    """
    instr = (
        "Agent: Ticketing — Given the classification and conversation context, return a JSON object "
        "with keys: short_description, description, urgency, impact (optional), category, subcategory, requested_for (optional). "
        "Short_description should be <=160 chars. Respond ONLY with the JSON object."
    )
    context = {"classification": classification, "session_state": session_state, "user_problem": user_problem}
    prompt = f"{instr}\nCONTEXT: {json.dumps(context)}\nUser issue: {user_problem}"
    result = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
    try:
        match = re.search(r"{[\s\S]*?}", result)
        if not match:
            raise ValueError("No JSON object found in LLM output")
        json_str = match.group(0)
        payload = json.loads(json_str)
        # Ensure short_description present
        if "short_description" not in payload:
            payload["short_description"] = payload.get("description", user_problem)[:160]
        return payload
    except Exception as e:
        print(f"[ERROR] Preparing ticket payload via LLM failed: {e}\nRaw output: {result}")
        return None


# --- New KB search helper that uses ServiceNow KB and summarizes via LLM ---
def search_kb_and_summarize(query, top_n=5):
    """
    Search ServiceNow KB (kb_knowledge) for the query, then ask LLM (with MASTER_PROMPT) to summarize top results.
    Returns a list of summarized articles (title, snippet, url).
    """
    try:
        url = f"{SERVICE_NOW_BASE}/kb_knowledge?sysparm_query=active=true^NQ^titleLIKE{query}&sysparm_limit={top_n}&sysparm_exclude_reference_link=True"
        resp = SESSION.get(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
        resp.raise_for_status()
        items = resp.json().get("result", [])
        if not items:
            return []
        # Prepare a compact representation for the LLM
        compact = []
        for it in items:
            compact.append({
                "title": it.get("short_description") or it.get("title"),
                "snippet": (it.get("text") or it.get("summary") or "")[:800],
                "url": f"https://{SERVICE_NOW_INSTANCE}/kb_view.do?sys_kb_id={it.get('sys_id')}"
            })
        # Ask LLM to summarize
        instr = (
            "Agent: KB-Summarizer — Given a list of knowledge articles (title, snippet, url), return a JSON array of objects "
            "with keys: title, summary (one-sentence), url. Respond ONLY with the JSON array."
        )
        prompt = f"{instr}\nARTICLES: {json.dumps(compact)}"
        summary_out = query_azure_openai(prompt, system=MASTER_PROMPT, temperature=0.0)
        try:
            match = re.search(r"\[[\s\S]*?\]", summary_out)
            if not match:
                raise ValueError("No JSON array found")
            arr = json.loads(match.group(0))
            return arr
        except Exception as e:
            print(f"[ERROR] Parsing KB summarizer output: {e}\nRaw: {summary_out}")
            # Fallback: return compact items limited fields
            return compact
    except Exception as e:
        print(f"[ERROR] ServiceNow KB search failed: {e}")
        return []


# --- RAG-Enhanced KB Search (uses vector store when available, falls back to ServiceNow) ---
def search_kb_with_rag(query: str, top_n: int = 5):
    """
    Search knowledge base using RAG semantic search when available,
    otherwise fall back to ServiceNow KB search.
    Returns a list of articles with title, snippet, url, and relevance score.
    """
    if RAG_ENABLED:
        try:
            rag_engine = get_rag_engine()
            results = rag_engine.find_relevant_kb(query, top_k=top_n)
            
            articles = []
            for r in results:
                articles.append({
                    "title": r.document.title,
                    "snippet": r.document.content[:500] + "..." if len(r.document.content) > 500 else r.document.content,
                    "url": r.document.url,
                    "score": round(r.score, 3),
                    "confidence": r.confidence,
                    "source": r.document.source,
                    "doc_type": r.document.doc_type.value
                })
            
            if articles:
                print(f"[RAG] Found {len(articles)} KB articles via semantic search")
                return articles
        except Exception as e:
            print(f"[RAG] Semantic KB search failed, falling back to ServiceNow: {e}")
    
    # Fallback to ServiceNow KB search
    return search_kb_and_summarize(query, top_n)


def learn_from_ticket_resolution(ticket_number: str, problem: str, resolution: str, category: str = ""):
    """
    Learn from a ticket resolution by adding it to the RAG knowledge base.
    Called automatically when a ticket is marked as resolved.
    """
    if not RAG_ENABLED:
        return False
    
    try:
        learner = get_knowledge_learner()
        doc_id = learner.learn_from_resolution(
            ticket_number=ticket_number,
            problem_description=problem,
            resolution_steps=resolution,
            category=category,
            was_helpful=True
        )
        
        if doc_id:
            print(f"[RAG] Learned from ticket {ticket_number} resolution (doc_id: {doc_id})")
            return True
        else:
            print(f"[RAG] Resolution for {ticket_number} did not meet quality threshold")
            return False
    except Exception as e:
        print(f"[RAG] Failed to learn from ticket {ticket_number}: {e}")
        return False


# --- FastAPI app ---
# `app` is instantiated earlier in the file so route decorators may be defined above.

# Live WebSocket-based backend log streaming removed.
# Previously the app exposed a /ws/logs websocket and in-process enqueue_log/_broadcast_log helpers.
# Those were removed because in-memory websocket connection sets do not work reliably when
# the server runs with multiple processes or a reload-enabled reloader.

@app.post("/chat/")
async def chat(request: Request):
    print("[DEBUG] Received chat request")
    data = await request.json()
    print("[DEBUG] Request data:", data)
    # Extract language preference (from JSON body or query param)
    lang = get_request_language(request, data)
    user_id = int(data.get("user_id", 1))
    message = data.get("message")
    session_state = data.get("session_state", {})
    # persist language preference into session_state so downstream orchestration can use it
    try:
        session_state['language'] = lang
    except Exception:
        pass
    # ensure chat_history present and persist the user turn
    session_state.setdefault("chat_history", [])
    session_state["chat_history"].append(("You", message))
    # --- LangGraph orchestration ---
    input_state = ITSMState(
        user_id=user_id,
        message=message,
        session_state=session_state
    )
    # Short-circuit simple confirmation flows (yes/no) to avoid calling LLM classifiers
    # when the session is awaiting resolution confirmation or troubleshoot consent.
    # Additionally, when awaiting_escalation_confirmation we deterministically route the
    # turn to the orchestrator node so simple 'yes'/'no' replies are handled without
    # invoking the LLM-based classifier or greeting detector which can misclassify short replies.
    if session_state.get("awaiting_resolution_confirmation") or session_state.get("awaiting_troubleshoot_consent"):
        try:
            print("[DEBUG] Short-circuiting to troubleshooting node for confirmation handling (resolution/consent only)")
            node_result = node_troubleshooting(input_state)
            # convert to dict-like result_state for downstream handling
            if isinstance(node_result, ITSMState):
                result_state = node_result.dict()
            else:
                try:
                    result_state = (node_result.dict() if hasattr(node_result, 'dict') else dict(node_result))
                except Exception:
                    result_state = input_state.dict()
        except Exception as e:
            print(f"[ERROR] troubleshooting short-circuit failed, falling back to workflow invoke: {e}")
            result_state = itsm_workflow.invoke(input_state)
    elif session_state.get("awaiting_escalation_confirmation"):
        try:
            print("[DEBUG] Short-circuiting to orchestrator node to deterministically handle escalation confirmation")
            # Run orchestrator node first to deterministically interpret the simple yes/no.
            node_result = node_orchestrator(input_state)
            # Now invoke the full workflow so downstream routing (e.g., ticketing) runs if orchestrator set flags.
            try:
                wf_result = itsm_workflow.invoke(input_state)
                if isinstance(wf_result, ITSMState):
                    result_state = wf_result.dict()
                else:
                    try:
                        result_state = (wf_result.dict() if hasattr(wf_result, 'dict') else dict(wf_result))
                    except Exception:
                        result_state = input_state.dict()
            except Exception as e:
                print(f"[ERROR] workflow invoke after orchestrator short-circuit failed: {e}")
                # Fallback: expose orchestrator's immediate state so caller can retry
                if isinstance(node_result, ITSMState):
                    result_state = node_result.dict()
                else:
                    try:
                        result_state = (node_result.dict() if hasattr(node_result, 'dict') else dict(node_result))
                    except Exception:
                        result_state = input_state.dict()
        except Exception as e:
            print(f"[ERROR] troubleshooting short-circuit failed, falling back to workflow invoke: {e}")
            result_state = itsm_workflow.invoke(input_state)
    else:
        result_state = itsm_workflow.invoke(input_state)
    # Normalize response fields to avoid sending None to frontend
    resp_text = result_state.get("response") or ""
    solved = result_state.get("solved") if result_state.get("solved") is not None else False
    ticket = result_state.get("ticket_info") or None
    session_out = result_state.get("session_state") or {}
    # Keep agent_stage for server-side logs only; do not expose to frontend to avoid UI showing internal agent status
    agent_stage = result_state.get("agent_stage") or None
    # live backend log streaming removed; no-op here
    print(f"[DEBUG] Agent stage: {agent_stage}; Response: {resp_text}")
    out = {
        "response": resp_text,
        "solved": solved,
        "ticket": ticket,
        "session_state": session_out,
        # Expose agent_stage so frontend can reflect which agent handled the request
        "agent_stage": agent_stage,
        "logs": [],
        # RAG enhancement metadata
        "rag_enabled": RAG_ENABLED
    }
    
    # Include RAG sources and confidence if available from session_state
    if session_out.get('rag_sources'):
        out['rag_sources'] = session_out.get('rag_sources')
        out['rag_confidence'] = session_out.get('rag_confidence', 'unknown')

    # If orchestrator determined this is a request that needs clarification (clarify node) and
    # there are no pending_slots yet, try to find a matching ServiceNow catalog item and suggest it.
    try:
        st = session_out or {}
        # Heuristic: type=request and not yet clarified and not already awaiting clarifier slots
        t = st.get('type') or (result_state.get('classification') or {}).get('type')
        pending = st.get('pending_slots') or []
        if t == 'request' and not st.get('clarified') and not pending:
                # Fetch a page of active catalog items and ask the LLM to pick the best match
                user_msg = (result_state.get('message') or input_state.message) or ''
                params = {"sysparm_query": "active=true", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 200}
                catalog_items = []
                try:
                    catalog_items = _sn_table_get('sc_cat_item', params)
                except Exception as e:
                    print(f"[WARN] Failed to fetch catalog items: {e}")
                picked_idx = -1
                tried_kw = []
                final_query = None
                if catalog_items:
                    # let the LLM pick the best index
                    picked_idx = _llm_pick_catalog_index(catalog_items, user_msg)
                    # for debug, also extract keywords we might have used earlier
                    tried_kw = _llm_extract_catalog_keywords(user_msg, max_keywords=3)
                    if picked_idx is not None and picked_idx >= 0 and picked_idx < len(catalog_items):
                        suggestion = catalog_items[picked_idx]
                        final_query = suggestion.get('name') or suggestion.get('short_description')
                        out['catalog_suggestion'] = {'item': {'sys_id': suggestion.get('sys_id'), 'name': suggestion.get('name'), 'short_description': suggestion.get('short_description')}}
                        out['session_state'] = {**st, 'catalog_suggestion': out['catalog_suggestion']['item'], 'awaiting_catalog_confirmation': True}
                out['catalog_debug'] = {'keywords_tried': tried_kw, 'final_query': final_query, 'picked_index': picked_idx, 'catalog_count': len(catalog_items)}
    except Exception as e:
        print(f"[WARN] Catalog suggestion check failed: {e}")
    # If the orchestrator already fetched catalog variables (pending_slots) after user confirmed a suggestion,
    # include the full catalog item + variables in the chat response so the frontend can render a form.
    try:
        st = session_out or {}
        # Only include when a suggestion exists, user has confirmed (no awaiting flag) and there are pending slots or slot_questions
        if st.get('catalog_suggestion') and not st.get('awaiting_catalog_confirmation') and (st.get('pending_slots') or st.get('slot_questions')):
            item = st.get('catalog_suggestion')
            item_id = None
            if isinstance(item, dict):
                item_id = item.get('sys_id') or item.get('id')
            if item_id:
                try:
                    # Reuse the canonical catalog variable normalization so chat responses include the same
                    # choices and variable-set merging as the /catalog/{id}/variables endpoint.
                    norm = catalog_item(item_id)
                    if isinstance(norm, dict):
                        # catalog_item returns {'item': {...}, 'variables': [...]}
                        variables = norm.get('variables') or []
                    else:
                        variables = []
                    # Fetch item metadata if possible for display; fall back to minimal item record
                    try:
                        params = {"sysparm_query": f"sys_id={item_id}", "sysparm_exclude_reference_link": True}
                        items = _sn_table_get('sc_cat_item', params)
                        it = items[0] if items else {'sys_id': item_id}
                    except Exception:
                        it = {'sys_id': item_id}

                    out['catalog_item'] = {'item': {'sys_id': it.get('sys_id'), 'name': it.get('name'), 'short_description': it.get('short_description')}, 'variables': variables}
                    try:
                        out['catalog_item_form'] = {
                            'item': {'sys_id': it.get('sys_id'), 'name': it.get('name'), 'short_description': it.get('short_description')},
                            'variables': variables,
                            'pending_slots': st.get('pending_slots') or [],
                            'slot_questions': st.get('slot_questions') or {},
                            'pending_question': st.get('pending_question')
                        }
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[WARN] building catalog_item for chat response failed: {e}")
    except Exception:
        pass
    return out


@app.get("/api/tickets")
def get_tickets(type: str = "all", limit: int = 50):
    """Fetch tickets from ServiceNow - both incidents and requests."""
    from requests.auth import HTTPBasicAuth
    
    tickets = []
    display_query = "&sysparm_display_value=true"
    fields = "sys_id,number,short_description,description,state,priority,category,u_category,sys_created_on,assigned_to,caller_id"
    
    try:
        # Fetch incidents if type is 'all' or 'incidents'
        if type in ['all', 'incidents']:
            try:
                inc_url = f"{SERVICE_NOW_BASE}/incident?sysparm_exclude_reference_link=True&sysparm_fields={fields}&sysparm_limit={limit}&sysparm_orderby=sys_created_on&sysparm_orderbyDesc=sys_created_on{display_query}"
                resp = SESSION.get(inc_url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                                   headers={"Accept": "application/json"}, timeout=15, verify=VERIFY_TLS)
                resp.raise_for_status()
                incidents = resp.json().get("result", [])
                for inc in incidents:
                    # Normalize state value
                    state_raw = inc.get("state")
                    if isinstance(state_raw, dict):
                        status = state_raw.get("display_value") or state_raw.get("value") or "New"
                    else:
                        status = str(state_raw) if state_raw else "New"
                    
                    tickets.append({
                        "sys_id": inc.get("sys_id"),
                        "number": inc.get("number"),
                        "short_description": inc.get("short_description"),
                        "description": inc.get("description"),
                        "status": status,
                        "state": status,
                        "priority": inc.get("priority"),
                        "category": inc.get("category") or inc.get("u_category"),
                        "sys_created_on": inc.get("sys_created_on"),
                        "type": "incident"
                    })
            except Exception as e:
                print(f"[WARN] Failed to fetch incidents: {e}")
        
        # Fetch requests if type is 'all' or 'requests'
        if type in ['all', 'requests']:
            # Try sc_request table (service requests)
            try:
                req_url = f"{SERVICE_NOW_BASE}/sc_request?sysparm_exclude_reference_link=True&sysparm_fields={fields},request_state&sysparm_limit={limit}&sysparm_orderby=sys_created_on&sysparm_orderbyDesc=sys_created_on{display_query}"
                resp = SESSION.get(req_url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                                   headers={"Accept": "application/json"}, timeout=15, verify=VERIFY_TLS)
                resp.raise_for_status()
                requests_data = resp.json().get("result", [])
                for req in requests_data:
                    # Normalize state value
                    state_raw = req.get("request_state") or req.get("state")
                    if isinstance(state_raw, dict):
                        status = state_raw.get("display_value") or state_raw.get("value") or "New"
                    else:
                        status = str(state_raw) if state_raw else "New"
                    
                    tickets.append({
                        "sys_id": req.get("sys_id"),
                        "number": req.get("number"),
                        "short_description": req.get("short_description"),
                        "description": req.get("description"),
                        "status": status,
                        "state": status,
                        "priority": req.get("priority"),
                        "category": req.get("category") or req.get("u_category"),
                        "sys_created_on": req.get("sys_created_on"),
                        "type": "request"
                    })
            except Exception as e:
                print(f"[WARN] Failed to fetch sc_request: {e}")
            
            # Also try sc_req_item (requested items / RITM)
            try:
                ritm_url = f"{SERVICE_NOW_BASE}/sc_req_item?sysparm_exclude_reference_link=True&sysparm_fields={fields}&sysparm_limit={limit}&sysparm_orderby=sys_created_on&sysparm_orderbyDesc=sys_created_on{display_query}"
                resp = SESSION.get(ritm_url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                                   headers={"Accept": "application/json"}, timeout=15, verify=VERIFY_TLS)
                resp.raise_for_status()
                ritms = resp.json().get("result", [])
                for ritm in ritms:
                    # Normalize state value
                    state_raw = ritm.get("state")
                    if isinstance(state_raw, dict):
                        status = state_raw.get("display_value") or state_raw.get("value") or "New"
                    else:
                        status = str(state_raw) if state_raw else "New"
                    
                    tickets.append({
                        "sys_id": ritm.get("sys_id"),
                        "number": ritm.get("number"),
                        "short_description": ritm.get("short_description"),
                        "description": ritm.get("description"),
                        "status": status,
                        "state": status,
                        "priority": ritm.get("priority"),
                        "category": ritm.get("category") or ritm.get("u_category"),
                        "sys_created_on": ritm.get("sys_created_on"),
                        "type": "request_item"
                    })
            except Exception as e:
                print(f"[WARN] Failed to fetch sc_req_item: {e}")
        
        # Sort all tickets by creation date (newest first)
        tickets.sort(key=lambda x: x.get("sys_created_on") or "", reverse=True)
        
        # Limit total results
        tickets = tickets[:limit]
        
        return {
            "status": "ok",
            "tickets": tickets,
            "total": len(tickets),
            "type_filter": type
        }
    except Exception as e:
        print(f"[ERROR] Failed to fetch tickets: {e}")
        return {
            "status": "error",
            "error": str(e),
            "tickets": [],
            "total": 0
        }


@app.get("/ticket/{ticket_number}")
def get_ticket(ticket_number: str):
    try:
        # Ask ServiceNow to return display values where possible so we can show human-friendly state labels
        display_query = "&sysparm_display_value=true"

        # Try incident table first
        url = f"{SERVICE_NOW_BASE}/incident?sysparm_exclude_reference_link=True&sysparm_query=number={ticket_number}&sysparm_limit=1{display_query}"
        resp = SESSION.get(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                           headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
        resp.raise_for_status()
        data = resp.json().get("result", [])
        if data:
            r = data[0]
            # Determine human-friendly state
            state_raw = r.get("state")
            state_code = None
            human_state = None
            # If ServiceNow returned an object for the choice field, try to extract value/display_value
            if isinstance(state_raw, dict):
                state_code = state_raw.get("value")
                human_state = state_raw.get("display_value") or state_raw.get("value")
            else:
                state_str = str(state_raw) if state_raw is not None else ""
                # If numeric code, map common codes to labels, else assume it's already human-friendly
                mapping = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed"}
                if state_str.isdigit():
                    state_code = state_str
                    human_state = mapping.get(state_str, state_str)
                else:
                    human_state = state_str

            result = {
                "id": r.get("sys_id"),
                "number": r.get("number"),
                "state": human_state,
                "state_code": state_code,
                "assigned_to": r.get("assigned_to", {}).get("display_value") if isinstance(r.get("assigned_to"), dict) else r.get("assigned_to"),
                "priority": r.get("priority"),
                "category": r.get("u_category") or r.get("category"),
                "subcategory": r.get("u_subcategory") or r.get("subcategory"),
                "urgency": r.get("u_urgency") or r.get("urgency"),
                "description": r.get("description"),
                "created_at": r.get("sys_created_on")
            }
            return result

        # If not found in incident, try sc_request
        url2 = f"{SERVICE_NOW_BASE}/sc_request?sysparm_exclude_reference_link=True&sysparm_query=number={ticket_number}&sysparm_limit=1{display_query}"
        resp2 = SESSION.get(url2, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS),
                            headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
        resp2.raise_for_status()
        data2 = resp2.json().get("result", [])
        if data2:
            r = data2[0]
            state_raw = r.get("state")
            state_code = None
            human_state = None
            if isinstance(state_raw, dict):
                state_code = state_raw.get("value")
                human_state = state_raw.get("display_value") or state_raw.get("value")
            else:
                state_str = str(state_raw) if state_raw is not None else ""
                mapping = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed"}
                if state_str.isdigit():
                    state_code = state_str
                    human_state = mapping.get(state_str, state_str)
                else:
                    human_state = state_str

            result = {
                "id": r.get("sys_id"),
                "number": r.get("number"),
                "state": human_state,
                "state_code": state_code,
                "assigned_to": r.get("assigned_to", {}).get("display_value") if isinstance(r.get("assigned_to"), dict) else r.get("assigned_to"),
                "priority": r.get("priority"),
                "category": r.get("u_category") or r.get("category"),
                "subcategory": r.get("u_subcategory") or r.get("subcategory"),
                "urgency": r.get("u_urgency") or r.get("urgency"),
                "description": r.get("description"),
                "created_at": r.get("sys_created_on")
            }
            return result

        return {"error": "Ticket not found"}
    except Exception as e:
        print(f"[ERROR] ServiceNow ticket retrieval failed: {e}")
        return {"error": "ServiceNow error"}

def detect_status_check(user_message):
    """Simple heuristic to detect ticket status check requests and extract ticket id/number."""
    if not user_message:
        return None
    patterns = [r"(?:INC|REQ|RITM)?\s*#?([A-Za-z]{3}\d{6,}|\d{4,})", r"check ticket\s+([A-Za-z0-9_-]+)", r"status of ticket\s+([A-Za-z0-9_-]+)"]
    msg = user_message.strip().lower()
    if any(k in msg for k in ["check ticket", "ticket status", "status of ticket", "check inc", "check request"]) or re.search(r"\b(inc|req|ritm)[0-9]{3,}\b", user_message, re.IGNORECASE):
        for p in patterns:
            m = re.search(p, user_message, re.IGNORECASE)
            if m:
                return m.group(1)
        m2 = re.search(r"(INC\w+|REQ\w+|RITM\w+|\d{4,})", user_message, re.IGNORECASE)
        if m2:
            return m2.group(1)
    return None


@app.get('/debug/mtom/{item_id}')
def debug_mtom(item_id: str):
    """Debug endpoint: return sc_item_option_mtom rows for the given catalog item sys_id."""
    try:
        params = {"sysparm_query": f"cat_item={item_id}", "sysparm_limit": 500}
        rows = _sn_table_get('sc_item_option_mtom', params)
        # Return raw rows for inspection
        return {'count': len(rows) if isinstance(rows, list) else 0, 'rows': rows}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/debug/sc_item_option')
def debug_sc_item_option(ids: str):
    """Debug: fetch sc_item_option rows for comma-separated sys_id list.
    Example: /debug/sc_item_option?ids=fb55...,cf36...,7b55...
    """
    try:
        if not ids:
            return JSONResponse(status_code=400, content={'error': 'missing ids parameter'})
        q = 'sys_idIN' + ids
        params = {"sysparm_query": q, "sysparm_fields": "sys_id,item_option_new,variable,question,element,variable_name,question_text", "sysparm_limit": 500}
        rows = _sn_table_get('sc_item_option', params)
        return {'count': len(rows) if isinstance(rows, list) else 0, 'rows': rows}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/debug/variable_set')
def debug_variable_set(sys_id: str | None = None, name: str | None = None):
    """Debug helper: lookup a Variable Set by sys_id or name and return its variables.
    This will try the ServiceNow table `item_option_new_set` first (common name for variable sets),
    and fall back to `variable_set` for compatibility.
    Examples:
      /debug/variable_set?sys_id=<sys_id>
      /debug/variable_set?name=it_to_it
    """
    try:
        if not sys_id and not name:
            return JSONResponse(status_code=400, content={'error': 'provide sys_id or name'})
        # Find the variable set: prefer the item_option_new_set table, then fallback to variable_set
        varset = None
        if sys_id:
            params1 = {"sysparm_query": f"sys_id={sys_id}", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 1}
            try:
                vs1 = _sn_table_get('item_option_new_set', params1)
                varset = vs1[0] if vs1 else None
            except Exception:
                varset = None
            if not varset:
                params2 = {"sysparm_query": f"sys_id={sys_id}", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 1}
                vs2 = _sn_table_get('variable_set', params2)
                varset = vs2[0] if vs2 else None
        else:
            # search by name (LIKE) in item_option_new_set first
            params1 = {"sysparm_query": f"nameLIKE{name}", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 10}
            try:
                vs1 = _sn_table_get('item_option_new_set', params1)
                varset = vs1[0] if vs1 else None
            except Exception:
                varset = None
            if not varset:
                params2 = {"sysparm_query": f"nameLIKE{name}", "sysparm_fields": "sys_id,name,short_description", "sysparm_limit": 10}
                vs2 = _sn_table_get('variable_set', params2)
                varset = vs2[0] if vs2 else None
        if not varset:
            return JSONResponse(status_code=404, content={'error': 'variable_set not found'})
        vs_sys_id = varset.get('sys_id')
        # fetch variables in that set
        vparams = {"sysparm_query": f"variable_set={vs_sys_id}", "sysparm_fields": "sys_id,variable_name,element,name,question,question_text,type,mandatory,default_value,reference,choice", "sysparm_limit": 500}
        vars = _sn_table_get('item_option_new', vparams)
        return {'variable_set': {'sys_id': varset.get('sys_id'), 'name': varset.get('name'), 'short_description': varset.get('short_description')}, 'variables': vars}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/debug/choices')
def debug_choices(var_id: str | None = None, variable_name: str | None = None):
    """Debug helper: inspect choice rows for an item_option_new variable.
    Try common tables: item_option_new_choice and sys_choice with multiple query patterns.
    Examples: /debug/choices?var_id=<sys_id>
    """
    try:
        if not var_id and not variable_name:
            return JSONResponse(status_code=400, content={'error': 'provide var_id or variable_name'})
        results = {}
        # try item_option_new_choice table linked by item_option_new, question, or variable
        qlist = []
        if var_id:
            qlist.append(f"item_option_new={var_id}")
            qlist.append(f"question={var_id}")
            qlist.append(f"variable={var_id}")
        if variable_name:
            qlist.append(f"item_option_newLIKE{variable_name}")
            qlist.append(f"questionLIKE{variable_name}")
        found = []
        for q in qlist:
            try:
                rows = _sn_table_get('item_option_new_choice', {"sysparm_query": q, "sysparm_limit": 500})
                if rows:
                    results[f'item_option_new_choice::{q}'] = rows
                    found.extend(rows)
            except Exception:
                continue

        # try sys_choice: look for choices defined on an assumed table 'item_option_new' or for element names
        try:
            qsys = None
            if variable_name:
                qsys = f"elementLIKE{variable_name}^ORlabelLIKE{variable_name}"
            elif var_id:
                qsys = f"name=item_option_new^elementLIKE{var_id}"
            if qsys:
                sc_rows = _sn_table_get('sys_choice', {"sysparm_query": qsys, "sysparm_limit": 500})
                if sc_rows:
                    results[f'sys_choice::{qsys}'] = sc_rows
        except Exception:
            pass

        return {'queried_patterns': qlist, 'results': results}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/debug/item_option_new')
def debug_item_option_new(sys_id: str | None = None):
    """Return raw item_option_new row for inspection."""
    try:
        if not sys_id:
            return JSONResponse(status_code=400, content={'error': 'provide sys_id'})
        params = {"sysparm_query": f"sys_id={sys_id}", "sysparm_limit": 1}
        rows = _sn_table_get('item_option_new', params)
        if not rows:
            return JSONResponse(status_code=404, content={'error': 'item_option_new not found'})
        return {'row': rows[0]}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# /debug/azure_tls endpoint removed — TLS verification is controlled by VERIFY_TLS (set to False for dev).

@app.get('/catalog/resolve_variable_choices')
def resolve_variable_choices(instance_url: str | None = None, variable_sys_id: str | None = None, variable_name: str | None = None, include_inactive: bool = False):
    """Resolve a variable and return its question_choice rows as JSON per the requested contract.

    Query params:
      - instance_url (optional, defaults to configured SERVICE_NOW_INSTANCE)
      - variable_sys_id OR variable_name
      - include_inactive (bool, default false)

    Returns exactly the JSON object described in the spec.
    """
    # Helper: build base table API URL
    try:
        inst = instance_url.rstrip('/') if instance_url else f"https://{SERVICE_NOW_INSTANCE}"
        table_base = inst + '/api/now/table'
    except Exception:
        return {'resolved_table': None, 'question_sys_id': None, 'choices': []}

    # Ensure at least one identifier provided
    if not variable_sys_id and not variable_name:
        return {'resolved_table': None, 'question_sys_id': None, 'choices': []}

    # HTTP helper with retry on 429/5xx
    def get_with_retry(url, params=None, max_retries=3):
        backoff = 0.5
        last_resp = None
        for attempt in range(max_retries):
            try:
                resp = SESSION.get(url, auth=HTTPBasicAuth(SERVICE_NOW_USER, SERVICE_NOW_PASS), params=params or {}, headers={"Accept": "application/json"}, timeout=20, verify=VERIFY_TLS)
                # If 429 or 5xx, prepare to retry
                if resp.status_code == 429 or (500 <= resp.status_code < 600):
                    last_resp = resp
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json().get('result', [])
            except requests.exceptions.RequestException as e:
                # For network errors or HTTP errors, decide if retryable
                if hasattr(e, 'response') and e.response is not None and (e.response.status_code == 429 or (500 <= e.response.status_code < 600)):
                    last_resp = e.response
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # non-retryable
                return None
        # exhausted retries
        try:
            return last_resp.json().get('result', []) if last_resp is not None else None
        except Exception:
            return None

    # 1) Resolve variable to a record and its sys_id + type
    resolved_table = None
    question_sys_id = None
    type_display = None

    # Try by sys_id first
    if variable_sys_id:
        # Try item_option_new by sys_id
        rows = get_with_retry(f"{table_base}/item_option_new", params={"sysparm_query": f"sys_id={variable_sys_id}", "sysparm_limit": 1, "sysparm_fields": "sys_id,type"})
        if rows and len(rows) > 0:
            resolved_table = 'item_option_new'
            question_sys_id = rows[0].get('sys_id')
            type_display = rows[0].get('type')
        else:
            # Try question table by sys_id
            rows = get_with_retry(f"{table_base}/question", params={"sysparm_query": f"sys_id={variable_sys_id}", "sysparm_limit": 1, "sysparm_fields": "sys_id,type"})
            if rows and len(rows) > 0:
                resolved_table = 'question'
                question_sys_id = rows[0].get('sys_id')
                type_display = rows[0].get('type')

    else:
        # Resolve by name
        qname = variable_name
        # Try item_option_new by name/element/variable_name
        rows = get_with_retry(f"{table_base}/item_option_new", params={"sysparm_query": f"name={qname}^ORelement={qname}^ORvariable_name={qname}", "sysparm_limit": 1, "sysparm_fields": "sys_id,type,name,element,variable_name"})
        if rows and len(rows) > 0:
            resolved_table = 'item_option_new'
            question_sys_id = rows[0].get('sys_id')
            type_display = rows[0].get('type')
        else:
            # Try question table by name
            rows = get_with_retry(f"{table_base}/question", params={"sysparm_query": f"name={qname}", "sysparm_limit": 1, "sysparm_fields": "sys_id,type,name"})
            if rows and len(rows) > 0:
                resolved_table = 'question'
                question_sys_id = rows[0].get('sys_id')
                type_display = rows[0].get('type')

    if not question_sys_id:
        return {'resolved_table': None, 'question_sys_id': None, 'choices': []}

    # 2) Build question_choice query
    q = f"question={question_sys_id}^ORDERBYsequence"
    if not include_inactive:
        q = f"question={question_sys_id}^active=true^ORDERBYsequence"

    params = {
        'sysparm_query': q,
        'sysparm_fields': 'label,value,sequence,active',
        'sysparm_display_value': 'all',
        'sysparm_limit': '1000'
    }

    choices_rows = get_with_retry(f"{table_base}/question_choice", params=params)
    if not choices_rows:
        # return empty choices array if no rows
        return {'resolved_table': resolved_table, 'question_sys_id': question_sys_id, 'choices': []}

    choices_out = []
    for r in choices_rows:
        # Normalize value which can be a primitive or an object {value, display_value}
        raw_val = r.get('value')
        if isinstance(raw_val, dict):
            value = raw_val.get('value') or raw_val.get('display_value') or ''
            dv = raw_val.get('display_value') or ''
        else:
            value = raw_val or ''
            dv = ''

        # Normalize label: prefer explicit label, then display_value, then value
        label = r.get('label') or r.get('display_value') or dv or ''

        seq = r.get('sequence') or r.get('order') or r.get('sys_sequence')
        try:
            seq_num = int(seq) if seq is not None and str(seq).isdigit() else 0
        except Exception:
            seq_num = 0
        active_flag = False
        a = r.get('active')
        if isinstance(a, bool):
            active_flag = a
        else:
            active_flag = str(a).lower() in ('true', '1', 'yes')
        choices_out.append({'label': str(label), 'value': str(value), 'sequence': seq_num, 'active': bool(active_flag)})

    # Ensure sorted by sequence asc
    choices_out = sorted(choices_out, key=lambda x: x.get('sequence', 0))

    return {'resolved_table': resolved_table, 'question_sys_id': question_sys_id, 'choices': choices_out}


# Fallback for client-side routing: if the path doesn't start with /api and the file doesn't exist,
# return index.html so the SPA router can handle the path. This is placed at the end of the file so
# explicit API routes are registered and matched first.
@app.get('/{full_path:path}', include_in_schema=False)
def spa_fallback(full_path: str, request: Request):
    # If this is an API call or known API prefix, let the normal routes handle it
    if full_path.startswith(('api', 'catalog', 'chat', 'ticket', 'debug')):
        raise HTTPException(status_code=404)

    if STATIC_SOURCE:
        candidate = os.path.join(STATIC_SOURCE, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        # else return index
        index_path = os.path.join(STATIC_SOURCE, 'index.html')
        if os.path.isfile(index_path):
            return FileResponse(index_path)

    # If no frontend, return a helpful JSON
    return JSONResponse({"status": "ok", "message": "No frontend found"})
