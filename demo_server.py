"""
AskOps Demo Server - Lightweight demo server with mock responses
This server demonstrates the UI without requiring Azure OpenAI or ServiceNow credentials.
"""

import os
import random
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(
    title="AskOps Demo Server",
    description="AI-Powered ITSM Assistant Demo",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the built frontend
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "askgen-react", "dist")

if os.path.isdir(STATIC_DIR):
    # Mount assets separately to avoid conflicts
    assets_dir = os.path.join(STATIC_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Mock data
MOCK_TICKETS = {
    "INC0010001": {
        "number": "INC0010001",
        "status": "In Progress",
        "priority": "2",
        "assigned_to": "John Smith",
        "short_description": "VPN Connection Issues",
        "description": "User reports intermittent VPN disconnections when working from home.",
        "created": "2024-12-10 09:30:00"
    },
    "INC0010002": {
        "number": "INC0010002",
        "status": "Resolved",
        "priority": "3",
        "assigned_to": "Jane Doe",
        "short_description": "Email not syncing on mobile",
        "description": "Corporate email not syncing on mobile device after recent password change.",
        "created": "2024-12-11 14:15:00"
    },
    "REQ0010001": {
        "number": "REQ0010001",
        "status": "New",
        "priority": "4",
        "assigned_to": "Unassigned",
        "short_description": "New laptop request",
        "description": "Request for new MacBook Pro for development work.",
        "created": "2024-12-12 10:00:00"
    }
}

TROUBLESHOOTING_RESPONSES = {
    "password": [
        "I can help you with password reset! Here are your options:\n\n**Self-Service Reset:**\n1. Go to the password reset portal at password.company.com\n2. Enter your username and click 'Forgot Password'\n3. Follow the email verification steps\n\nIf you're locked out, I can create a ticket to have IT unlock your account. Would you like me to do that?",
        "To reset your password:\n\n1. **Via Self-Service Portal:** Visit password.company.com and follow the reset flow\n2. **Via Phone:** Call the helpdesk at ext. 4357 with your employee ID\n\nWould you like me to open a password reset ticket for you?"
    ],
    "vpn": [
        "Let's troubleshoot your VPN connection:\n\n**Quick Fixes:**\n1. Disconnect and reconnect the VPN\n2. Check your internet connection\n3. Make sure VPN client is updated to the latest version\n\n**If still not working:**\n- Try restarting your computer\n- Switch to a different network if available\n- Check if VPN server is under maintenance (status.company.com)\n\nDid any of these steps help?",
        "VPN issues are common! Let's fix this:\n\n**Step 1:** Right-click the VPN icon and select 'Disconnect'\n**Step 2:** Wait 30 seconds, then reconnect\n**Step 3:** If prompted, allow the connection and enter your credentials\n\nIf the problem persists, please try rebooting your machine. Let me know if you need further assistance!"
    ],
    "laptop": [
        "I understand your laptop won't boot. Let's troubleshoot:\n\n**Immediate steps:**\n1. **Check power:** Is it plugged in? Try a different outlet.\n2. **Hard reset:** Hold the power button for 15 seconds, then release and press again.\n3. **External display:** Connect to an external monitor to check if it's a display issue.\n\n**If still no response:**\n- Check for any LED lights or sounds when pressing power\n- Remove any USB devices and try again\n\nWould you like me to escalate this to hardware support?",
        "Let's get your laptop working:\n\n**Quick diagnostics:**\n1. Is the charging light on when plugged in?\n2. Does it make any sounds when you press the power button?\n3. Have you tried holding power for 15+ seconds?\n\nIf none of these work, it might be a hardware issue and I can create a ticket for you to get a replacement or repair."
    ],
    "default": [
        "I understand you're experiencing an IT issue. Let me help!\n\n**Before we proceed, can you tell me:**\n- What exactly is happening?\n- When did this issue start?\n- Have you tried any troubleshooting steps?\n\nThis will help me provide better assistance or create an accurate ticket if needed.",
        "Thanks for reaching out! I'm here to help with your IT needs.\n\n**Common issues I can assist with:**\n- Password resets\n- VPN connectivity\n- Software installation requests\n- Hardware problems\n- Access requests\n\nWhat specific issue are you experiencing today?"
    ]
}

AGENT_STAGES = [
    "Orchestrator",
    "Classification Agent",
    "Troubleshooting Agent", 
    "Ticketing Agent"
]


class ChatRequest(BaseModel):
    message: str
    session_state: Optional[Dict[str, Any]] = {}


def get_mock_response(message: str) -> Dict[str, Any]:
    """Generate a mock response based on the user's message."""
    msg_lower = message.lower()
    
    # Check for ticket lookup
    for prefix in ["inc", "req", "ritm"]:
        if prefix in msg_lower:
            # Extract ticket number
            import re
            match = re.search(r'(INC|REQ|RITM)\d+', message, re.IGNORECASE)
            if match:
                ticket_id = match.group().upper()
                if ticket_id in MOCK_TICKETS:
                    return {
                        "response": f"Here's the status of ticket {ticket_id}:",
                        "ticket": MOCK_TICKETS[ticket_id],
                        "agent_stage": "Status Check Agent"
                    }
                else:
                    return {
                        "response": f"I couldn't find ticket {ticket_id}. Please verify the ticket number and try again.",
                        "agent_stage": "Status Check Agent"
                    }
    
    # Determine category and get response
    if any(word in msg_lower for word in ["password", "reset", "unlock", "login", "credentials"]):
        response = random.choice(TROUBLESHOOTING_RESPONSES["password"])
        agent = "Troubleshooting Agent"
    elif any(word in msg_lower for word in ["vpn", "connect", "network", "remote", "work from home"]):
        response = random.choice(TROUBLESHOOTING_RESPONSES["vpn"])
        agent = "Troubleshooting Agent"
    elif any(word in msg_lower for word in ["laptop", "computer", "boot", "start", "turn on", "power"]):
        response = random.choice(TROUBLESHOOTING_RESPONSES["laptop"])
        agent = "Troubleshooting Agent"
    elif any(word in msg_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        response = "Hello! 👋 I'm AskOps, your AI-powered IT assistant. How can I help you today?"
        agent = "Greeting Agent"
    elif any(word in msg_lower for word in ["thanks", "thank you", "bye", "goodbye"]):
        response = "You're welcome! Feel free to reach out if you need anything else. Have a great day! 🌟"
        agent = "Orchestrator"
    elif any(word in msg_lower for word in ["create ticket", "open ticket", "escalate", "help desk"]):
        # Simulate ticket creation
        new_ticket_num = f"INC{random.randint(1000000, 9999999)}"
        response = f"I've created a ticket for you:\n\n**Ticket Number:** {new_ticket_num}\n**Status:** New\n**Priority:** Medium\n\nA technician will review your request and reach out within 4 business hours."
        agent = "Ticketing Agent"
        return {
            "response": response,
            "ticket": {
                "number": new_ticket_num,
                "status": "New",
                "priority": "3",
                "assigned_to": "Unassigned",
                "short_description": "User request via AskOps"
            },
            "agent_stage": agent
        }
    else:
        response = random.choice(TROUBLESHOOTING_RESPONSES["default"])
        agent = "Classification Agent"
    
    return {
        "response": response,
        "agent_stage": agent
    }


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend index.html"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "status": "ok",
        "message": "AskOps Demo Server is running",
        "endpoints": {
            "chat": "POST /chat/",
            "ticket": "GET /ticket/{id}",
            "status": "GET /api/status"
        }
    })


@app.get("/api/status")
async def api_status():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "mode": "demo",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.post("/chat/")
async def chat(request: ChatRequest):
    """Main chat endpoint with mock AI responses"""
    message = request.message.strip()
    session_state = request.session_state or {}
    
    if not message:
        return JSONResponse({
            "response": "I didn't catch that. Could you please repeat your question?",
            "agent_stage": "Orchestrator"
        })
    
    # Simulate processing time for realism
    time.sleep(0.5)
    
    result = get_mock_response(message)
    result["session_state"] = session_state
    
    return JSONResponse(result)


@app.get("/ticket/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get ticket status by ID"""
    ticket_id = ticket_id.upper()
    
    if ticket_id in MOCK_TICKETS:
        return JSONResponse(MOCK_TICKETS[ticket_id])
    
    # Return 404 for unknown tickets
    return JSONResponse(
        status_code=404,
        content={"error": f"Ticket {ticket_id} not found"}
    )


@app.get("/catalog/item/{item_id}")
async def get_catalog_item(item_id: str):
    """Get catalog item details (mock)"""
    return JSONResponse({
        "sys_id": item_id,
        "name": "Software Installation Request",
        "short_description": "Request for software installation",
        "variables": [
            {"name": "software_name", "label": "Software Name", "type": "string", "mandatory": True},
            {"name": "justification", "label": "Business Justification", "type": "textarea", "mandatory": True},
            {"name": "urgency", "label": "Urgency", "type": "choice", "choices": ["Low", "Medium", "High"], "mandatory": False}
        ]
    })


@app.get("/api/rag/health")
async def rag_health():
    """RAG engine health check (mock)"""
    return JSONResponse({
        "status": "healthy",
        "mode": "demo",
        "documents": 0,
        "message": "RAG engine is in demo mode"
    })


# SPA fallback - serve index.html for all unmatched routes
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """Serve index.html for client-side routing"""
    # Skip if this looks like an API call
    if full_path.startswith("api/") or full_path.startswith("chat") or full_path.startswith("ticket"):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    
    return JSONResponse(status_code=404, content={"error": "Not found"})


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AskOps Demo Server...")
    print(f"📁 Serving frontend from: {STATIC_DIR}")
    print(f"🌐 Server will be available at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
