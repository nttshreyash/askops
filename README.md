# AskOps - Intelligent ITOps Automation Platform

An AI-powered IT Operations automation platform that provides intelligent ticket resolution, prescriptive analytics, and seamless ServiceNow integration.

## Features

### Core Capabilities

- **Auto-Resolution Engine** - Automatically resolves common IT issues without human intervention
- **Prescriptive Analytics** - Provides step-by-step guidance to IT agents for faster resolution
- **Runbook Automation** - Executes predefined runbooks for common issues (password reset, VPN troubleshooting, etc.)
- **Agent Assist Dashboard** - Real-time diagnosis, suggestions, and quick actions for support staff
- **Incident Correlation** - Detects patterns and correlates related incidents
- **Learning Loop** - Continuously improves from resolved tickets and feedback
- **RAG-Enhanced Knowledge Base** - Semantic search across knowledge articles and past resolutions

### Integrations

- **ServiceNow ITSM** - Full integration with incidents, requests, and service catalog
- **Azure OpenAI** - Intelligent responses powered by GPT models
- **Real-time Status Monitoring** - Live connection status for all integrations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │   Chat   │ │ Tickets  │ │Integration│ │  Agent Assist   │   │
│  │  Panel   │ │Dashboard │ │  Status   │ │   Dashboard     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LangGraph Orchestrator                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │   │
│  │  │Greeting │ │Classify │ │Troubleshoot│ │   Ticketing   │ │   │
│  │  │Detector │ │  Agent  │ │   Agent    │ │     Agent     │ │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ITOps Engine                           │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │Auto-Resolver│ │Agent Assist │ │ Incident Correlator │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │   Triage    │ │   Runbook   │ │  Learning Engine    │ │   │
│  │  │   Engine    │ │   Engine    │ │                     │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RAG Engine                             │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │Vector Store │ │ Embeddings  │ │ Knowledge Learner   │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │  ServiceNow  │ │ Azure OpenAI │ │   Active Directory       │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- ServiceNow instance with API access
- Azure OpenAI deployment

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nttshreyash/askops.git
cd askops
```

2. Create and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install and build frontend:
```bash
cd askgen-react
npm install
npm run build
cd ..
```

5. Run the server:
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

6. Access the application at `http://localhost:8000`

## Configuration

Create a `.env` file with the following variables:

```env
# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# ServiceNow settings
SERVICE_NOW_INSTANCE=your-instance.service-now.com
SERVICE_NOW_USER=your-username
SERVICE_NOW_PASS=your-password

# Optional settings
VERIFY_TLS=true
AZURE_VERIFY_TLS=false
```

## API Endpoints

### Chat
- `POST /chat/` - Main chat endpoint for user interactions

### Tickets
- `GET /api/tickets` - Fetch tickets from ServiceNow (incidents + requests)
- `GET /ticket/{ticket_number}` - Get specific ticket details

### ITOps Engine
- `POST /api/itops/triage` - Intelligent ticket triage
- `POST /api/itops/auto-resolve` - Attempt automatic resolution
- `POST /api/itops/agent-assist` - Get agent assistance
- `GET /api/itops/runbooks` - List available runbooks
- `POST /api/itops/correlate` - Correlate incidents

### Integrations
- `GET /api/integrations/status` - Check all integration statuses
- `GET /api/status` - Health check

### Catalog
- `GET /catalog/items` - List ServiceNow catalog items
- `GET /catalog/item/{item_id}` - Get catalog item details
- `POST /catalog/order` - Submit catalog order

## Project Structure

```
askops/
├── backend.py              # Main FastAPI application
├── itops_engine/           # ITOps automation modules
│   ├── __init__.py
│   ├── auto_resolver.py    # Auto-resolution engine
│   ├── agent_assist.py     # Agent assistance module
│   ├── triage_engine.py    # Intelligent triage
│   ├── runbook_engine.py   # Runbook automation
│   ├── incident_correlator.py
│   ├── learning_engine.py
│   └── itops_api.py        # ITOps API router
├── rag_engine/             # RAG knowledge base
│   ├── __init__.py
│   ├── rag_engine.py       # Main RAG engine
│   ├── vector_store.py     # Vector storage
│   ├── embeddings.py       # Embedding generation
│   ├── knowledge_learner.py
│   └── api.py              # RAG API router
├── askgen-react/           # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.jsx
│   │   └── styles.css
│   └── package.json
├── requirements.txt
├── Dockerfile
└── README.md
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
