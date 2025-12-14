AskGen React UI

This is a Vite + React scaffold of the Streamlit AskGen UI.

Commands:
- npm install
- npm run dev

Environment:
- VITE_BACKEND_URL defaults to http://127.0.0.1:8000

Optional environment variables for multi-backend support:

- VITE_BACKEND_CLOUD_URL: URL for the Cloud LLM backend (defaults to VITE_BACKEND_URL if not set).
- VITE_BACKEND_ONPREM_URL: URL for the OnPrem LLM backend (defaults to http://127.0.0.1:8001).

Frontend toggle:

The UI includes a Backend selector (Cloud LLM / OnPrem LLM). The selected mode is persisted to localStorage as `backend_mode` and controls whether the app sends requests to the Cloud or OnPrem backend URL. Use these env vars to point each mode to the appropriate server.

Example usage:

1. Start the cloud backend (e.g. `backend.py`) on port 8000.
2. Start the on-prem backend (e.g. `backend2.py`) on port 8001.
3. Run the React app with `npm run dev` and pick the desired backend in the UI.
