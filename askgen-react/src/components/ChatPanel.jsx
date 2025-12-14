import React, { useEffect, useRef, useState } from 'react'
import ChatBubble from './ChatBubble'
import AgentVisualizer from './AgentVisualizer'
import IntelligentTriageCard from './IntelligentTriageCard'
import AgentAssistDashboard from './AgentAssistDashboard'

const CLOUD_URL = (import.meta.env.VITE_BACKEND_CLOUD_URL) || (import.meta.env.VITE_BACKEND_URL) || ''

const ticketRegexes = [
  /((?:INC|REQ|RITM)[-_]?\d+)/i,
  /check ticket\s+([A-Za-z0-9_-]+)/i,
  /status of ticket\s+([A-Za-z0-9_-]+)/i
]

export default function ChatPanel({ llmMode }) {
  const [chatHistory, setChatHistory] = useState(() => {
    try {
      const raw = localStorage.getItem('chat_history')
      return raw ? JSON.parse(raw) : []
    } catch { return [] }
  })
  const [inputValue, setInputValue] = useState('')
  const [sessionState, setSessionState] = useState({})
  const [activeAgent, setActiveAgent] = useState('Orchestrator')
  const [errorToast, setErrorToast] = useState(null)
  const [backendUrl] = useState(CLOUD_URL)
  const [showAgentAssist, setShowAgentAssist] = useState(false)
  const [currentTriageResult, setCurrentTriageResult] = useState(null)
  const [intelligentTriageEnabled, setIntelligentTriageEnabled] = useState(true)

  const messagesRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify(chatHistory))
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    }
  }, [chatHistory])

  useEffect(() => {
    if (chatHistory.length === 0) {
      setChatHistory([{
        speaker: 'bot',
        text: "👋 Hello! I'm AskOps, your AI-powered IT assistant.\n\nI can help you with:\n• Troubleshooting IT issues\n• Checking ticket status\n• Requesting services\n\nHow can I help you today?",
        ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }])
    }
  }, [])

  function showError(msg) {
    setErrorToast(msg)
    setTimeout(() => setErrorToast(null), 4500)
  }

  function detectTicketId(text) {
    for (const r of ticketRegexes) {
      const m = text.match(r)
      if (m) return m[1] || m[0]
    }
    return null
  }

  async function checkTicket(id) {
    try {
      setActiveAgent('Status Check Agent')
      const res = await fetch(`${backendUrl}/ticket/${id}`)
      if (!res.ok) throw new Error(`Ticket not found`)
      return await res.json()
    } catch (e) {
      return { error: e.message }
    } finally {
      setActiveAgent('Orchestrator')
    }
  }

  async function postChat(payload) {
    try {
      const res = await fetch(`${backendUrl}/chat/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) throw new Error(`Request failed`)
      return await res.json()
    } catch (e) {
      return { error: e.message }
    }
  }

  // Intelligent triage before processing
  async function triageIssue(description) {
    try {
      const res = await fetch(`${backendUrl}/api/itops/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description,
          session_state: sessionState
        })
      })
      if (!res.ok) return null
      const data = await res.json()
      if (data.status === 'ok') {
        return data.triage_result
      }
      return null
    } catch (e) {
      console.log('Triage not available:', e.message)
      return null
    }
  }

  async function sendMessage() {
    const text = inputValue.trim()
    if (!text) return

    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    setChatHistory(ch => [...ch, { speaker: 'user', text, ts }])
    setInputValue('')

    const ticketId = detectTicketId(text)
    
    if (ticketId) {
      setChatHistory(ch => [...ch, { speaker: 'progress', text: `Looking up ${ticketId}...` }])
      const res = await checkTicket(ticketId)
      setChatHistory(ch => {
        const filtered = ch.filter(c => c.speaker !== 'progress')
        if (res.error) {
          return [...filtered, { speaker: 'bot', text: `Could not find ticket ${ticketId}. Please check the number.`, ts }]
        }
        return [...filtered, { speaker: 'bot', text: `Here's your ticket status:`, ts, ticket: res }]
      })
      return
    }

    setChatHistory(ch => [...ch, { speaker: 'progress', text: 'Analyzing your request with AI...' }])
    setActiveAgent('Intelligent Triage')
    
    // Try intelligent triage first
    let triageResult = null
    if (intelligentTriageEnabled) {
      triageResult = await triageIssue(text)
      if (triageResult) {
        setCurrentTriageResult(triageResult)
        setActiveAgent(triageResult.decision === 'auto_resolve' ? 'Auto-Resolution Agent' : 'Triage Agent')
        
        // If auto-resolved, show success and triage card
        if (triageResult.auto_resolution_successful) {
          setChatHistory(ch => {
            const filtered = ch.filter(c => c.speaker !== 'progress')
            return [...filtered, { 
              speaker: 'bot', 
              text: triageResult.customer_message,
              ts,
              triageResult 
            }]
          })
          setActiveAgent('Orchestrator')
          return
        }
      }
    }
    
    // Proceed with normal chat flow
    setActiveAgent('Classification Agent')
    const result = await postChat({ message: text, session_state: sessionState })
    setActiveAgent(result?.agent_stage || 'Orchestrator')

    setChatHistory(ch => {
      const filtered = ch.filter(c => c.speaker !== 'progress')
      if (result.error) {
        showError(result.error)
        return [...filtered, { speaker: 'bot', text: `Sorry, something went wrong. Please try again.`, ts }]
      }
      if (result.ticket) {
        return [...filtered, { speaker: 'bot', text: result.response || 'Ticket created:', ts, ticket: result.ticket }]
      }
      if (result.catalog_suggestion) {
        const sug = result.catalog_suggestion.item || result.catalog_suggestion
        return [...filtered, { speaker: 'catalog_suggestion', text: sug, ts }]
      }
      // Include triage result in response if available
      return [...filtered, { 
        speaker: 'bot', 
        text: result.response || 'Done.', 
        ts,
        triageResult: triageResult 
      }]
    })

    if (result.session_state) setSessionState(result.session_state)
  }

  function clearChat() {
    setChatHistory([{
      speaker: 'bot',
      text: "👋 Hello! How can I help you today?",
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }])
    setSessionState({})
    localStorage.removeItem('chat_history')
    inputRef.current?.focus()
  }

  async function sendQuickAction(text) {
    setInputValue(text)
    setTimeout(() => sendMessage(), 100)
  }

  const quickActions = [
    { label: 'Reset password', icon: '🔑' },
    { label: 'VPN not working', icon: '🌐' },
    { label: "Laptop won't boot", icon: '💻' }
  ]

  return (
    <div className="chat-wrapper">
      <div className="chat-card">
        <div className="panel-header">
          <div className="panel-left">
            <div className="panel-badge">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
              </svg>
              AskOps
            </div>
            <span className="panel-meta">Active: <strong>{activeAgent}</strong></span>
          </div>
          <button className="clear-chat-btn" onClick={clearChat}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
            </svg>
            Clear
          </button>
        </div>

        <AgentVisualizer active={activeAgent} />

        <div className="chat-messages" ref={messagesRef}>
          {chatHistory.map((m, idx) => (
            <React.Fragment key={idx}>
              <ChatBubble
                speaker={m.speaker}
                text={m.text}
                ticket={m.ticket}
                timestamp={m.ts}
                sessionState={sessionState}
              />
              {m.triageResult && (
                <IntelligentTriageCard 
                  triageResult={m.triageResult}
                  onViewDetails={() => setShowAgentAssist(true)}
                />
              )}
            </React.Fragment>
          ))}
        </div>
        
        {/* Agent Assist Modal */}
        {showAgentAssist && currentTriageResult && (
          <div className="agent-assist-modal">
            <div className="modal-backdrop" onClick={() => setShowAgentAssist(false)} />
            <div className="modal-content">
              <AgentAssistDashboard 
                ticketId={currentTriageResult.ticket_id}
                issueDescription={chatHistory.filter(m => m.speaker === 'user').pop()?.text || ''}
                onClose={() => setShowAgentAssist(false)}
              />
            </div>
          </div>
        )}

        <div className="quick-actions">
          {quickActions.map(action => (
            <button key={action.label} className="pill" onClick={() => { setInputValue(action.label); }}>
              <span>{action.icon}</span>
              <span>{action.label}</span>
            </button>
          ))}
        </div>

        <div className="sticky-input">
          <input
            ref={inputRef}
            placeholder="Describe your issue or enter a ticket number..."
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
          />
          <button onClick={sendMessage}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        {errorToast && <div className="error-toast">⚠️ {errorToast}</div>}
      </div>
      
      <style>{`
        .agent-assist-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .modal-backdrop {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
        }
        
        .modal-content {
          position: relative;
          width: 90%;
          max-width: 700px;
          max-height: 90vh;
          z-index: 1001;
        }
      `}</style>
    </div>
  )
}
