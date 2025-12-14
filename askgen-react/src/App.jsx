import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import LLMSelector from './components/LLMSelector'
import TicketsDashboard from './components/TicketsDashboard'
import Integrations from './components/Integrations'
import AuditPage from './components/AuditPage'
import AgentAssistPage from './components/AgentAssistPage'
import './styles.css'

export default function App() {
  const [page, setPage] = useState('self')
  const [llmMode, setLlmMode] = useState(() => {
    try { return localStorage.getItem('llm_mode') || 'cloud' } catch { return 'cloud' }
  })
  const [settingsDrawerOpen, setSettingsDrawerOpen] = useState(false)

  useEffect(() => {
    try { localStorage.setItem('llm_mode', llmMode) } catch {}
  }, [llmMode])

  return (
    <div className={`app-root ${page === 'self' ? 'self-page' : ''} ${settingsDrawerOpen ? 'drawer-open' : ''}`}>
      <Header onNavigate={setPage} currentPage={page} />
      
      <div className="main-grid">
        <Sidebar currentPage={page} onNavigate={setPage} />
        
        <main className="main-col">
          <div className="page-content">
            {page === 'self' && (
              <>
                <ChatPanel llmMode={llmMode} />
                <div className="right-panel">
                  <div className="right-panel-inner">
                    <div className="right-panel-content">
                      <LLMSelector mode={llmMode} setMode={setLlmMode} />
                    </div>
                  </div>
                </div>
              </>
            )}
            
            {page === 'agent-assist' && <AgentAssistPage />}
            {page === 'tickets' && <TicketsDashboard />}
            {page === 'audit' && <AuditPage />}
            {page === 'integrations' && <Integrations />}
            {page === 'settings' && (
              <div className="page-container">
                <div className="page-header">
                  <h1>Settings</h1>
                  <p>Configure your AskOps experience</p>
                </div>
                <div className="card">
                  <LLMSelector mode={llmMode} setMode={setLlmMode} />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Mobile Settings Drawer */}
      {settingsDrawerOpen && (
        <div className="drawer-overlay">
          <div className="drawer-backdrop" onClick={() => setSettingsDrawerOpen(false)} />
          <div className="drawer-panel">
            <button className="drawer-close" onClick={() => setSettingsDrawerOpen(false)}>✕</button>
            <LLMSelector mode={llmMode} setMode={setLlmMode} />
          </div>
        </div>
      )}
    </div>
  )
}
