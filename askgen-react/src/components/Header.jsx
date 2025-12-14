import React, { useState, useEffect } from 'react'
import UserSettings from './UserSettings'

export default function Header({ onNavigate, currentPage }) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [compact, setCompact] = useState(() => {
    try { return localStorage.getItem('compact') === 'true' } catch { return false }
  })
  const [darkMode, setDarkMode] = useState(() => {
    try { return localStorage.getItem('dark_mode') === 'true' } catch { return false }
  })

  useEffect(() => {
    try { localStorage.setItem('compact', compact.toString()) } catch {}
    document.body.classList.toggle('compact', compact)
  }, [compact])
  
  useEffect(() => {
    try { localStorage.setItem('dark_mode', darkMode.toString()) } catch {}
    document.body.classList.toggle('dark-mode', darkMode)
  }, [darkMode])

  return (
    <>
      <header className="app-header">
        <div className="brand">
          <div className="logo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div className="brand-text">
            <div className="title">AskOps</div>
            <div className="subtitle">AI-Powered ITSM Assistant</div>
          </div>
        </div>

        <div className="header-actions">
          <div className="header-toggle-group">
            <button 
              className={`header-toggle ${compact ? 'active' : ''}`}
              onClick={() => setCompact(!compact)}
              title="Compact view"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <line x1="9" y1="3" x2="9" y2="21"/>
              </svg>
            </button>
            <button 
              className={`header-toggle ${darkMode ? 'active' : ''}`}
              onClick={() => setDarkMode(!darkMode)}
              title="Dark mode"
            >
              {darkMode ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5"/>
                  <line x1="12" y1="1" x2="12" y2="3"/>
                  <line x1="12" y1="21" x2="12" y2="23"/>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                  <line x1="1" y1="12" x2="3" y2="12"/>
                  <line x1="21" y1="12" x2="23" y2="12"/>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </button>
          </div>
          
          <button className="avatar-btn" onClick={() => setSettingsOpen(true)} title="Profile">
            U
          </button>
        </div>
      </header>
      
      <UserSettings visible={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  )
}
