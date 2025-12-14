import React from 'react'

export default function ChatBubble({ speaker, text, ticket, timestamp, sessionState }) {
  
  // Progress/Loading bubble
  if (speaker === 'progress') {
    return (
      <div className="chat-row">
        <div className="chat-avatar">🤖</div>
        <div className="chat-bubble progress-bubble">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="loader-dots">
              <span></span><span></span><span></span>
            </div>
            <span style={{ color: '#92400E', fontSize: '13px' }}>{text}</span>
          </div>
        </div>
      </div>
    )
  }

  // Catalog suggestion
  if (speaker === 'catalog_suggestion' && text) {
    const item = text.item || text
    return (
      <div className="chat-row">
        <div className="chat-avatar">🤖</div>
        <div className="chat-bubble bot-bubble catalog-card">
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontWeight: 600, marginBottom: '4px' }}>{item.name}</div>
            <div style={{ fontSize: '13px', color: '#6B7280' }}>{item.short_description}</div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={() => window.dispatchEvent(new CustomEvent('askgen:catalog:confirm', { detail: item }))}
              style={{ background: '#10B981', color: '#fff' }}
            >
              ✓ Yes, use this
            </button>
            <button 
              onClick={() => window.dispatchEvent(new CustomEvent('askgen:catalog:decline', { detail: item }))}
              style={{ background: '#fff', color: '#6B7280', border: '1px solid #D1D5DB' }}
            >
              ✗ No
            </button>
          </div>
        </div>
      </div>
    )
  }

  // User or Bot message
  const isUser = speaker === 'user'
  
  return (
    <div className={`chat-row ${isUser ? 'user-row' : ''}`}>
      {!isUser && <div className="chat-avatar">🤖</div>}
      
      <div className={`chat-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
        <div style={{ whiteSpace: 'pre-wrap' }}>{text}</div>
        
        {/* Ticket Card */}
        {ticket && (
          <div className="ticket-card">
            <div className="ticket-header">
              <span className="ticket-number">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: '6px', verticalAlign: 'middle' }}>
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                {ticket.number || ticket.id}
              </span>
              <span className={`ticket-priority priority-${ticket.priority || '3'}`} style={{
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 500,
                background: getPriorityColor(ticket.priority).bg,
                color: getPriorityColor(ticket.priority).text
              }}>
                {getPriorityLabel(ticket.priority)}
              </span>
            </div>
            <div className="ticket-info">
              <div className="ticket-info-row">
                <span className="ticket-info-label">Status:</span>
                <span className="ticket-info-value" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: getStatusColor(ticket.status || ticket.state)
                  }}></span>
                  {ticket.status || ticket.state || 'Unknown'}
                </span>
              </div>
              <div className="ticket-info-row">
                <span className="ticket-info-label">Assigned:</span>
                <span className="ticket-info-value">{ticket.assigned_to || 'Unassigned'}</span>
              </div>
              {ticket.description && (
                <div style={{ marginTop: '8px', fontSize: '12px', color: '#6B7280' }}>
                  {ticket.description.slice(0, 150)}{ticket.description.length > 150 ? '...' : ''}
                </div>
              )}
            </div>
          </div>
        )}
        
        {timestamp && <div className="ts">{timestamp}</div>}
      </div>
      
      {isUser && <div className="chat-avatar" style={{ background: '#1C64F2', color: '#fff' }}>👤</div>}
    </div>
  )
}

function getPriorityColor(priority) {
  const colors = {
    '1': { bg: '#FEE2E2', text: '#DC2626' },
    '2': { bg: '#FEF3C7', text: '#D97706' },
    '3': { bg: '#DBEAFE', text: '#1D4ED8' },
    '4': { bg: '#D1FAE5', text: '#059669' },
    '5': { bg: '#E5E7EB', text: '#6B7280' }
  }
  return colors[priority] || colors['3']
}

function getPriorityLabel(priority) {
  const labels = { '1': 'Critical', '2': 'High', '3': 'Medium', '4': 'Low', '5': 'Planning' }
  return labels[priority] || 'Medium'
}

function getStatusColor(status) {
  const colors = {
    'New': '#3B82F6',
    'In Progress': '#F59E0B',
    'Resolved': '#10B981',
    'Closed': '#6B7280',
    'On Hold': '#8B5CF6'
  }
  return colors[status] || '#6B7280'
}
