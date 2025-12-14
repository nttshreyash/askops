import React, { useState, useEffect } from 'react';

/**
 * Agent Assist Dashboard
 * 
 * Provides prescriptive guidance to IT agents with:
 * - Real-time suggestions
 * - Diagnostic commands
 * - Similar resolved cases
 * - KB article recommendations
 * - Quick actions
 */
export default function AgentAssistDashboard({ ticketId, issueDescription, onClose }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [assistance, setAssistance] = useState(null);
  const [activeTab, setActiveTab] = useState('suggestions');
  const [executingAction, setExecutingAction] = useState(null);
  const [copiedCommand, setCopiedCommand] = useState(null);

  // API base URL
  const getApiUrl = () => {
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
      return '';
    }
    return '';
  };

  // Fetch agent assistance
  useEffect(() => {
    if (!issueDescription) return;
    
    const fetchAssistance = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${getApiUrl()}/api/itops/agent-assist`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticket_id: ticketId || 'temp_' + Date.now(),
            issue_description: issueDescription,
          })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        if (data.status === 'ok') {
          setAssistance(data.assistance);
        } else {
          throw new Error(data.message || 'Failed to get assistance');
        }
      } catch (err) {
        console.error('Agent assist error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAssistance();
  }, [ticketId, issueDescription]);

  // Copy command to clipboard
  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedCommand(id);
      setTimeout(() => setCopiedCommand(null), 2000);
    });
  };

  // Execute quick action
  const executeAction = async (action) => {
    setExecutingAction(action.action_id);
    // In a real implementation, this would call the backend
    setTimeout(() => {
      setExecutingAction(null);
    }, 2000);
  };

  // Priority badge color
  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return '#ef4444';
      case 'high': return '#f97316';
      case 'medium': return '#eab308';
      case 'low': return '#22c55e';
      default: return '#6b7280';
    }
  };

  // Confidence indicator
  const ConfidenceIndicator = ({ confidence }) => {
    const percent = Math.round(confidence * 100);
    const color = percent >= 80 ? '#22c55e' : percent >= 60 ? '#eab308' : '#ef4444';
    return (
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '12px',
        color: color
      }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10"/>
          <path d={`M12 2 A 10 10 0 ${percent > 50 ? 1 : 0} 1 ${12 + 10 * Math.sin(percent * 3.6 * Math.PI / 180)} ${12 - 10 * Math.cos(percent * 3.6 * Math.PI / 180)}`}/>
        </svg>
        {percent}% confident
      </span>
    );
  };

  if (loading) {
    return (
      <div className="agent-assist-dashboard loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Analyzing ticket and generating recommendations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="agent-assist-dashboard error">
        <div className="error-message">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <p>Failed to load assistance: {error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  if (!assistance) return null;

  return (
    <div className="agent-assist-dashboard">
      {/* Header */}
      <div className="assist-header">
        <div className="assist-title">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
          </svg>
          <h2>AI Agent Assist</h2>
          {assistance.escalation_recommended && (
            <span className="escalation-badge">Escalation Recommended</span>
          )}
        </div>
        {onClose && (
          <button className="close-btn" onClick={onClose}>×</button>
        )}
      </div>

      {/* Summary */}
      <div className="assist-summary">
        <div className="summary-item">
          <span className="label">Ticket:</span>
          <span className="value">{assistance.ticket_id}</span>
        </div>
        <div className="summary-item">
          <span className="label">Est. Time:</span>
          <span className="value">{assistance.estimated_resolution_time} min</span>
        </div>
        <div className="summary-item">
          <span className="label">Priority:</span>
          <span className="value">{assistance.priority_assessment}</span>
        </div>
      </div>

      {/* Diagnosis */}
      <div className="assist-diagnosis">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
            <path d="M14 2v6h6"/>
            <path d="M16 13H8"/>
            <path d="M16 17H8"/>
            <path d="M10 9H8"/>
          </svg>
          Diagnosis
        </h3>
        <p>{assistance.diagnosis}</p>
      </div>

      {/* Tabs */}
      <div className="assist-tabs">
        <button 
          className={activeTab === 'suggestions' ? 'active' : ''} 
          onClick={() => setActiveTab('suggestions')}
        >
          Suggestions ({assistance.suggestions?.length || 0})
        </button>
        <button 
          className={activeTab === 'similar' ? 'active' : ''} 
          onClick={() => setActiveTab('similar')}
        >
          Similar Cases ({assistance.similar_tickets?.length || 0})
        </button>
        <button 
          className={activeTab === 'kb' ? 'active' : ''} 
          onClick={() => setActiveTab('kb')}
        >
          KB Articles ({assistance.kb_articles?.length || 0})
        </button>
        <button 
          className={activeTab === 'actions' ? 'active' : ''} 
          onClick={() => setActiveTab('actions')}
        >
          Quick Actions
        </button>
      </div>

      {/* Tab Content */}
      <div className="assist-content">
        {/* Suggestions Tab */}
        {activeTab === 'suggestions' && (
          <div className="suggestions-list">
            {assistance.suggestions?.map((suggestion, idx) => (
              <div key={suggestion.id || idx} className="suggestion-card">
                <div className="suggestion-header">
                  <span 
                    className="priority-badge"
                    style={{ backgroundColor: getPriorityColor(suggestion.priority) }}
                  >
                    {suggestion.priority}
                  </span>
                  <span className="type-badge">{suggestion.type.replace('_', ' ')}</span>
                  <ConfidenceIndicator confidence={suggestion.confidence} />
                </div>
                <h4>{suggestion.title}</h4>
                <p>{suggestion.description}</p>
                
                {suggestion.action_details?.command && (
                  <div className="command-block">
                    <code>{suggestion.action_details.command}</code>
                    <button 
                      className="copy-btn"
                      onClick={() => copyToClipboard(suggestion.action_details.command, suggestion.id)}
                    >
                      {copiedCommand === suggestion.id ? '✓ Copied' : 'Copy'}
                    </button>
                  </div>
                )}
                
                {suggestion.action_details?.resolution && (
                  <div className="resolution-block">
                    <strong>Resolution:</strong>
                    <p>{suggestion.action_details.resolution}</p>
                  </div>
                )}
                
                <div className="suggestion-footer">
                  <span className="time-estimate">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M12 6v6l4 2"/>
                    </svg>
                    ~{suggestion.estimated_time} min
                  </span>
                  <span className="reasoning">{suggestion.reasoning}</span>
                </div>
              </div>
            ))}
            
            {(!assistance.suggestions || assistance.suggestions.length === 0) && (
              <div className="empty-state">
                <p>No suggestions available for this issue.</p>
              </div>
            )}
          </div>
        )}

        {/* Similar Cases Tab */}
        {activeTab === 'similar' && (
          <div className="similar-list">
            {assistance.similar_tickets?.map((ticket, idx) => (
              <div key={idx} className="similar-card">
                <div className="similar-header">
                  <span className="ticket-number">{ticket.ticket_number}</span>
                  <span className="similarity-score">
                    {Math.round(ticket.similarity_score * 100)}% match
                  </span>
                </div>
                <p className="description">{ticket.short_description}</p>
                {ticket.resolution && (
                  <div className="resolution">
                    <strong>Resolution:</strong>
                    <p>{ticket.resolution}</p>
                  </div>
                )}
                {ticket.resolution_time && (
                  <span className="resolution-time">Resolved in {ticket.resolution_time}</span>
                )}
              </div>
            ))}
            
            {(!assistance.similar_tickets || assistance.similar_tickets.length === 0) && (
              <div className="empty-state">
                <p>No similar cases found in the knowledge base.</p>
              </div>
            )}
          </div>
        )}

        {/* KB Articles Tab */}
        {activeTab === 'kb' && (
          <div className="kb-list">
            {assistance.kb_articles?.map((article, idx) => (
              <div key={idx} className="kb-card">
                <h4>{article.title}</h4>
                <p>{article.summary}</p>
                <div className="kb-footer">
                  <span className="relevance">
                    {Math.round(article.relevance_score * 100)}% relevant
                  </span>
                  {article.url && (
                    <a href={article.url} target="_blank" rel="noopener noreferrer">
                      View Article →
                    </a>
                  )}
                </div>
              </div>
            ))}
            
            {(!assistance.kb_articles || assistance.kb_articles.length === 0) && (
              <div className="empty-state">
                <p>No KB articles found for this issue.</p>
              </div>
            )}
          </div>
        )}

        {/* Quick Actions Tab */}
        {activeTab === 'actions' && (
          <div className="actions-grid">
            {assistance.quick_actions?.map((action, idx) => (
              <button 
                key={idx} 
                className={`action-btn ${executingAction === action.action_id ? 'executing' : ''}`}
                onClick={() => executeAction(action)}
                disabled={executingAction !== null}
              >
                <span className="action-icon">
                  {action.icon === 'user-check' && '👤'}
                  {action.icon === 'unlock' && '🔓'}
                  {action.icon === 'key' && '🔑'}
                  {action.icon === 'wifi' && '📶'}
                  {action.icon === 'refresh' && '🔄'}
                  {action.icon === 'mail' && '✉️'}
                  {action.icon === 'trash' && '🗑️'}
                  {!action.icon && '⚡'}
                </span>
                <span className="action-name">{action.name}</span>
                <span className="action-desc">{action.description}</span>
                {executingAction === action.action_id && (
                  <span className="action-loading">Running...</span>
                )}
              </button>
            ))}
            
            {(!assistance.quick_actions || assistance.quick_actions.length === 0) && (
              <div className="empty-state">
                <p>No quick actions available.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Escalation recommendation */}
      {assistance.escalation_recommended && assistance.escalation_team && (
        <div className="escalation-panel">
          <div className="escalation-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <div>
              <strong>Escalation Recommended</strong>
              <p>Consider escalating to: <strong>{assistance.escalation_team}</strong></p>
            </div>
            <button className="escalate-btn">Escalate Now</button>
          </div>
        </div>
      )}

      <style>{`
        .agent-assist-dashboard {
          background: var(--bg-primary, #fff);
          border-radius: 12px;
          border: 1px solid var(--border-color, #e5e7eb);
          overflow: hidden;
          max-height: 90vh;
          display: flex;
          flex-direction: column;
        }

        .agent-assist-dashboard.loading,
        .agent-assist-dashboard.error {
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .loading-spinner {
          text-align: center;
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid var(--border-color, #e5e7eb);
          border-top-color: var(--primary-color, #3b82f6);
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 16px;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .error-message {
          text-align: center;
          color: #ef4444;
        }

        .error-message svg {
          margin-bottom: 8px;
        }

        .error-message button {
          margin-top: 12px;
          padding: 8px 16px;
          background: var(--primary-color, #3b82f6);
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
        }

        .assist-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
          color: white;
        }

        .assist-title {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .assist-title h2 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .escalation-badge {
          background: #f97316;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 500;
        }

        .close-btn {
          background: rgba(255,255,255,0.2);
          border: none;
          color: white;
          width: 32px;
          height: 32px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 18px;
        }

        .close-btn:hover {
          background: rgba(255,255,255,0.3);
        }

        .assist-summary {
          display: flex;
          gap: 24px;
          padding: 12px 20px;
          background: var(--bg-secondary, #f9fafb);
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .summary-item {
          display: flex;
          gap: 8px;
        }

        .summary-item .label {
          color: var(--text-secondary, #6b7280);
          font-size: 13px;
        }

        .summary-item .value {
          font-weight: 500;
          font-size: 13px;
        }

        .assist-diagnosis {
          padding: 16px 20px;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .assist-diagnosis h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 8px;
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary, #111827);
        }

        .assist-diagnosis p {
          margin: 0;
          font-size: 14px;
          color: var(--text-secondary, #4b5563);
          line-height: 1.5;
        }

        .assist-tabs {
          display: flex;
          gap: 4px;
          padding: 12px 20px;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
          background: var(--bg-secondary, #f9fafb);
        }

        .assist-tabs button {
          padding: 8px 16px;
          border: none;
          background: transparent;
          color: var(--text-secondary, #6b7280);
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          border-radius: 6px;
          transition: all 0.2s;
        }

        .assist-tabs button:hover {
          background: var(--bg-primary, #fff);
          color: var(--text-primary, #111827);
        }

        .assist-tabs button.active {
          background: var(--primary-color, #3b82f6);
          color: white;
        }

        .assist-content {
          flex: 1;
          overflow-y: auto;
          padding: 16px 20px;
        }

        .suggestions-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .suggestion-card {
          background: var(--bg-secondary, #f9fafb);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
          padding: 16px;
        }

        .suggestion-header {
          display: flex;
          gap: 8px;
          align-items: center;
          margin-bottom: 8px;
        }

        .priority-badge,
        .type-badge {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .priority-badge {
          color: white;
        }

        .type-badge {
          background: var(--bg-primary, #fff);
          color: var(--text-secondary, #6b7280);
          border: 1px solid var(--border-color, #e5e7eb);
        }

        .suggestion-card h4 {
          margin: 0 0 8px;
          font-size: 14px;
          font-weight: 600;
        }

        .suggestion-card > p {
          margin: 0 0 12px;
          font-size: 13px;
          color: var(--text-secondary, #4b5563);
        }

        .command-block {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #1e293b;
          border-radius: 6px;
          padding: 8px 12px;
          margin-bottom: 12px;
        }

        .command-block code {
          flex: 1;
          color: #22c55e;
          font-family: 'Monaco', 'Menlo', monospace;
          font-size: 12px;
          overflow-x: auto;
        }

        .copy-btn {
          padding: 4px 10px;
          background: rgba(255,255,255,0.1);
          border: none;
          color: white;
          border-radius: 4px;
          font-size: 11px;
          cursor: pointer;
        }

        .copy-btn:hover {
          background: rgba(255,255,255,0.2);
        }

        .resolution-block {
          background: var(--bg-primary, #fff);
          border-radius: 6px;
          padding: 12px;
          margin-bottom: 12px;
        }

        .resolution-block strong {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .resolution-block p {
          margin: 4px 0 0;
          font-size: 13px;
        }

        .suggestion-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .time-estimate {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .similar-list,
        .kb-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .similar-card,
        .kb-card {
          background: var(--bg-secondary, #f9fafb);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
          padding: 16px;
        }

        .similar-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .ticket-number {
          font-weight: 600;
          color: var(--primary-color, #3b82f6);
        }

        .similarity-score {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
          background: var(--bg-primary, #fff);
          padding: 2px 8px;
          border-radius: 4px;
        }

        .kb-card h4 {
          margin: 0 0 8px;
          font-size: 14px;
          font-weight: 600;
        }

        .kb-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 12px;
        }

        .kb-footer a {
          color: var(--primary-color, #3b82f6);
          text-decoration: none;
          font-size: 13px;
          font-weight: 500;
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 12px;
        }

        .action-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 16px;
          background: var(--bg-secondary, #f9fafb);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
          cursor: pointer;
          text-align: left;
          transition: all 0.2s;
        }

        .action-btn:hover:not(:disabled) {
          border-color: var(--primary-color, #3b82f6);
          transform: translateY(-1px);
        }

        .action-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .action-btn.executing {
          border-color: var(--primary-color, #3b82f6);
          background: rgba(59, 130, 246, 0.05);
        }

        .action-icon {
          font-size: 24px;
          margin-bottom: 8px;
        }

        .action-name {
          font-weight: 600;
          font-size: 13px;
          margin-bottom: 4px;
        }

        .action-desc {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .action-loading {
          margin-top: 8px;
          font-size: 11px;
          color: var(--primary-color, #3b82f6);
        }

        .empty-state {
          text-align: center;
          padding: 32px;
          color: var(--text-secondary, #6b7280);
        }

        .escalation-panel {
          border-top: 1px solid var(--border-color, #e5e7eb);
          padding: 16px 20px;
          background: rgba(249, 115, 22, 0.05);
        }

        .escalation-content {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .escalation-content div {
          flex: 1;
        }

        .escalation-content strong {
          display: block;
          margin-bottom: 4px;
        }

        .escalation-content p {
          margin: 0;
          font-size: 13px;
          color: var(--text-secondary, #4b5563);
        }

        .escalate-btn {
          padding: 8px 16px;
          background: #f97316;
          color: white;
          border: none;
          border-radius: 6px;
          font-weight: 500;
          cursor: pointer;
        }

        .escalate-btn:hover {
          background: #ea580c;
        }
      `}</style>
    </div>
  );
}
