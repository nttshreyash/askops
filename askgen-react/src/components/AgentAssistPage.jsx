import React, { useState, useEffect } from 'react';
import AgentAssistDashboard from './AgentAssistDashboard';

/**
 * Agent Assist Page
 * 
 * Full-page view for IT agents to:
 * - Enter ticket details for AI analysis
 * - Get prescriptive guidance
 * - View runbooks
 * - Access diagnostic commands
 */
export default function AgentAssistPage() {
  const [ticketId, setTicketId] = useState('');
  const [issueDescription, setIssueDescription] = useState('');
  const [showAssist, setShowAssist] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [runbooks, setRunbooks] = useState([]);
  const [loadingMetrics, setLoadingMetrics] = useState(true);

  // Fetch ITOps metrics on load
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch('/api/itops/metrics');
        if (res.ok) {
          const data = await res.json();
          setMetrics(data.metrics);
        }
      } catch (e) {
        console.log('Metrics not available:', e.message);
      } finally {
        setLoadingMetrics(false);
      }
    };

    const fetchRunbooks = async () => {
      try {
        const res = await fetch('/api/itops/runbooks');
        if (res.ok) {
          const data = await res.json();
          setRunbooks(data.runbooks || []);
        }
      } catch (e) {
        console.log('Runbooks not available:', e.message);
      }
    };

    fetchMetrics();
    fetchRunbooks();
  }, []);

  const handleAnalyze = () => {
    if (issueDescription.trim()) {
      setShowAssist(true);
    }
  };

  return (
    <div className="agent-assist-page">
      <div className="page-header">
        <div className="header-content">
          <h1>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
            AI Agent Assist
          </h1>
          <p>Intelligent guidance for IT support agents</p>
        </div>
        
        <div className="header-stats">
          {loadingMetrics ? (
            <span>Loading metrics...</span>
          ) : metrics ? (
            <>
              <div className="stat-item">
                <span className="stat-value">{metrics.triage?.total_triaged || 0}</span>
                <span className="stat-label">Tickets Triaged</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{metrics.triage?.auto_resolution_rate || 'N/A'}</span>
                <span className="stat-label">Auto-Resolution Rate</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{metrics.agent_assist?.total_sessions || 0}</span>
                <span className="stat-label">Assist Sessions</span>
              </div>
            </>
          ) : null}
        </div>
      </div>

      {!showAssist ? (
        <div className="assist-input-section">
          <div className="input-card">
            <h2>Analyze a Ticket</h2>
            <p>Enter ticket details to get AI-powered assistance</p>
            
            <div className="form-group">
              <label>Ticket ID (optional)</label>
              <input
                type="text"
                placeholder="INC0012345"
                value={ticketId}
                onChange={(e) => setTicketId(e.target.value)}
              />
            </div>
            
            <div className="form-group">
              <label>Issue Description</label>
              <textarea
                placeholder="Describe the issue in detail..."
                value={issueDescription}
                onChange={(e) => setIssueDescription(e.target.value)}
                rows={5}
              />
            </div>
            
            <button 
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={!issueDescription.trim()}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35"/>
              </svg>
              Analyze & Get Assistance
            </button>
          </div>

          {/* Quick Templates */}
          <div className="templates-section">
            <h3>Quick Templates</h3>
            <div className="template-grid">
              {[
                { label: 'Password Reset', desc: 'User forgot password' },
                { label: 'VPN Issue', desc: 'VPN not connecting' },
                { label: 'Account Locked', desc: 'Too many login attempts' },
                { label: 'Email Problem', desc: 'Outlook not syncing' },
                { label: 'Software Install', desc: 'Need software installed' },
                { label: 'Network Issue', desc: 'Internet not working' }
              ].map(template => (
                <button 
                  key={template.label}
                  className="template-btn"
                  onClick={() => setIssueDescription(template.desc)}
                >
                  <span className="template-label">{template.label}</span>
                  <span className="template-desc">{template.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Available Runbooks */}
          {runbooks.length > 0 && (
            <div className="runbooks-section">
              <h3>Available Runbooks</h3>
              <div className="runbook-grid">
                {runbooks.slice(0, 6).map(rb => (
                  <div key={rb.id} className="runbook-card">
                    <div className="runbook-header">
                      <span className="runbook-name">{rb.name}</span>
                      <span className={`risk-badge ${rb.risk_level}`}>{rb.risk_level}</span>
                    </div>
                    <p className="runbook-desc">{rb.description}</p>
                    <div className="runbook-footer">
                      <span>~{rb.estimated_duration} min</span>
                      <span>{rb.success_rate} success rate</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="assist-result-section">
          <button className="back-btn" onClick={() => setShowAssist(false)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5"/>
              <path d="M12 19l-7-7 7-7"/>
            </svg>
            Back to Input
          </button>
          
          <AgentAssistDashboard
            ticketId={ticketId}
            issueDescription={issueDescription}
            onClose={() => setShowAssist(false)}
          />
        </div>
      )}

      <style>{`
        .agent-assist-page {
          padding: 24px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 32px;
          flex-wrap: wrap;
          gap: 16px;
        }

        .header-content h1 {
          display: flex;
          align-items: center;
          gap: 12px;
          margin: 0 0 8px;
          font-size: 24px;
          font-weight: 700;
        }

        .header-content p {
          margin: 0;
          color: var(--text-secondary, #6b7280);
        }

        .header-stats {
          display: flex;
          gap: 24px;
        }

        .stat-item {
          text-align: center;
          padding: 12px 20px;
          background: var(--bg-secondary, #f9fafb);
          border-radius: 8px;
        }

        .stat-value {
          display: block;
          font-size: 24px;
          font-weight: 700;
          color: var(--primary-color, #3b82f6);
        }

        .stat-label {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .assist-input-section {
          display: grid;
          gap: 24px;
        }

        .input-card {
          background: var(--bg-primary, #fff);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 12px;
          padding: 24px;
        }

        .input-card h2 {
          margin: 0 0 4px;
          font-size: 18px;
        }

        .input-card > p {
          margin: 0 0 20px;
          color: var(--text-secondary, #6b7280);
        }

        .form-group {
          margin-bottom: 16px;
        }

        .form-group label {
          display: block;
          margin-bottom: 6px;
          font-weight: 500;
          font-size: 14px;
        }

        .form-group input,
        .form-group textarea {
          width: 100%;
          padding: 12px;
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
          font-size: 14px;
          resize: vertical;
        }

        .form-group input:focus,
        .form-group textarea:focus {
          outline: none;
          border-color: var(--primary-color, #3b82f6);
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .analyze-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          width: 100%;
          padding: 14px;
          background: linear-gradient(135deg, #3b82f6, #1d4ed8);
          color: white;
          border: none;
          border-radius: 8px;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .analyze-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .analyze-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .templates-section,
        .runbooks-section {
          background: var(--bg-primary, #fff);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 12px;
          padding: 24px;
        }

        .templates-section h3,
        .runbooks-section h3 {
          margin: 0 0 16px;
          font-size: 16px;
        }

        .template-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 12px;
        }

        .template-btn {
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

        .template-btn:hover {
          border-color: var(--primary-color, #3b82f6);
          transform: translateY(-2px);
        }

        .template-label {
          font-weight: 600;
          font-size: 13px;
          margin-bottom: 4px;
        }

        .template-desc {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .runbook-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 16px;
        }

        .runbook-card {
          padding: 16px;
          background: var(--bg-secondary, #f9fafb);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
        }

        .runbook-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .runbook-name {
          font-weight: 600;
          font-size: 14px;
        }

        .risk-badge {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .risk-badge.low {
          background: #dcfce7;
          color: #166534;
        }

        .risk-badge.medium {
          background: #fef3c7;
          color: #92400e;
        }

        .risk-badge.high {
          background: #fee2e2;
          color: #991b1b;
        }

        .runbook-desc {
          margin: 0 0 12px;
          font-size: 13px;
          color: var(--text-secondary, #6b7280);
        }

        .runbook-footer {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .assist-result-section {
          position: relative;
        }

        .back-btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          margin-bottom: 16px;
          background: var(--bg-secondary, #f9fafb);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 6px;
          cursor: pointer;
          font-size: 13px;
          font-weight: 500;
        }

        .back-btn:hover {
          background: var(--bg-primary, #fff);
        }

        @media (max-width: 768px) {
          .page-header {
            flex-direction: column;
          }

          .header-stats {
            width: 100%;
            justify-content: space-between;
          }

          .template-grid {
            grid-template-columns: 1fr 1fr;
          }
        }
      `}</style>
    </div>
  );
}
