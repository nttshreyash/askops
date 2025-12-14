import React from 'react';

/**
 * Intelligent Triage Card
 * 
 * Displays the result of AI-powered ticket triage including:
 * - Auto-resolution status
 * - Prescriptive steps for agents
 * - Similar cases
 * - Routing recommendations
 */
export default function IntelligentTriageCard({ triageResult, onViewDetails }) {
  if (!triageResult) return null;

  const {
    decision,
    priority,
    confidence,
    reasoning,
    auto_resolution_attempted,
    auto_resolution_successful,
    prescriptive_steps,
    recommended_team,
    estimated_resolution_time,
    customer_message,
    similar_cases,
    kb_articles
  } = triageResult;

  // Decision badge styling
  const getDecisionStyle = () => {
    switch (decision) {
      case 'auto_resolve':
        return { bg: '#22c55e', icon: '✓' };
      case 'self_service':
        return { bg: '#3b82f6', icon: '🔧' };
      case 'assign_tier1':
        return { bg: '#8b5cf6', icon: '👤' };
      case 'assign_tier2':
        return { bg: '#f97316', icon: '👥' };
      case 'escalate_immediate':
        return { bg: '#ef4444', icon: '🚨' };
      default:
        return { bg: '#6b7280', icon: '📋' };
    }
  };

  const decisionStyle = getDecisionStyle();

  // Priority color
  const getPriorityColor = () => {
    if (priority.includes('critical') || priority.includes('p1')) return '#ef4444';
    if (priority.includes('high') || priority.includes('p2')) return '#f97316';
    if (priority.includes('medium') || priority.includes('p3')) return '#eab308';
    return '#22c55e';
  };

  return (
    <div className="intelligent-triage-card">
      {/* Header */}
      <div className="triage-header">
        <div className="triage-badge" style={{ backgroundColor: decisionStyle.bg }}>
          <span className="badge-icon">{decisionStyle.icon}</span>
          <span className="badge-text">
            {decision.replace(/_/g, ' ').toUpperCase()}
          </span>
        </div>
        <span className="confidence-indicator">
          {Math.round(confidence * 100)}% confidence
        </span>
      </div>

      {/* Auto Resolution Status */}
      {auto_resolution_attempted && (
        <div className={`auto-resolve-status ${auto_resolution_successful ? 'success' : 'failed'}`}>
          {auto_resolution_successful ? (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
                <path d="M22 4L12 14.01l-3-3"/>
              </svg>
              <span>Issue automatically resolved!</span>
            </>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f97316" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
              </svg>
              <span>Auto-resolution attempted - prescriptive steps provided</span>
            </>
          )}
        </div>
      )}

      {/* Customer Message */}
      <div className="customer-message">
        <p>{customer_message}</p>
      </div>

      {/* Key Info */}
      <div className="triage-info">
        <div className="info-item">
          <span className="info-label">Priority</span>
          <span className="info-value priority-value" style={{ color: getPriorityColor() }}>
            {priority.replace(/_/g, ' ').toUpperCase()}
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Est. Time</span>
          <span className="info-value">{estimated_resolution_time} min</span>
        </div>
        {recommended_team && (
          <div className="info-item">
            <span className="info-label">Assigned To</span>
            <span className="info-value">{recommended_team}</span>
          </div>
        )}
      </div>

      {/* Prescriptive Steps */}
      {prescriptive_steps && prescriptive_steps.length > 0 && (
        <div className="prescriptive-section">
          <h4>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
              <path d="M14 2v6h6"/>
              <path d="M12 18v-6"/>
              <path d="M9 15l3 3 3-3"/>
            </svg>
            Resolution Steps
          </h4>
          <div className="steps-list">
            {prescriptive_steps.slice(0, 3).map((step, idx) => (
              <div key={idx} className="step-item">
                <span className="step-number">{step.step_number || idx + 1}</span>
                <div className="step-content">
                  <span className="step-action">{step.action}</span>
                  {step.automated && (
                    <span className="automated-badge">🤖 Automated</span>
                  )}
                </div>
              </div>
            ))}
            {prescriptive_steps.length > 3 && (
              <div className="more-steps">
                +{prescriptive_steps.length - 3} more steps
              </div>
            )}
          </div>
        </div>
      )}

      {/* Similar Cases */}
      {similar_cases && similar_cases.length > 0 && (
        <div className="similar-section">
          <h4>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
            </svg>
            Similar Resolved Cases
          </h4>
          <div className="similar-cases">
            {similar_cases.slice(0, 2).map((ticket, idx) => (
              <div key={idx} className="similar-case">
                <span className="case-id">{ticket.ticket_id}</span>
                <span className="case-similarity">
                  {Math.round(ticket.similarity * 100)}% match
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* View Details Button */}
      {onViewDetails && (
        <button className="view-details-btn" onClick={onViewDetails}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
          </svg>
          View Agent Assist Dashboard
        </button>
      )}

      <style>{`
        .intelligent-triage-card {
          background: var(--bg-primary, #fff);
          border-radius: 12px;
          border: 1px solid var(--border-color, #e5e7eb);
          overflow: hidden;
          margin: 8px 0;
        }

        .triage-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          background: var(--bg-secondary, #f9fafb);
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .triage-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 4px 12px;
          border-radius: 6px;
          color: white;
          font-size: 12px;
          font-weight: 600;
        }

        .badge-icon {
          font-size: 14px;
        }

        .confidence-indicator {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
          background: var(--bg-primary, #fff);
          padding: 4px 8px;
          border-radius: 4px;
        }

        .auto-resolve-status {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          font-size: 13px;
          font-weight: 500;
        }

        .auto-resolve-status.success {
          background: rgba(34, 197, 94, 0.1);
          color: #15803d;
        }

        .auto-resolve-status.failed {
          background: rgba(249, 115, 22, 0.1);
          color: #c2410c;
        }

        .customer-message {
          padding: 16px;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .customer-message p {
          margin: 0;
          font-size: 14px;
          color: var(--text-primary, #111827);
          line-height: 1.5;
        }

        .triage-info {
          display: flex;
          gap: 24px;
          padding: 12px 16px;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
          background: var(--bg-secondary, #f9fafb);
        }

        .info-item {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .info-label {
          font-size: 11px;
          color: var(--text-secondary, #6b7280);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .info-value {
          font-size: 13px;
          font-weight: 600;
        }

        .priority-value {
          text-transform: uppercase;
        }

        .prescriptive-section,
        .similar-section {
          padding: 12px 16px;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .prescriptive-section h4,
        .similar-section h4 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 12px;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-primary, #111827);
        }

        .steps-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .step-item {
          display: flex;
          align-items: flex-start;
          gap: 12px;
        }

        .step-number {
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--primary-color, #3b82f6);
          color: white;
          border-radius: 50%;
          font-size: 12px;
          font-weight: 600;
          flex-shrink: 0;
        }

        .step-content {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .step-action {
          font-size: 13px;
          color: var(--text-primary, #111827);
        }

        .automated-badge {
          font-size: 11px;
          color: var(--primary-color, #3b82f6);
        }

        .more-steps {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
          margin-left: 36px;
        }

        .similar-cases {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .similar-case {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: var(--bg-secondary, #f9fafb);
          border-radius: 6px;
        }

        .case-id {
          font-size: 13px;
          font-weight: 500;
          color: var(--primary-color, #3b82f6);
        }

        .case-similarity {
          font-size: 12px;
          color: var(--text-secondary, #6b7280);
        }

        .view-details-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px;
          background: var(--primary-color, #3b82f6);
          color: white;
          border: none;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: background 0.2s;
        }

        .view-details-btn:hover {
          background: #2563eb;
        }
      `}</style>
    </div>
  );
}
