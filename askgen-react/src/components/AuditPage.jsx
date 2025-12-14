import React from 'react'

export default function AuditPage() {
  const stats = [
    { label: 'Total Conversations', value: '1,247', icon: '💬', color: '#EBF5FF', textColor: '#1C64F2' },
    { label: 'Issues Resolved', value: '892', icon: '✓', color: '#D1FAE5', textColor: '#059669' },
    { label: 'Tickets Created', value: '355', icon: '🎫', color: '#FEF3C7', textColor: '#D97706' },
    { label: 'Avg Resolution Time', value: '4.2m', icon: '⏱️', color: '#E0E7FF', textColor: '#4F46E5' },
  ]

  const recentActivity = [
    { action: 'Password reset completed', agent: 'Troubleshooting Agent', time: '2 min ago', status: 'success' },
    { action: 'Ticket INC0010034 created', agent: 'Ticketing Agent', time: '15 min ago', status: 'info' },
    { action: 'VPN issue escalated', agent: 'Classification Agent', time: '32 min ago', status: 'warning' },
    { action: 'KB article suggested', agent: 'RAG Engine', time: '1 hour ago', status: 'success' },
    { action: 'New conversation started', agent: 'Orchestrator', time: '1 hour ago', status: 'info' },
  ]

  const agentPerformance = [
    { name: 'Troubleshooting Agent', calls: 523, success: 89 },
    { name: 'Ticketing Agent', calls: 355, success: 98 },
    { name: 'Classification Agent', calls: 892, success: 94 },
    { name: 'RAG Engine', calls: 1247, success: 87 },
  ]

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Analytics</h1>
        <p>Monitor AI assistant performance and usage</p>
      </div>

      <div className="stats-grid">
        {stats.map(stat => (
          <div className="stat-card" key={stat.label}>
            <div className="stat-icon" style={{ background: stat.color, color: stat.textColor }}>{stat.icon}</div>
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label">{stat.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent Activity</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {recentActivity.map((item, idx) => (
              <div key={idx} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                padding: '12px',
                background: '#F9FAFB',
                borderRadius: '8px'
              }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: item.status === 'success' ? '#10B981' : item.status === 'warning' ? '#F59E0B' : '#3B82F6'
                }}></div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 500, fontSize: '13px' }}>{item.action}</div>
                  <div style={{ fontSize: '12px', color: '#6B7280' }}>{item.agent}</div>
                </div>
                <div style={{ fontSize: '12px', color: '#9CA3AF' }}>{item.time}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Agent Performance</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {agentPerformance.map(agent => (
              <div key={agent.name}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 500 }}>{agent.name}</span>
                  <span style={{ fontSize: '12px', color: '#6B7280' }}>{agent.calls} calls • {agent.success}% success</span>
                </div>
                <div style={{ height: '8px', background: '#E5E7EB', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${agent.success}%`,
                    background: agent.success >= 90 ? '#10B981' : agent.success >= 80 ? '#3B82F6' : '#F59E0B',
                    borderRadius: '4px'
                  }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
