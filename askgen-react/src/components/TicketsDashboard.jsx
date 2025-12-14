import React, { useState, useEffect, useCallback } from 'react'

export default function TicketsDashboard() {
  const [tickets, setTickets] = useState([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [ticketType, setTicketType] = useState('all') // 'all', 'incidents', 'requests'

  // Determine backend URL
  const backendUrl = import.meta.env.VITE_BACKEND_CLOUD_URL || 
                     import.meta.env.VITE_BACKEND_URL || 
                     ''

  const fetchTickets = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch(`${backendUrl}/api/tickets?type=${ticketType}&limit=50`)
      if (!resp.ok) throw new Error(`Failed to fetch tickets: ${resp.status}`)
      const data = await resp.json()
      setTickets(data.tickets || [])
    } catch (err) {
      console.error('Failed to fetch tickets:', err)
      setError(err.message)
      // Set empty array on error
      setTickets([])
    } finally {
      setLoading(false)
    }
  }, [backendUrl, ticketType])

  useEffect(() => {
    fetchTickets()
  }, [fetchTickets])

  // Normalize status for filtering
  const normalizeStatus = (status) => {
    if (!status) return 'new'
    const s = status.toLowerCase()
    if (s.includes('new') || s === '1') return 'new'
    if (s.includes('progress') || s.includes('work') || s === '2') return 'in-progress'
    if (s.includes('resolved') || s === '6') return 'resolved'
    if (s.includes('closed') || s === '7') return 'closed'
    if (s.includes('pending') || s === '3') return 'pending'
    return 'new'
  }

  const stats = {
    total: tickets.length,
    open: tickets.filter(t => ['new', 'in-progress', 'pending'].includes(normalizeStatus(t.status || t.state))).length,
    resolved: tickets.filter(t => normalizeStatus(t.status || t.state) === 'resolved').length,
    closed: tickets.filter(t => normalizeStatus(t.status || t.state) === 'closed').length,
    incidents: tickets.filter(t => (t.number || '').startsWith('INC')).length,
    requests: tickets.filter(t => (t.number || '').startsWith('REQ') || (t.number || '').startsWith('RITM')).length
  }

  const filteredTickets = filter === 'all' 
    ? tickets 
    : tickets.filter(t => normalizeStatus(t.status || t.state) === filter)

  return (
    <div className="page-container">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>My Tickets</h1>
            <p>View and manage your support tickets from ServiceNow</p>
          </div>
          <button 
            className="btn btn-primary" 
            onClick={fetchTickets}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {loading ? (
              <>
                <span className="spinner-small"></span>
                Loading...
              </>
            ) : (
              <>
                🔄 Refresh
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ background: '#FEF2F2', borderColor: '#FECACA', marginBottom: '20px' }}>
          <div style={{ color: '#DC2626', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>⚠️</span>
            <span>Failed to load tickets: {error}</span>
          </div>
        </div>
      )}

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#EBF5FF', color: '#1C64F2' }}>📋</div>
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Tickets</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#FEF3C7', color: '#D97706' }}>⏳</div>
          <div className="stat-value">{stats.open}</div>
          <div className="stat-label">Open</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#D1FAE5', color: '#059669' }}>✓</div>
          <div className="stat-value">{stats.resolved}</div>
          <div className="stat-label">Resolved</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#E5E7EB', color: '#6B7280' }}>📁</div>
          <div className="stat-value">{stats.closed}</div>
          <div className="stat-label">Closed</div>
        </div>
      </div>

      {/* Ticket Type Toggle */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontWeight: 500, color: '#374151' }}>Ticket Type:</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[
              { key: 'all', label: 'All', count: stats.total },
              { key: 'incidents', label: 'Incidents', count: stats.incidents },
              { key: 'requests', label: 'Requests', count: stats.requests }
            ].map(t => (
              <button
                key={t.key}
                className={`tab-btn ${ticketType === t.key ? 'active' : ''}`}
                onClick={() => setTicketType(t.key)}
                style={{ padding: '6px 16px', fontSize: '13px' }}
              >
                {t.label} ({t.count})
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Tickets</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            {['all', 'new', 'in-progress', 'pending', 'resolved', 'closed'].map(f => (
              <button
                key={f}
                className={`tab-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
                style={{ padding: '6px 12px', fontSize: '12px' }}
              >
                {f === 'all' ? 'All' : f.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
              </button>
            ))}
          </div>
        </div>

        <div className="table-container">
          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#6B7280' }}>
              <div className="spinner" style={{ margin: '0 auto 16px' }}></div>
              Loading tickets from ServiceNow...
            </div>
          ) : filteredTickets.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#6B7280' }}>
              No tickets found
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticket #</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Priority</th>
                  <th>Category</th>
                  <th>Description</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {filteredTickets.map(ticket => (
                  <tr key={ticket.number || ticket.sys_id}>
                    <td style={{ fontWeight: 600, color: '#1C64F2' }}>{ticket.number}</td>
                    <td>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 500,
                        background: (ticket.number || '').startsWith('INC') ? '#FEE2E2' : '#DBEAFE',
                        color: (ticket.number || '').startsWith('INC') ? '#DC2626' : '#1D4ED8'
                      }}>
                        {(ticket.number || '').startsWith('INC') ? 'Incident' : 'Request'}
                      </span>
                    </td>
                    <td>
                      <span className={`status-badge ${normalizeStatus(ticket.status || ticket.state)}`}>
                        {ticket.status || ticket.state || 'New'}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        background: getPriorityColor(ticket.priority).bg,
                        color: getPriorityColor(ticket.priority).text
                      }}>
                        {getPriorityLabel(ticket.priority)}
                      </span>
                    </td>
                    <td>{ticket.category || ticket.u_category || '-'}</td>
                    <td style={{ maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {ticket.short_description || ticket.description || '-'}
                    </td>
                    <td style={{ color: '#6B7280', fontSize: '12px' }}>
                      {ticket.sys_created_on ? new Date(ticket.sys_created_on).toLocaleDateString() : ticket.created || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <style>{`
        .spinner-small {
          width: 16px;
          height: 16px;
          border: 2px solid #fff;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          display: inline-block;
        }
        .spinner {
          width: 32px;
          height: 32px;
          border: 3px solid #E5E7EB;
          border-top-color: #1C64F2;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

function getPriorityColor(priority) {
  const p = String(priority)
  const colors = {
    '1': { bg: '#FEE2E2', text: '#DC2626' },
    '2': { bg: '#FEF3C7', text: '#D97706' },
    '3': { bg: '#DBEAFE', text: '#1D4ED8' },
    '4': { bg: '#D1FAE5', text: '#059669' },
    '5': { bg: '#E5E7EB', text: '#6B7280' }
  }
  return colors[p] || colors['3']
}

function getPriorityLabel(priority) {
  const p = String(priority)
  const labels = { '1': 'Critical', '2': 'High', '3': 'Medium', '4': 'Low', '5': 'Planning' }
  return labels[p] || priority || 'Medium'
}
