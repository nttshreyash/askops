import React, { useState, useEffect } from 'react'

export default function Integrations() {
  const [integrationStatus, setIntegrationStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [testingConnection, setTestingConnection] = useState(null)

  // Determine backend URL
  const backendUrl = import.meta.env.VITE_BACKEND_CLOUD_URL || 
                     import.meta.env.VITE_BACKEND_URL || 
                     ''

  const fetchIntegrationStatus = async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch(`${backendUrl}/api/integrations/status`)
      if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`)
      const data = await resp.json()
      setIntegrationStatus(data)
    } catch (err) {
      setError(err.message)
      console.error('Failed to fetch integration status:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchIntegrationStatus()
  }, [])

  const testConnection = async (integrationKey) => {
    setTestingConnection(integrationKey)
    await fetchIntegrationStatus()
    setTestingConnection(null)
  }

  // Map backend integration keys to display info
  const integrationDisplayInfo = {
    servicenow: {
      name: 'ServiceNow',
      description: 'ITSM platform for ticket management and service catalog',
      icon: '🎫',
      color: '#D1FAE5',
    },
    azure_openai: {
      name: 'Azure OpenAI',
      description: 'AI language model for intelligent responses',
      icon: '🤖',
      color: '#E0E7FF',
    },
    rag_engine: {
      name: 'RAG Knowledge Base',
      description: 'Vector database for knowledge-enhanced responses',
      icon: '📚',
      color: '#FEF3C7',
    },
    itops_engine: {
      name: 'ITOps Engine',
      description: 'Auto-resolution, agent assist, and prescriptive analytics',
      icon: '⚡',
      color: '#DDD6FE',
    },
  }

  // Static integrations that aren't dynamically checked
  const staticIntegrations = [
    {
      key: 'active_directory',
      name: 'Active Directory',
      description: 'User authentication and directory services',
      icon: '👥',
      color: '#EBF5FF',
      status: 'connected',
      message: 'Integrated via ServiceNow',
    },
    {
      key: 'slack',
      name: 'Slack',
      description: 'Team communication and notifications',
      icon: '💬',
      color: '#FCE7F3',
      status: 'disconnected',
      message: 'Not configured',
    },
    {
      key: 'teams',
      name: 'Microsoft Teams',
      description: 'Enterprise collaboration platform',
      icon: '📺',
      color: '#E0E7FF',
      status: 'disconnected',
      message: 'Not configured',
    },
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return '#059669'
      case 'warning': return '#D97706'
      case 'error': return '#DC2626'
      default: return '#6B7280'
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'connected': return 'Connected'
      case 'warning': return 'Warning'
      case 'error': return 'Error'
      default: return 'Disconnected'
    }
  }

  // Build integration list from dynamic + static
  const buildIntegrations = () => {
    const items = []
    
    // Add dynamic integrations from backend
    if (integrationStatus?.integrations) {
      Object.entries(integrationStatus.integrations).forEach(([key, data]) => {
        const display = integrationDisplayInfo[key] || {
          name: key,
          description: 'Integration',
          icon: '🔌',
          color: '#E5E7EB',
        }
        items.push({
          key,
          ...display,
          status: data.status,
          message: data.message,
          details: data,
          isDynamic: true,
        })
      })
    }
    
    // Add static integrations
    staticIntegrations.forEach(int => {
      items.push({ ...int, isDynamic: false })
    })
    
    return items
  }

  const integrations = buildIntegrations()

  return (
    <div className="page-container">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>Integrations</h1>
            <p>Manage connected services and APIs</p>
          </div>
          <button 
            className="btn btn-primary" 
            onClick={fetchIntegrationStatus}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {loading ? (
              <>
                <span className="spinner-small"></span>
                Testing...
              </>
            ) : (
              <>
                🔄 Refresh Status
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ background: '#FEF2F2', borderColor: '#FECACA', marginBottom: '20px' }}>
          <div style={{ color: '#DC2626', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>⚠️</span>
            <span>Failed to check integration status: {error}</span>
          </div>
        </div>
      )}

      {integrationStatus && (
        <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <strong>Last Status Check:</strong>{' '}
              {new Date(integrationStatus.timestamp).toLocaleString()}
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '14px' }}>
              <span style={{ color: '#059669' }}>
                ✓ {Object.values(integrationStatus.integrations).filter(i => i.status === 'connected').length} Connected
              </span>
              <span style={{ color: '#DC2626' }}>
                ✗ {Object.values(integrationStatus.integrations).filter(i => i.status !== 'connected').length} Issues
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="integrations-grid">
        {integrations.map(integration => (
          <div className="integration-card" key={integration.key}>
            <div className="integration-icon" style={{ background: integration.color }}>
              {integration.icon}
            </div>
            <div className="integration-name">{integration.name}</div>
            <p style={{ fontSize: '13px', color: '#6B7280', marginTop: '4px', marginBottom: '12px' }}>
              {integration.description}
            </p>
            <div className="integration-status">
              <span 
                className="status-dot" 
                style={{ 
                  backgroundColor: getStatusColor(integration.status),
                  boxShadow: integration.status === 'connected' ? '0 0 8px rgba(5, 150, 105, 0.5)' : 'none'
                }}
              ></span>
              <span style={{ color: getStatusColor(integration.status), fontWeight: 500 }}>
                {getStatusText(integration.status)}
              </span>
            </div>
            <div style={{ fontSize: '12px', color: '#6B7280', marginTop: '4px', minHeight: '32px' }}>
              {integration.message}
              {integration.details?.instance && (
                <div style={{ marginTop: '4px', color: '#9CA3AF' }}>
                  Instance: {integration.details.instance}
                </div>
              )}
              {integration.details?.response_time_ms && (
                <div style={{ marginTop: '2px', color: '#9CA3AF' }}>
                  Response: {integration.details.response_time_ms}ms
                </div>
              )}
              {integration.details?.components && (
                <div style={{ marginTop: '4px', color: '#9CA3AF', fontSize: '11px' }}>
                  Components: {Object.keys(integration.details.components).filter(k => integration.details.components[k]).join(', ')}
                </div>
              )}
            </div>
            {integration.isDynamic ? (
              <button 
                className="btn btn-ghost"
                style={{ marginTop: '16px', width: '100%' }}
                onClick={() => testConnection(integration.key)}
                disabled={testingConnection === integration.key}
              >
                {testingConnection === integration.key ? 'Testing...' : 'Test Connection'}
              </button>
            ) : (
              <button 
                className={`btn ${integration.status === 'connected' ? 'btn-ghost' : 'btn-primary'}`}
                style={{ marginTop: '16px', width: '100%' }}
              >
                {integration.status === 'connected' ? 'Configure' : 'Connect'}
              </button>
            )}
          </div>
        ))}
      </div>

      <style>{`
        .spinner-small {
          width: 16px;
          height: 16px;
          border: 2px solid #fff;
          border-top-color: transparent;
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
