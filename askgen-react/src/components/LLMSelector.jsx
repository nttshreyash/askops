import React, { useEffect, useState } from 'react'

export default function LLMSelector({ mode, setMode }) {
  const stored = (key, fallback) => {
    try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
  }

  const [localMode, setLocalMode] = useState(mode || stored('llm_mode', 'cloud'))
  const [cloudModel, setCloudModel] = useState(() => stored('cloud_model', 'azure-gpt'))
  const [temperature, setTemperature] = useState(() => parseFloat(stored('temperature', '0.5')))
  const [maxTokens, setMaxTokens] = useState(() => parseInt(stored('max_tokens', '1024')))
  const [streaming, setStreaming] = useState(() => stored('streaming', 'false') === 'true')
  const [status, setStatus] = useState(null)

  useEffect(() => { setLocalMode(mode) }, [mode])

  function applySettings() {
    try {
      localStorage.setItem('cloud_model', cloudModel)
      localStorage.setItem('temperature', String(temperature))
      localStorage.setItem('max_tokens', String(maxTokens))
      localStorage.setItem('streaming', String(streaming))
      localStorage.setItem('llm_mode', localMode)
      setMode?.(localMode)
      setStatus('saved')
      setTimeout(() => setStatus(null), 2000)
    } catch {
      setStatus('error')
      setTimeout(() => setStatus(null), 2000)
    }
  }

  return (
    <div className="settings-form">
      <div className="settings-header">
        <div>
          <h3>⚙️ AI Settings</h3>
          <p>Configure your AI assistant</p>
        </div>
        {status && (
          <span className="mode-badge" style={{
            background: status === 'saved' ? '#D1FAE5' : '#FEE2E2',
            color: status === 'saved' ? '#059669' : '#DC2626'
          }}>
            {status === 'saved' ? '✓ Saved' : '✗ Error'}
          </span>
        )}
      </div>

      <div className="settings-row">
        <label>Deployment Mode</label>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`tab-btn ${localMode === 'cloud' ? 'active' : ''}`}
            onClick={() => setLocalMode('cloud')}
          >
            ☁️ Cloud
          </button>
          <button
            className={`tab-btn ${localMode === 'private' ? 'active' : ''}`}
            onClick={() => setLocalMode('private')}
          >
            🔒 Private
          </button>
        </div>
        <span className="helper-text">
          {localMode === 'cloud' ? 'Using hosted cloud APIs' : 'Using on-premises model'}
        </span>
      </div>

      <div className="settings-row">
        <label>Cloud Model</label>
        <select value={cloudModel} onChange={e => setCloudModel(e.target.value)}>
          <option value="azure-gpt">Azure OpenAI (GPT-4)</option>
          <option value="google-vertex">Google Vertex AI</option>
          <option value="openai-gpt">OpenAI Direct</option>
        </select>
      </div>

      <div className="settings-row">
        <label style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Temperature</span>
          <span style={{ color: '#1C64F2', fontWeight: 600 }}>{temperature.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={temperature}
          onChange={e => setTemperature(parseFloat(e.target.value))}
        />
        <span className="helper-text">Lower = more focused, Higher = more creative</span>
      </div>

      <div className="settings-row">
        <label>Max Response Tokens</label>
        <input
          type="number"
          value={maxTokens}
          onChange={e => setMaxTokens(parseInt(e.target.value) || 1024)}
          min={16}
          max={65536}
        />
        <span className="helper-text">Recommended: 512–4096</span>
      </div>

      <div className="settings-row">
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={streaming}
            onChange={e => setStreaming(e.target.checked)}
            style={{ width: '16px', height: '16px' }}
          />
          <span>Enable Streaming</span>
        </label>
        <span className="helper-text">Show responses as they're generated</span>
      </div>

      <div style={{ display: 'flex', gap: '8px', marginTop: '16px', paddingTop: '16px', borderTop: '1px solid #E5E7EB' }}>
        <button className="btn btn-primary" style={{ flex: 1 }} onClick={applySettings}>
          ✓ Apply Settings
        </button>
      </div>
    </div>
  )
}
