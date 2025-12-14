import React, { useState } from 'react'

export default function UserSettings({ visible, onClose }) {
  const [name, setName] = useState(() => {
    try { return localStorage.getItem('user_name') || 'User' } catch { return 'User' }
  })
  const [email, setEmail] = useState(() => {
    try { return localStorage.getItem('user_email') || 'user@example.com' } catch { return '' }
  })

  function save() {
    try {
      localStorage.setItem('user_name', name)
      localStorage.setItem('user_email', email)
    } catch {}
    onClose?.()
  }

  if (!visible) return null

  return (
    <div className="settings-modal" onClick={onClose}>
      <div className="settings-modal-content" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>User Settings</h3>
          <button 
            onClick={onClose}
            style={{
              width: '32px',
              height: '32px',
              border: 'none',
              background: '#F3F4F6',
              borderRadius: '50%',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            ✕
          </button>
        </div>

        <div className="settings-row">
          <label>Name</label>
          <input 
            type="text"
            value={name} 
            onChange={e => setName(e.target.value)}
          />
        </div>

        <div className="settings-row">
          <label>Email</label>
          <input 
            type="email"
            value={email} 
            onChange={e => setEmail(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px', paddingTop: '16px', borderTop: '1px solid #E5E7EB' }}>
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save}>Save Changes</button>
        </div>
      </div>
    </div>
  )
}
