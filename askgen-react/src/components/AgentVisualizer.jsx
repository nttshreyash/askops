import React from 'react'

const agents = [
  { id: 'Orchestrator', label: 'Orchestrator', icon: '🎯' },
  { id: 'Greeting Agent', label: 'Greeting', icon: '👋' },
  { id: 'Classification Agent', label: 'Classify', icon: '🏷️' },
  { id: 'Clarify Agent', label: 'Clarify', icon: '❓' },
  { id: 'Troubleshooting Agent', label: 'Troubleshoot', icon: '🔧' },
  { id: 'Ticketing Agent', label: 'Ticketing', icon: '🎫' },
  { id: 'Status Check Agent', label: 'Status', icon: '📊' },
]

export default function AgentVisualizer({ active = 'Orchestrator' }) {
  return (
    <div className="agent-visualizer">
      {agents.map((agent, index) => (
        <React.Fragment key={agent.id}>
          <div className={`agent-node ${agent.id === active ? 'active' : ''}`}>
            <div className="agent-dot">
              {agent.icon}
            </div>
            <div className="agent-label">{agent.label}</div>
          </div>
          {index < agents.length - 1 && <div className="agent-connector" />}
        </React.Fragment>
      ))}
    </div>
  )
}
