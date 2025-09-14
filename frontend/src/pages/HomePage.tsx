import React, { useState, useEffect } from 'react'
import { Grid, Column, Tile, Button, InlineNotification } from '@carbon/react'
import { Add, View } from '@carbon/icons-react'
import { PromptInterface } from '@/components/prompt/PromptInterface'
import { WorkflowMonitor } from '@/components/dashboard/WorkflowMonitor'
import { useWorkflowAPI } from '@hooks/useWorkflowAPI'
import { useWebSocket } from '@hooks/useWebSocket'
import type { PromptData, WorkflowState } from '@types'
import './HomePage.scss'

export const HomePage: React.FC = () => {
  const [showPromptInterface, setShowPromptInterface] = useState(false)
  const [recentWorkflows, setRecentWorkflows] = useState<WorkflowState[]>([])
  const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(null)

  const { getWorkflows, getSystemMetrics } = useWorkflowAPI()
  const { isConnected, connectionState } = useWebSocket('ws://localhost:8000/ws', 'homepage-user')

  // Load recent workflows on component mount
  useEffect(() => {
    const loadRecentWorkflows = async () => {
      try {
        const response = await getWorkflows({ limit: 5, sortBy: 'startTime', sortOrder: 'desc' })
        if (response.success && response.data) {
          setRecentWorkflows(response.data)
        }
      } catch (error) {
        console.error('Failed to load recent workflows:', error)
      }
    }

    loadRecentWorkflows()
  }, [getWorkflows])

  const handlePromptSubmit = (promptData: PromptData) => {
    console.log('Prompt submitted:', promptData)
    // The PromptInterface will handle the actual submission
    setShowPromptInterface(false)
  }

  const handleWorkflowStart = (workflowId: string) => {
    console.log('Workflow started:', workflowId)
    setCurrentWorkflowId(workflowId)
    // Optionally refresh recent workflows
    setTimeout(async () => {
      const response = await getWorkflows({ limit: 5, sortBy: 'startTime', sortOrder: 'desc' })
      if (response.success && response.data) {
        setRecentWorkflows(response.data)
      }
    }, 1000)
  }

  const getConnectionStatusMessage = () => {
    switch (connectionState) {
      case 'connected':
        return { kind: 'success' as const, title: 'Connected', subtitle: 'Real-time updates active' }
      case 'connecting':
        return { kind: 'info' as const, title: 'Connecting', subtitle: 'Establishing connection...' }
      case 'error':
        return { kind: 'error' as const, title: 'Connection Error', subtitle: 'Failed to connect to server' }
      case 'disconnected':
        return { kind: 'warning' as const, title: 'Disconnected', subtitle: 'No real-time updates' }
      default:
        return null
    }
  }

  const connectionStatus = getConnectionStatusMessage()

  return (
    <div className="home-page">
      <div className="page-header">
        <div className="header-content">
          <h1>Project Maestro Dashboard</h1>
          <p className="header-subtitle">
            AI-powered game development orchestration platform
          </p>
        </div>
        
        <div className="header-actions">
          <Button
            kind="primary"
            renderIcon={Add}
            onClick={() => setShowPromptInterface(true)}
            disabled={connectionState === 'error'}
          >
            New Request
          </Button>
          <Button
            kind="secondary"
            renderIcon={View}
            onClick={() => {
              // Navigate to workflow page or scroll to workflow section
              const workflowSection = document.getElementById('workflow-section')
              workflowSection?.scrollIntoView({ behavior: 'smooth' })
            }}
          >
            View Workflows
          </Button>
        </div>
      </div>

      {/* Connection status notification */}
      {connectionStatus && connectionState !== 'connected' && (
        <div className="connection-notification">
          <InlineNotification
            kind={connectionStatus.kind}
            title={connectionStatus.title}
            subtitle={connectionStatus.subtitle}
            hideCloseButton
            lowContrast
          />
        </div>
      )}

      <Grid className="dashboard-grid">
        {/* Quick Stats */}
        <Column span={16} className="stats-section">
          <div className="stats-container">
            <Tile className="stat-tile">
              <div className="stat-content">
                <div className="stat-number">{recentWorkflows.length}</div>
                <div className="stat-label">Recent Workflows</div>
              </div>
            </Tile>
            <Tile className="stat-tile">
              <div className="stat-content">
                <div className="stat-number">
                  {recentWorkflows.filter(w => w.status === 'running').length}
                </div>
                <div className="stat-label">Active</div>
              </div>
            </Tile>
            <Tile className="stat-tile">
              <div className="stat-content">
                <div className="stat-number">
                  {recentWorkflows.filter(w => w.status === 'complete').length}
                </div>
                <div className="stat-label">Completed</div>
              </div>
            </Tile>
            <Tile className="stat-tile">
              <div className="stat-content">
                <div className="stat-number">6</div>
                <div className="stat-label">AI Agents</div>
              </div>
            </Tile>
          </div>
        </Column>

        {/* Main Content Area */}
        <Column span={16} className="main-content">
          {showPromptInterface ? (
            <div className="prompt-section">
              <div className="section-header">
                <h2>Create New Workflow</h2>
                <Button
                  kind="ghost"
                  onClick={() => setShowPromptInterface(false)}
                >
                  Cancel
                </Button>
              </div>
              
              <PromptInterface
                onSubmit={handlePromptSubmit}
                onWorkflowStart={handleWorkflowStart}
                className="homepage-prompt"
              />
            </div>
          ) : (
            <div className="welcome-section">
              <Tile className="welcome-tile">
                <div className="welcome-content">
                  <h2>Welcome to Project Maestro</h2>
                  <p>
                    Your AI-powered game development assistant. Create games, manage workflows, 
                    and orchestrate multiple AI agents to bring your ideas to life.
                  </p>
                  
                  <div className="feature-highlights">
                    <div className="feature-item">
                      <h4>🎮 Game Development</h4>
                      <p>Create complete games with AI assistance</p>
                    </div>
                    <div className="feature-item">
                      <h4>🤖 Multi-Agent System</h4>
                      <p>6 specialized AI agents working together</p>
                    </div>
                    <div className="feature-item">
                      <h4>📊 Real-time Monitoring</h4>
                      <p>Track progress and performance live</p>
                    </div>
                    <div className="feature-item">
                      <h4>⚡ Workflow Automation</h4>
                      <p>Automated task orchestration and management</p>
                    </div>
                  </div>

                  <div className="cta-section">
                    <Button
                      kind="primary"
                      size="lg"
                      renderIcon={Add}
                      onClick={() => setShowPromptInterface(true)}
                      disabled={connectionState === 'error'}
                    >
                      Start New Project
                    </Button>
                    <p className="cta-subtitle">
                      Describe your game idea and let our AI agents build it for you
                    </p>
                  </div>
                </div>
              </Tile>
            </div>
          )}
        </Column>

        {/* Workflow Monitor Section */}
        <Column span={16} className="workflow-section" id="workflow-section">
          <div className="section-header">
            <h2>Workflow Activity</h2>
            <div className="connection-indicator">
              {isConnected && (
                <div className="indicator-dot connected" title="Connected to real-time updates" />
              )}
              {!isConnected && (
                <div className="indicator-dot disconnected" title="Disconnected from real-time updates" />
              )}
            </div>
          </div>
          
          <WorkflowMonitor className="homepage-monitor" />
        </Column>
      </Grid>
    </div>
  )
}