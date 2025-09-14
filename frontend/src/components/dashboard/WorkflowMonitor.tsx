import React, { useState, useEffect, useMemo } from 'react'
import {
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  TableSelectAll,
  TableSelectRow,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
  ProgressBar,
  Tag,
  Button,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Search,
  Dropdown,
  Loading,
  InlineNotification,
  Tile,
  Grid,
  Column,
} from '@carbon/react'
import { 
  Play,
  Pause,
  Stop,
  Renew,
  View,
  Download,
  Filter,
} from '@carbon/icons-react'
import { useWebSocket } from '@hooks/useWebSocket'
import { useWorkflowAPI } from '@hooks/useWorkflowAPI'
import { AgentFlowDiagram } from '@/components/visualization/AgentFlowDiagram'
import { maestroCustomColors } from '@/setup/carbon-theme'
import type { WorkflowState, LogEntry, Agent, WebSocketMessage, FilterOptions } from '@types'
import './WorkflowMonitor.scss'

interface WorkflowMonitorProps {
  className?: string
}

export const WorkflowMonitor: React.FC<WorkflowMonitorProps> = ({
  className = '',
}) => {
  const [workflows, setWorkflows] = useState<WorkflowState[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null)
  const [realtimeLogs, setRealtimeLogs] = useState<LogEntry[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedRows, setSelectedRows] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState(0)

  const { 
    getWorkflows, 
    pauseWorkflow, 
    resumeWorkflow, 
    cancelWorkflow,
    loading: apiLoading 
  } = useWorkflowAPI()

  const { messages, isConnected } = useWebSocket(
    'ws://localhost:8000/ws',
    'monitor-user'
  )

  // Handle WebSocket messages
  useEffect(() => {
    const handleMessage = (msg: WebSocketMessage) => {
      switch (msg.type) {
        case 'workflow_update':
          updateWorkflow(msg.data)
          break
        case 'log_entry':
          addLogEntry(msg.data)
          break
        case 'agent_status':
          updateAgentStatus(msg.data)
          break
      }
    }

    messages.forEach(handleMessage)
  }, [messages])

  // Load initial workflows
  useEffect(() => {
    const loadWorkflows = async () => {
      const response = await getWorkflows()
      if (response.success && response.data) {
        setWorkflows(response.data)
      }
    }

    loadWorkflows()
    // Refresh every 30 seconds
    const interval = setInterval(loadWorkflows, 30000)
    return () => clearInterval(interval)
  }, [getWorkflows])

  const updateWorkflow = (data: Partial<WorkflowState> & { id: string }) => {
    setWorkflows(prev => {
      const index = prev.findIndex(w => w.id === data.id)
      if (index >= 0) {
        const updated = [...prev]
        updated[index] = { ...updated[index], ...data }
        return updated
      }
      return [...prev, data as WorkflowState]
    })
  }

  const addLogEntry = (entry: LogEntry) => {
    setRealtimeLogs(prev => {
      const newLogs = [...prev, entry].slice(-500) // Keep last 500 logs
      return newLogs.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
    })
  }

  const updateAgentStatus = (data: { workflowId: string; agentId: string; status: Agent['status']; progress?: number }) => {
    setWorkflows(prev => prev.map(workflow => {
      if (workflow.id === data.workflowId) {
        return {
          ...workflow,
          agents: workflow.agents.map(agent => 
            agent.id === data.agentId
              ? { ...agent, status: data.status, progress: data.progress || agent.progress }
              : agent
          )
        }
      }
      return workflow
    }))
  }

  // Filtered and searched workflows
  const filteredWorkflows = useMemo(() => {
    return workflows.filter(workflow => {
      const matchesSearch = !searchQuery || 
        workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        workflow.description?.toLowerCase().includes(searchQuery.toLowerCase())
      
      const matchesStatus = statusFilter === 'all' || workflow.status === statusFilter
      
      return matchesSearch && matchesStatus
    })
  }, [workflows, searchQuery, statusFilter])

  const selectedWorkflowData = useMemo(() => {
    return workflows.find(w => w.id === selectedWorkflow)
  }, [workflows, selectedWorkflow])

  // Workflow statistics
  const workflowStats = useMemo(() => {
    const total = workflows.length
    const running = workflows.filter(w => w.status === 'running').length
    const completed = workflows.filter(w => w.status === 'complete').length
    const failed = workflows.filter(w => w.status === 'error').length
    const paused = workflows.filter(w => w.status === 'paused').length
    
    return { total, running, completed, failed, paused }
  }, [workflows])

  // Table headers
  const tableHeaders = [
    { key: 'name', header: 'Workflow Name' },
    { key: 'status', header: 'Status' },
    { key: 'progress', header: 'Progress' },
    { key: 'agents', header: 'Agents' },
    { key: 'startTime', header: 'Started' },
    { key: 'duration', header: 'Duration' },
    { key: 'actions', header: 'Actions' },
  ]

  // Table rows
  const tableRows = filteredWorkflows.map(workflow => ({
    id: workflow.id,
    name: workflow.name,
    status: workflow.status,
    progress: workflow.progress,
    agents: workflow.agents.length,
    startTime: new Date(workflow.startTime).toLocaleString(),
    duration: workflow.endTime 
      ? `${Math.round((new Date(workflow.endTime).getTime() - new Date(workflow.startTime).getTime()) / 1000)}s`
      : workflow.status === 'running' 
        ? `${Math.round((Date.now() - new Date(workflow.startTime).getTime()) / 1000)}s`
        : '-',
    actions: workflow.id,
  }))

  const getStatusTagProps = (status: WorkflowState['status']) => {
    switch (status) {
      case 'running': return { type: 'blue' as const, text: 'Running' }
      case 'complete': return { type: 'green' as const, text: 'Complete' }
      case 'error': return { type: 'red' as const, text: 'Error' }
      case 'paused': return { type: 'gray' as const, text: 'Paused' }
      default: return { type: 'gray' as const, text: 'Idle' }
    }
  }

  const handleWorkflowAction = async (action: string, workflowId: string) => {
    try {
      switch (action) {
        case 'pause':
          await pauseWorkflow(workflowId)
          break
        case 'resume':
          await resumeWorkflow(workflowId)
          break
        case 'cancel':
          await cancelWorkflow(workflowId)
          break
      }
    } catch (error) {
      console.error(`Failed to ${action} workflow:`, error)
    }
  }

  const handleRowSelection = (selectedRows: readonly string[]) => {
    setSelectedRows([...selectedRows])
  }

  const statusFilterOptions = [
    { id: 'all', text: 'All Workflows' },
    { id: 'running', text: 'Running' },
    { id: 'complete', text: 'Complete' },
    { id: 'error', text: 'Error' },
    { id: 'paused', text: 'Paused' },
    { id: 'idle', text: 'Idle' },
  ]

  return (
    <div className={`workflow-monitor ${className}`}>
      {/* Header with statistics */}
      <div className="monitor-header">
        <div className="header-content">
          <h2>Workflow Monitor</h2>
          <div className="connection-status">
            <Tag type={isConnected ? 'green' : 'red'} size="sm">
              {isConnected ? 'Connected' : 'Offline'}
            </Tag>
          </div>
        </div>
        
        <Grid className="stats-grid">
          <Column sm={4} md={2} lg={2}>
            <Tile className="stat-tile">
              <div className="stat-number">{workflowStats.total}</div>
              <div className="stat-label">Total</div>
            </Tile>
          </Column>
          <Column sm={4} md={2} lg={2}>
            <Tile className="stat-tile stat-running">
              <div className="stat-number">{workflowStats.running}</div>
              <div className="stat-label">Running</div>
            </Tile>
          </Column>
          <Column sm={4} md={2} lg={2}>
            <Tile className="stat-tile stat-completed">
              <div className="stat-number">{workflowStats.completed}</div>
              <div className="stat-label">Complete</div>
            </Tile>
          </Column>
          <Column sm={4} md={2} lg={2}>
            <Tile className="stat-tile stat-failed">
              <div className="stat-number">{workflowStats.failed}</div>
              <div className="stat-label">Failed</div>
            </Tile>
          </Column>
          <Column sm={4} md={2} lg={2}>
            <Tile className="stat-tile">
              <div className="stat-number">{workflowStats.paused}</div>
              <div className="stat-label">Paused</div>
            </Tile>
          </Column>
        </Grid>
      </div>

      {/* Controls */}
      <div className="monitor-controls">
        <Search
          size="lg"
          placeholder="Search workflows..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="workflow-search"
        />
        
        <Dropdown
          id="status-filter"
          titleText=""
          label="Filter by status"
          items={statusFilterOptions}
          selectedItem={statusFilterOptions.find(opt => opt.id === statusFilter)}
          onChange={({ selectedItem }) => setStatusFilter(selectedItem?.id || 'all')}
        />

        <Button
          kind="secondary"
          renderIcon={Renew}
          onClick={() => {
            // Refresh workflows
            getWorkflows()
          }}
          disabled={apiLoading}
        >
          Refresh
        </Button>
      </div>

      {/* Main content with tabs */}
      <Tabs selectedIndex={activeTab} onChange={({ selectedIndex }) => setActiveTab(selectedIndex)}>
        <TabList aria-label="Workflow monitor tabs">
          <Tab>Workflows</Tab>
          <Tab disabled={!selectedWorkflowData}>Workflow Details</Tab>
          <Tab>Live Logs</Tab>
        </TabList>
        
        <TabPanels>
          {/* Workflows Table */}
          <TabPanel>
            <div className="workflows-table-container">
              <DataTable
                rows={tableRows}
                headers={tableHeaders}
                radio={false}
                isSortable
                useZebraStyles
                onSelectionChange={handleRowSelection}
              >
                {({ rows, headers, getHeaderProps, getRowProps, getSelectionProps, getTableProps, getTableContainerProps }) => (
                  <TableContainer {...getTableContainerProps()}>
                    <Table {...getTableProps()}>
                      <TableHead>
                        <TableRow>
                          <TableSelectAll {...getSelectionProps()} />
                          {headers.map((header) => (
                            <TableHeader key={header.key} {...getHeaderProps({ header })}>
                              {header.header}
                            </TableHeader>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {rows.map((row) => (
                          <TableRow 
                            key={row.id} 
                            {...getRowProps({ row })}
                            className={selectedWorkflow === row.id ? 'selected-row' : ''}
                            onClick={() => setSelectedWorkflow(row.id)}
                          >
                            <TableSelectRow {...getSelectionProps({ row })} />
                            {row.cells.map((cell) => {
                              if (cell.info.header === 'status') {
                                const statusProps = getStatusTagProps(cell.value)
                                return (
                                  <TableCell key={cell.id}>
                                    <Tag type={statusProps.type} size="sm">
                                      {statusProps.text}
                                    </Tag>
                                  </TableCell>
                                )
                              } else if (cell.info.header === 'progress') {
                                return (
                                  <TableCell key={cell.id}>
                                    <ProgressBar value={cell.value} size="sm" />
                                  </TableCell>
                                )
                              } else if (cell.info.header === 'actions') {
                                const workflow = workflows.find(w => w.id === cell.value)
                                return (
                                  <TableCell key={cell.id}>
                                    <div className="action-buttons">
                                      {workflow?.status === 'running' && (
                                        <Button
                                          kind="ghost"
                                          size="sm"
                                          hasIconOnly
                                          iconDescription="Pause"
                                          renderIcon={Pause}
                                          onClick={(e) => {
                                            e.stopPropagation()
                                            handleWorkflowAction('pause', cell.value)
                                          }}
                                        />
                                      )}
                                      {workflow?.status === 'paused' && (
                                        <Button
                                          kind="ghost"
                                          size="sm"
                                          hasIconOnly
                                          iconDescription="Resume"
                                          renderIcon={Play}
                                          onClick={(e) => {
                                            e.stopPropagation()
                                            handleWorkflowAction('resume', cell.value)
                                          }}
                                        />
                                      )}
                                      {(workflow?.status === 'running' || workflow?.status === 'paused') && (
                                        <Button
                                          kind="ghost"
                                          size="sm"
                                          hasIconOnly
                                          iconDescription="Cancel"
                                          renderIcon={Stop}
                                          onClick={(e) => {
                                            e.stopPropagation()
                                            handleWorkflowAction('cancel', cell.value)
                                          }}
                                        />
                                      )}
                                      <Button
                                        kind="ghost"
                                        size="sm"
                                        hasIconOnly
                                        iconDescription="View Details"
                                        renderIcon={View}
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          setSelectedWorkflow(cell.value)
                                          setActiveTab(1)
                                        }}
                                      />
                                    </div>
                                  </TableCell>
                                )
                              }
                              return (
                                <TableCell key={cell.id}>
                                  {cell.value}
                                </TableCell>
                              )
                            })}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </DataTable>
              
              {apiLoading && (
                <div className="loading-overlay">
                  <Loading withOverlay={false} />
                </div>
              )}
            </div>
          </TabPanel>

          {/* Workflow Details */}
          <TabPanel>
            {selectedWorkflowData ? (
              <div className="workflow-details">
                <div className="details-header">
                  <h3>{selectedWorkflowData.name}</h3>
                  <div className="workflow-info">
                    <Tag type={getStatusTagProps(selectedWorkflowData.status).type}>
                      {getStatusTagProps(selectedWorkflowData.status).text}
                    </Tag>
                    <span>Progress: {selectedWorkflowData.progress}%</span>
                  </div>
                </div>
                
                {selectedWorkflowData.description && (
                  <p className="workflow-description">{selectedWorkflowData.description}</p>
                )}

                <AgentFlowDiagram
                  workflowData={selectedWorkflowData}
                  interactive={true}
                  onNodeClick={(agent) => {
                    console.log('Agent clicked:', agent)
                  }}
                  className="workflow-diagram"
                />

                <div className="agent-details">
                  <h4>Agent Status</h4>
                  <div className="agents-grid">
                    {selectedWorkflowData.agents.map(agent => (
                      <Tile key={agent.id} className="agent-tile">
                        <div className="agent-header">
                          <h5>{agent.name}</h5>
                          <Tag type={getStatusTagProps(agent.status).type} size="sm">
                            {agent.status}
                          </Tag>
                        </div>
                        
                        {agent.status === 'executing' && (
                          <div className="agent-progress">
                            <ProgressBar value={agent.progress} size="sm" />
                            <span className="progress-text">{agent.progress}%</span>
                          </div>
                        )}

                        {agent.currentTask && (
                          <p className="current-task">{agent.currentTask}</p>
                        )}

                        {agent.metrics && (
                          <div className="agent-metrics">
                            <div className="metric">
                              <span className="metric-label">Tokens:</span>
                              <span className="metric-value">{agent.metrics.tokensUsed.toLocaleString()}</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Time:</span>
                              <span className="metric-value">{agent.metrics.executionTime.toFixed(2)}s</span>
                            </div>
                            <div className="metric">
                              <span className="metric-label">Memory:</span>
                              <span className="metric-value">{agent.metrics.memoryUsage.toFixed(1)}MB</span>
                            </div>
                          </div>
                        )}
                      </Tile>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-selection">
                <p>Select a workflow from the table to view details</p>
              </div>
            )}
          </TabPanel>

          {/* Live Logs */}
          <TabPanel>
            <div className="logs-container">
              <div className="logs-header">
                <h3>Live Logs</h3>
                <div className="logs-controls">
                  <Button
                    kind="ghost"
                    size="sm"
                    onClick={() => setRealtimeLogs([])}
                  >
                    Clear Logs
                  </Button>
                  <Button
                    kind="ghost"
                    size="sm"
                    renderIcon={Download}
                    onClick={() => {
                      const logData = realtimeLogs.map(log => 
                        `${log.timestamp} [${log.level.toUpperCase()}] ${log.agent}: ${log.message}`
                      ).join('\\n')
                      
                      const blob = new Blob([logData], { type: 'text/plain' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `workflow-logs-${new Date().toISOString()}.txt`
                      a.click()
                      URL.revokeObjectURL(url)
                    }}
                  >
                    Export
                  </Button>
                </div>
              </div>

              <StructuredListWrapper className="logs-list">
                <StructuredListHead>
                  <StructuredListRow head>
                    <StructuredListCell head>Time</StructuredListCell>
                    <StructuredListCell head>Level</StructuredListCell>
                    <StructuredListCell head>Agent</StructuredListCell>
                    <StructuredListCell head>Message</StructuredListCell>
                  </StructuredListRow>
                </StructuredListHead>
                <StructuredListBody>
                  {realtimeLogs.map((log, index) => (
                    <StructuredListRow key={`${log.id}-${index}`}>
                      <StructuredListCell>
                        <span className="log-timestamp">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                      </StructuredListCell>
                      <StructuredListCell>
                        <Tag 
                          type={
                            log.level === 'error' ? 'red' :
                            log.level === 'warning' ? 'warm-gray' :
                            log.level === 'info' ? 'blue' : 'gray'
                          }
                          size="sm"
                        >
                          {log.level}
                        </Tag>
                      </StructuredListCell>
                      <StructuredListCell>
                        <span className="log-agent">{log.agent}</span>
                      </StructuredListCell>
                      <StructuredListCell>
                        <span className="log-message">{log.message}</span>
                      </StructuredListCell>
                    </StructuredListRow>
                  ))}
                </StructuredListBody>
              </StructuredListWrapper>

              {realtimeLogs.length === 0 && (
                <div className="no-logs">
                  <p>No logs available. Logs will appear here as workflows are processed.</p>
                </div>
              )}
            </div>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </div>
  )
}