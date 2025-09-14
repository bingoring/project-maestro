// Core types for the Project Maestro frontend

export interface Agent {
  id: string
  name: string
  type: 'orchestrator' | 'codex' | 'canvas' | 'sonata' | 'labyrinth' | 'builder'
  status: 'idle' | 'planning' | 'executing' | 'complete' | 'error' | 'waiting'
  progress: number
  currentTask?: string
  logs: LogEntry[]
  metrics: AgentMetrics
  capabilities: string[]
}

export interface AgentMetrics {
  tokensUsed: number
  executionTime: number
  memoryUsage: number
  apiCalls: number
  cacheHits: number
  errorCount: number
  successRate: number
  lastUpdated: string
}

export interface LogEntry {
  id: string
  timestamp: string
  level: 'info' | 'warning' | 'error' | 'debug'
  agent: string
  message: string
  metadata?: Record<string, any>
}

export interface WorkflowState {
  id: string
  name: string
  description?: string
  status: 'idle' | 'running' | 'complete' | 'error' | 'paused'
  progress: number
  startTime: string
  endTime?: string
  agents: Agent[]
  connections: WorkflowConnection[]
  metadata?: Record<string, any>
}

export interface WorkflowConnection {
  id: string
  from: string
  to: string
  type: 'data' | 'control' | 'feedback'
  dataFlow?: number
  status: 'active' | 'inactive'
}

export interface PromptData {
  prompt: string
  context?: File[]
  complexity: 'simple' | 'moderate' | 'complex'
  timestamp: string
  metadata?: Record<string, any>
}

export interface WorkflowRequest {
  prompt: string
  context?: any
  userId: string
  timestamp: string
}

export interface WorkflowUpdate {
  workflowId: string
  eventType: string
  agent: string
  status: Agent['status']
  progress: number
  logs: LogEntry[]
  timestamp: string
}

export interface SystemMetrics {
  timestamp: string
  system: {
    cpuPercent: number
    memoryPercent: number
    diskUsagePercent: number
    networkIoBytes: number
    processCount: number
  }
  agents: Record<string, AgentMetrics>
  errors: {
    totalErrors: number
    recentErrors: LogEntry[]
  }
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  version: string
  timestamp: string
  components: Record<string, ComponentHealth>
  dependencies: Record<string, DependencyHealth>
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy'
  responseTime?: number
  error?: string
  details?: any
}

export interface DependencyHealth {
  version: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  error?: string
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'workflow_update' | 'agent_status' | 'log_entry' | 'system_metrics' | 'error'
  data: any
  timestamp?: string
}

// Theme types
export interface MaestroTheme {
  agent: {
    planning: string
    executing: string
    waiting: string
    error: string
    complete: string
    idle: string
  }
  workflow: {
    background: string
    border: string
    hover: string
  }
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
  timestamp: string
}

// Form types
export interface PromptFormData {
  prompt: string
  files: File[]
  complexity: PromptData['complexity']
}

// Chart data types
export interface ChartDataPoint {
  timestamp: string
  value: number
  label?: string
}

export interface AgentPerformanceData {
  agent: string
  metric: string
  value: number
  benchmark: number
  trend: 'up' | 'down' | 'stable'
}

// Navigation types
export interface NavigationItem {
  id: string
  label: string
  icon: React.ComponentType<any>
  path: string
  children?: NavigationItem[]
}

// Filter and search types
export interface FilterOptions {
  status?: Agent['status'][]
  agentType?: Agent['type'][]
  timeRange?: {
    start: string
    end: string
  }
}

export interface SearchParams {
  query: string
  filters: FilterOptions
  sortBy: string
  sortOrder: 'asc' | 'desc'
  limit: number
  offset: number
}