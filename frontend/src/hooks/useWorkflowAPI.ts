import { useState, useCallback } from 'react'
import axios, { AxiosError } from 'axios'
import { ApiResponse, WorkflowRequest, SystemMetrics, HealthStatus, Agent } from '@types'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor for authentication if needed
api.interceptors.request.use((config) => {
  // Add auth token if available
  const token = localStorage.getItem('auth-token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

interface UseWorkflowAPIReturn {
  submitPrompt: (prompt: string, context?: any, userId?: string) => Promise<ApiResponse>
  getWorkflowStatus: (workflowId: string) => Promise<ApiResponse>
  getWorkflows: (filters?: any) => Promise<ApiResponse>
  getAgentStatus: (agentId: string) => Promise<ApiResponse<Agent>>
  getAgentLogs: (agentId: string, limit?: number) => Promise<ApiResponse>
  getSystemMetrics: () => Promise<ApiResponse<SystemMetrics>>
  getHealthStatus: () => Promise<ApiResponse<HealthStatus>>
  pauseWorkflow: (workflowId: string) => Promise<ApiResponse>
  resumeWorkflow: (workflowId: string) => Promise<ApiResponse>
  cancelWorkflow: (workflowId: string) => Promise<ApiResponse>
  loading: boolean
  error: string | null
}

export const useWorkflowAPI = (): UseWorkflowAPIReturn => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRequest = useCallback(async <T>(
    requestFn: () => Promise<{ data: T }>
  ): Promise<ApiResponse<T>> => {
    setLoading(true)
    setError(null)

    try {
      const response = await requestFn()
      return {
        success: true,
        data: response.data,
        timestamp: new Date().toISOString(),
      }
    } catch (err) {
      const errorMessage = err instanceof AxiosError 
        ? err.response?.data?.message || err.message
        : 'Unknown error occurred'
      
      setError(errorMessage)
      return {
        success: false,
        error: errorMessage,
        timestamp: new Date().toISOString(),
      }
    } finally {
      setLoading(false)
    }
  }, [])

  const submitPrompt = useCallback(async (
    prompt: string, 
    context?: any, 
    userId = 'default-user'
  ) => {
    const workflowRequest: WorkflowRequest = {
      prompt,
      context,
      userId,
      timestamp: new Date().toISOString(),
    }

    return handleRequest(() => 
      api.post('/workflows/submit', workflowRequest)
    )
  }, [handleRequest])

  const getWorkflowStatus = useCallback(async (workflowId: string) => {
    return handleRequest(() => 
      api.get(`/workflows/${workflowId}/status`)
    )
  }, [handleRequest])

  const getWorkflows = useCallback(async (filters?: any) => {
    const params = filters ? { params: filters } : {}
    return handleRequest(() => 
      api.get('/workflows', params)
    )
  }, [handleRequest])

  const getAgentStatus = useCallback(async (agentId: string) => {
    return handleRequest(() => 
      api.get(`/agents/${agentId}/status`)
    )
  }, [handleRequest])

  const getAgentLogs = useCallback(async (agentId: string, limit = 100) => {
    return handleRequest(() => 
      api.get(`/agents/${agentId}/logs`, { params: { limit } })
    )
  }, [handleRequest])

  const getSystemMetrics = useCallback(async () => {
    return handleRequest(() => 
      api.get('/metrics')
    )
  }, [handleRequest])

  const getHealthStatus = useCallback(async () => {
    return handleRequest(() => 
      api.get('/health')
    )
  }, [handleRequest])

  const pauseWorkflow = useCallback(async (workflowId: string) => {
    return handleRequest(() => 
      api.post(`/workflows/${workflowId}/pause`)
    )
  }, [handleRequest])

  const resumeWorkflow = useCallback(async (workflowId: string) => {
    return handleRequest(() => 
      api.post(`/workflows/${workflowId}/resume`)
    )
  }, [handleRequest])

  const cancelWorkflow = useCallback(async (workflowId: string) => {
    return handleRequest(() => 
      api.post(`/workflows/${workflowId}/cancel`)
    )
  }, [handleRequest])

  return {
    submitPrompt,
    getWorkflowStatus,
    getWorkflows,
    getAgentStatus,
    getAgentLogs,
    getSystemMetrics,
    getHealthStatus,
    pauseWorkflow,
    resumeWorkflow,
    cancelWorkflow,
    loading,
    error,
  }
}