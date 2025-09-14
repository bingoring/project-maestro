import { useEffect, useState, useCallback, useRef } from 'react'
import { WebSocketMessage } from '@types'

interface UseWebSocketOptions {
  reconnectAttempts?: number
  reconnectInterval?: number
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  onMessage?: (message: WebSocketMessage) => void
}

interface UseWebSocketReturn {
  messages: WebSocketMessage[]
  sendMessage: (message: WebSocketMessage) => void
  isConnected: boolean
  error: Event | null
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error'
  lastMessage: WebSocketMessage | null
  disconnect: () => void
  reconnect: () => void
}

export const useWebSocket = (
  url: string,
  userId?: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn => {
  const {
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  } = options

  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Event | null>(null)
  const [connectionState, setConnectionState] = useState<UseWebSocketReturn['connectionState']>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const shouldReconnect = useRef(true)

  const getWebSocketUrl = useCallback(() => {
    const wsUrl = url.replace(/^http/, 'ws')
    return userId ? `${wsUrl}/${userId}` : wsUrl
  }, [url, userId])

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      setConnectionState('connecting')
      setError(null)

      const socketUrl = getWebSocketUrl()
      ws.current = new WebSocket(socketUrl)

      ws.current.onopen = () => {
        console.log('WebSocket connected to:', socketUrl)
        setIsConnected(true)
        setConnectionState('connected')
        setError(null)
        reconnectAttemptsRef.current = 0
        onConnect?.()
      }

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          message.timestamp = message.timestamp || new Date().toISOString()
          
          setMessages(prev => [...prev, message])
          setLastMessage(message)
          onMessage?.(message)
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.current.onerror = (errorEvent) => {
        console.error('WebSocket error:', errorEvent)
        setError(errorEvent)
        setConnectionState('error')
        onError?.(errorEvent)
      }

      ws.current.onclose = (closeEvent) => {
        console.log('WebSocket disconnected:', closeEvent.code, closeEvent.reason)
        setIsConnected(false)
        setConnectionState('disconnected')
        onDisconnect?.()

        // Attempt to reconnect if not manually disconnected and within retry limit
        if (
          shouldReconnect.current &&
          reconnectAttemptsRef.current < reconnectAttempts &&
          closeEvent.code !== 1000 // Normal closure
        ) {
          reconnectAttemptsRef.current += 1
          console.log(`Attempting reconnect ${reconnectAttemptsRef.current}/${reconnectAttempts}...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err)
      setError(err as Event)
      setConnectionState('error')
    }
  }, [getWebSocketUrl, reconnectAttempts, reconnectInterval, onConnect, onDisconnect, onError, onMessage])

  const disconnect = useCallback(() => {
    shouldReconnect.current = false
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (ws.current) {
      ws.current.close(1000, 'Manual disconnect')
      ws.current = null
    }

    setIsConnected(false)
    setConnectionState('disconnected')
  }, [])

  const reconnect = useCallback(() => {
    shouldReconnect.current = true
    reconnectAttemptsRef.current = 0
    disconnect()
    setTimeout(() => connect(), 100)
  }, [connect, disconnect])

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      try {
        const messageWithTimestamp = {
          ...message,
          timestamp: message.timestamp || new Date().toISOString(),
        }
        ws.current.send(JSON.stringify(messageWithTimestamp))
      } catch (err) {
        console.error('Failed to send WebSocket message:', err)
      }
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message)
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      shouldReconnect.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [connect])

  return {
    messages,
    sendMessage,
    isConnected,
    error,
    connectionState,
    lastMessage,
    disconnect,
    reconnect,
  }
}