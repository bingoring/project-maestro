# Project Maestro: 프론트엔드 인터페이스 구축 로드맵

## 🎨 Phase 0: 프론트엔드 인터페이스 구축 (1-2주)

### 0.1 IBM Carbon Design System 기반 UI 셋업
**우선순위**: 🔴 Critical  
**예상 소요시간**: 2일

#### 구현 내용
```typescript
// frontend/src/setup/carbon-theme.ts
import { Theme } from '@carbon/react';
import { g100 } from '@carbon/themes';

export const maestroTheme = {
  ...g100,
  // AI 워크플로우 시각화를 위한 커스텀 토큰
  custom: {
    agent: {
      planning: '#0f62fe',      // Blue 60
      executing: '#42be65',     // Green 40
      waiting: '#f1c21b',       // Yellow 30
      error: '#da1e28',         // Red 50
      complete: '#24a148'       // Green 50
    },
    workflow: {
      background: '#161616',    // Gray 100
      border: '#393939',        // Gray 80
      hover: '#4c4c4c'         // Gray 70
    }
  }
};

// frontend/src/components/layout/AppShell.tsx
import { 
  Header, 
  HeaderGlobalBar,
  HeaderGlobalAction,
  SideNav,
  SideNavItems,
  Content
} from '@carbon/react';
import { 
  Notification20, 
  UserAvatar20,
  Dashboard20,
  Flow20
} from '@carbon/icons-react';

export const AppShell: React.FC = ({ children }) => {
  return (
    <div className="maestro-app">
      <Header aria-label="Project Maestro">
        <HeaderName prefix="IBM">Project Maestro</HeaderName>
        <HeaderGlobalBar>
          <HeaderGlobalAction aria-label="Notifications">
            <Notification20 />
          </HeaderGlobalAction>
          <HeaderGlobalAction aria-label="User">
            <UserAvatar20 />
          </HeaderGlobalAction>
        </HeaderGlobalBar>
      </Header>
      
      <SideNav aria-label="Side navigation" expanded={true}>
        <SideNavItems>
          <SideNavLink icon={Dashboard20} href="/dashboard">
            Dashboard
          </SideNavLink>
          <SideNavLink icon={Flow20} href="/workflow">
            Workflow Monitor
          </SideNavLink>
        </SideNavItems>
      </SideNav>
      
      <Content>{children}</Content>
    </div>
  );
};
```

#### 기대 효과
- IBM Carbon Design System의 일관된 UI/UX
- AI 워크플로우에 최적화된 시각적 피드백
- 엔터프라이즈급 사용자 경험

### 0.2 프롬프트 입력 인터페이스
**우선순위**: 🔴 Critical  
**예상 소요시간**: 3일

#### 구현 내용
```typescript
// frontend/src/components/prompt/PromptInterface.tsx
import React, { useState, useRef } from 'react';
import { 
  TextArea, 
  Button, 
  Tag,
  InlineNotification,
  FileUploader
} from '@carbon/react';
import { Send20, Add20 } from '@carbon/icons-react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface PromptInterfaceProps {
  onSubmit: (prompt: PromptData) => void;
  isProcessing: boolean;
}

export const PromptInterface: React.FC<PromptInterfaceProps> = ({
  onSubmit,
  isProcessing
}) => {
  const [prompt, setPrompt] = useState('');
  const [context, setContext] = useState<File[]>([]);
  const [complexity, setComplexity] = useState<'simple' | 'moderate' | 'complex'>('moderate');
  
  // 실시간 복잡도 분석
  const analyzeComplexity = (text: string) => {
    const wordCount = text.split(' ').length;
    const hasCode = /```[\\s\\S]*?```/.test(text);
    const hasMultipleSteps = /\\d+\\.|step|then|after/gi.test(text);
    
    if (wordCount > 100 || hasCode || hasMultipleSteps) {
      return 'complex';
    } else if (wordCount > 50) {
      return 'moderate';
    }
    return 'simple';
  };
  
  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    setPrompt(text);
    setComplexity(analyzeComplexity(text));
  };
  
  const handleSubmit = () => {
    onSubmit({
      prompt,
      context,
      complexity,
      timestamp: new Date().toISOString()
    });
  };
  
  return (
    <div className="prompt-interface">
      <div className="prompt-header">
        <h2>AI Assistant Prompt</h2>
        <div className="complexity-indicator">
          <Tag type={
            complexity === 'simple' ? 'green' : 
            complexity === 'moderate' ? 'blue' : 'red'
          }>
            {complexity} query
          </Tag>
        </div>
      </div>
      
      <TextArea
        id="prompt-input"
        labelText="Enter your request"
        placeholder="Describe what you want to build or achieve..."
        rows={6}
        value={prompt}
        onChange={handlePromptChange}
        disabled={isProcessing}
      />
      
      <FileUploader
        labelTitle="Add Context"
        labelDescription="Upload files for additional context"
        buttonLabel="Add files"
        filenameStatus="edit"
        accept={['.txt', '.md', '.json', '.yaml', '.py', '.js', '.ts']}
        multiple
        onChange={(e) => setContext(Array.from(e.target.files || []))}
      />
      
      <div className="prompt-actions">
        <Button 
          kind="primary"
          onClick={handleSubmit}
          disabled={!prompt || isProcessing}
          renderIcon={Send20}
        >
          {isProcessing ? 'Processing...' : 'Send'}
        </Button>
        
        <Button 
          kind="ghost"
          onClick={() => {
            setPrompt('');
            setContext([]);
          }}
          disabled={isProcessing}
        >
          Clear
        </Button>
      </div>
      
      {isProcessing && (
        <InlineNotification
          kind="info"
          title="Processing"
          subtitle="Your request is being processed by the AI agents..."
          hideCloseButton
        />
      )}
    </div>
  );
};
```

#### 기대 효과
- 직관적인 프롬프트 입력 인터페이스
- 실시간 복잡도 분석으로 사용자 경험 향상
- 파일 업로드를 통한 컨텍스트 제공 지원

### 0.3 에이전트 진행 상황 시각화
**우선순위**: 🔴 Critical  
**예상 소요시간**: 4일

#### 구현 내용
```typescript
// frontend/src/components/agents/AgentProgressVisualization.tsx
import React, { useEffect, useState } from 'react';
import { 
  ProgressIndicator, 
  ProgressStep,
  Tile,
  SkeletonText,
  InlineLoading,
  Tag
} from '@carbon/react';
import { CheckmarkFilled20, ErrorFilled20 } from '@carbon/icons-react';
import * as d3 from 'd3';

interface AgentState {
  id: string;
  name: string;
  status: 'idle' | 'planning' | 'executing' | 'complete' | 'error';
  progress: number;
  currentTask?: string;
  logs: string[];
  metrics: {
    tokensUsed: number;
    executionTime: number;
    memoryUsage: number;
  };
}

export const AgentProgressVisualization: React.FC<{
  agents: AgentState[];
  workflow: WorkflowState;
}> = ({ agents, workflow }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  // D3.js를 사용한 워크플로우 그래프 시각화
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 400;
    
    // 에이전트 노드 그리기
    const nodes = agents.map((agent, i) => ({
      id: agent.id,
      name: agent.name,
      status: agent.status,
      x: (i + 1) * (width / (agents.length + 1)),
      y: height / 2
    }));
    
    // 노드 업데이트
    const nodeSelection = svg.selectAll('.agent-node')
      .data(nodes, d => d.id);
    
    // Enter
    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'agent-node')
      .attr('transform', d => `translate(${d.x}, ${d.y})`);
    
    nodeEnter.append('circle')
      .attr('r', 30)
      .attr('fill', d => getStatusColor(d.status))
      .attr('stroke', '#393939')
      .attr('stroke-width', 2);
    
    nodeEnter.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', 'white')
      .style('font-size', '12px')
      .text(d => d.name.substring(0, 3).toUpperCase());
    
    // Update
    nodeSelection.select('circle')
      .transition()
      .duration(500)
      .attr('fill', d => getStatusColor(d.status))
      .attr('r', d => d.status === 'executing' ? 35 : 30);
    
    // 연결선 그리기
    const links = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      links.push({
        source: nodes[i],
        target: nodes[i + 1]
      });
    }
    
    const linkSelection = svg.selectAll('.agent-link')
      .data(links);
    
    linkSelection.enter()
      .append('line')
      .attr('class', 'agent-link')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', '#4c4c4c')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');
    
  }, [agents]);
  
  const getStatusColor = (status: string) => {
    const colors = {
      idle: '#525252',
      planning: '#0f62fe',
      executing: '#42be65',
      complete: '#24a148',
      error: '#da1e28'
    };
    return colors[status] || '#525252';
  };
  
  return (
    <div className="agent-visualization">
      <div className="workflow-graph">
        <svg ref={svgRef} width="800" height="400" />
      </div>
      
      <div className="agent-details">
        {agents.map(agent => (
          <Tile key={agent.id} className="agent-tile">
            <div className="agent-header">
              <h4>{agent.name}</h4>
              <Tag type={
                agent.status === 'complete' ? 'green' :
                agent.status === 'error' ? 'red' :
                agent.status === 'executing' ? 'blue' : 'gray'
              }>
                {agent.status}
              </Tag>
            </div>
            
            {agent.status === 'executing' && (
              <InlineLoading 
                description={agent.currentTask || 'Processing...'}
              />
            )}
            
            {agent.status === 'complete' && (
              <div className="agent-metrics">
                <span>Tokens: {agent.metrics.tokensUsed}</span>
                <span>Time: {agent.metrics.executionTime}s</span>
                <span>Memory: {agent.metrics.memoryUsage}MB</span>
              </div>
            )}
            
            <div className="agent-logs">
              {agent.logs.slice(-3).map((log, i) => (
                <p key={i} className="log-entry">{log}</p>
              ))}
            </div>
          </Tile>
        ))}
      </div>
    </div>
  );
};
```

#### 기대 효과
- 실시간 에이전트 상태 시각화
- D3.js 기반 고급 워크플로우 그래프
- 에이전트별 성능 메트릭 모니터링

### 0.4 실시간 워크플로우 모니터링
**우선순위**: 🔴 Critical  
**예상 소요시간**: 3일

#### 구현 내용
```typescript
// frontend/src/components/monitoring/WorkflowMonitor.tsx
import React, { useEffect, useState } from 'react';
import { 
  DataTable, 
  DataTableHeader,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
  ProgressBar,
  InlineNotification
} from '@carbon/react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentProgressVisualization } from '../agents/AgentProgressVisualization';

export const WorkflowMonitor: React.FC = () => {
  const [workflows, setWorkflows] = useState<WorkflowState[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [realtimeLogs, setRealtimeLogs] = useState<LogEntry[]>([]);
  
  // WebSocket 연결로 실시간 업데이트
  const { messages, sendMessage } = useWebSocket('ws://localhost:8000/ws');
  
  useEffect(() => {
    messages.forEach(msg => {
      if (msg.type === 'workflow_update') {
        updateWorkflow(msg.data);
      } else if (msg.type === 'log_entry') {
        addLogEntry(msg.data);
      } else if (msg.type === 'agent_status') {
        updateAgentStatus(msg.data);
      }
    });
  }, [messages]);
  
  const updateWorkflow = (data: WorkflowUpdate) => {
    setWorkflows(prev => {
      const index = prev.findIndex(w => w.id === data.id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = { ...updated[index], ...data };
        return updated;
      }
      return [...prev, data];
    });
  };
  
  const addLogEntry = (entry: LogEntry) => {
    setRealtimeLogs(prev => [...prev.slice(-99), entry]);
  };
  
  return (
    <div className="workflow-monitor">
      <div className="monitor-header">
        <h2>Workflow Monitor</h2>
        <div className="monitor-stats">
          <Tag>Active: {workflows.filter(w => w.status === 'running').length}</Tag>
          <Tag>Completed: {workflows.filter(w => w.status === 'complete').length}</Tag>
          <Tag>Failed: {workflows.filter(w => w.status === 'error').length}</Tag>
        </div>
      </div>
      
      <div className="monitor-content">
        <div className="workflows-list">
          <DataTable
            rows={workflows}
            headers={[
              { key: 'id', header: 'ID' },
              { key: 'name', header: 'Workflow' },
              { key: 'status', header: 'Status' },
              { key: 'progress', header: 'Progress' },
              { key: 'startTime', header: 'Started' }
            ]}
            render={({ rows, headers }) => (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      {headers.map(header => (
                        <TableHeader key={header.key}>
                          {header.header}
                        </TableHeader>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {rows.map(row => (
                      <TableRow 
                        key={row.id}
                        onClick={() => setSelectedWorkflow(row.id)}
                        className={selectedWorkflow === row.id ? 'selected' : ''}
                      >
                        {row.cells.map(cell => (
                          <TableCell key={cell.id}>
                            {cell.info.header === 'progress' ? (
                              <ProgressBar value={cell.value} />
                            ) : (
                              cell.value
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          />
        </div>
        
        {selectedWorkflow && (
          <div className="workflow-details">
            <AgentProgressVisualization
              agents={getWorkflowAgents(selectedWorkflow)}
              workflow={workflows.find(w => w.id === selectedWorkflow)}
            />
          </div>
        )}
        
        <div className="realtime-logs">
          <h3>Real-time Logs</h3>
          <StructuredListWrapper>
            <StructuredListBody>
              {realtimeLogs.map((log, i) => (
                <StructuredListRow key={i}>
                  <StructuredListCell>
                    <span className="log-timestamp">{log.timestamp}</span>
                  </StructuredListCell>
                  <StructuredListCell>
                    <Tag type={log.level}>{log.agent}</Tag>
                  </StructuredListCell>
                  <StructuredListCell>
                    {log.message}
                  </StructuredListCell>
                </StructuredListRow>
              ))}
            </StructuredListBody>
          </StructuredListWrapper>
        </div>
      </div>
    </div>
  );
};
```

#### 기대 효과
- 실시간 워크플로우 상태 모니터링
- 에이전트별 로그 스트리밍
- 직관적인 테이블 기반 데이터 표시

## 🔌 Phase 0.5: 백엔드 API 통합 (3-4일)

### 0.5.1 FastAPI WebSocket 엔드포인트
**우선순위**: 🔴 Critical  
**예상 소요시간**: 2일

#### 구현 내용
```python
# src/project_maestro/api/websocket_manager.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import json

class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_sessions[user_id] = websocket
        
    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
            
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.user_sessions:
            websocket = self.user_sessions[user_id]
            await websocket.send_json(message)
            
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# src/project_maestro/api/streaming_endpoints.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio

router = APIRouter(prefix="/api/v1")

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()
            
            if data['type'] == 'prompt_submission':
                # 워크플로우 시작
                workflow_id = await start_workflow(data['prompt'])
                
                # 실시간 업데이트 스트리밍
                async for update in process_workflow_with_updates(workflow_id):
                    await manager.send_personal_message({
                        'type': 'workflow_update',
                        'data': update
                    }, user_id)
                    
            elif data['type'] == 'agent_action':
                # 에이전트 액션 처리
                result = await handle_agent_action(data['action'])
                await manager.send_personal_message({
                    'type': 'action_result',
                    'data': result
                }, user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        
async def process_workflow_with_updates(workflow_id: str):
    """워크플로우 처리 중 실시간 업데이트 생성"""
    
    from project_maestro.core.langgraph_orchestrator import LangGraphOrchestrator
    orchestrator = LangGraphOrchestrator()
    
    async for event in orchestrator.astream_events(workflow_id):
        yield {
            'workflow_id': workflow_id,
            'event_type': event.type,
            'agent': event.agent_name,
            'status': event.status,
            'progress': event.progress,
            'logs': event.logs,
            'timestamp': datetime.now().isoformat()
        }

@router.post("/stream/workflow")
async def stream_workflow(request: WorkflowRequest):
    """Server-Sent Events를 통한 워크플로우 스트리밍"""
    
    async def generate():
        workflow_id = await start_workflow(request.prompt)
        
        async for update in process_workflow_with_updates(workflow_id):
            yield f"data: {json.dumps(update)}\\n\\n"
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
```

#### 기대 효과
- 실시간 양방향 통신 지원
- WebSocket과 Server-Sent Events 모두 지원
- 확장 가능한 연결 관리 시스템

### 0.5.2 프론트엔드 API 통합 훅
**우선순위**: 🔴 Critical  
**예상 소요시간**: 2일

#### 구현 내용
```typescript
// frontend/src/hooks/useWebSocket.ts
import { useEffect, useState, useCallback, useRef } from 'react';

interface WebSocketMessage {
  type: string;
  data: any;
}

export const useWebSocket = (url: string) => {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();
  
  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        setIsConnected(true);
        setError(null);
        console.log('WebSocket connected');
      };
      
      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        setMessages(prev => [...prev, message]);
      };
      
      ws.current.onerror = (error) => {
        setError(new Error('WebSocket error'));
        console.error('WebSocket error:', error);
      };
      
      ws.current.onclose = () => {
        setIsConnected(false);
        // 자동 재연결
        reconnectTimeout.current = setTimeout(() => {
          connect();
        }, 3000);
      };
    } catch (err) {
      setError(err as Error);
    }
  }, [url]);
  
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  }, []);
  
  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
    }
    if (ws.current) {
      ws.current.close();
    }
  }, []);
  
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);
  
  return {
    messages,
    sendMessage,
    isConnected,
    error,
    disconnect
  };
};

// frontend/src/hooks/useWorkflowAPI.ts
import { useState, useCallback } from 'react';
import axios from 'axios';

export const useWorkflowAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const submitPrompt = useCallback(async (prompt: string, context?: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/v1/workflow/submit', {
        prompt,
        context,
        user_id: getUserId(),
        timestamp: new Date().toISOString()
      });
      
      return response.data;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  const getWorkflowStatus = useCallback(async (workflowId: string) => {
    const response = await axios.get(`/api/v1/workflow/${workflowId}/status`);
    return response.data;
  }, []);
  
  const getAgentLogs = useCallback(async (agentId: string, limit = 100) => {
    const response = await axios.get(`/api/v1/agent/${agentId}/logs`, {
      params: { limit }
    });
    return response.data;
  }, []);
  
  return {
    submitPrompt,
    getWorkflowStatus,
    getAgentLogs,
    loading,
    error
  };
};
```

#### 기대 효과
- React 기반 실시간 통신 훅
- 자동 재연결 및 에러 처리
- 타입 안전성과 재사용성

## 📊 프론트엔드 기술 스택

### 핵심 프레임워크
- **React 18+**: UI 프레임워크
- **TypeScript**: 타입 안정성
- **Vite**: 빌드 도구
- **React Query**: 서버 상태 관리
- **Zustand**: 클라이언트 상태 관리

### UI/UX 라이브러리
- **IBM Carbon Design System**: 디자인 시스템
- **D3.js**: 고급 시각화
- **Recharts**: 차트 컴포넌트
- **Framer Motion**: 애니메이션

### 개발 도구
- **ESLint/Prettier**: 코드 품질
- **Storybook**: 컴포넌트 문서화
- **Playwright**: E2E 테스팅
- **Vitest**: 유닛 테스팅

### 프로덕션 도구
- **Sentry**: 에러 모니터링
- **Google Analytics**: 사용자 분석
- **Lighthouse CI**: 성능 모니터링
- **Cloudflare**: CDN 및 보안

## 🚀 프론트엔드 구현 일정

### Week 1: 기초 설정 및 핵심 컴포넌트
- [ ] IBM Carbon Design System 셋업 및 테마 구성
- [ ] 프로젝트 구조 및 라우팅 설정
- [ ] 프롬프트 입력 인터페이스 구현
- [ ] 기본 레이아웃 및 네비게이션

### Week 2: 실시간 시각화 및 API 통합
- [ ] WebSocket 연결 및 실시간 통신 구현
- [ ] 에이전트 진행 상황 D3.js 시각화
- [ ] 워크플로우 모니터링 인터페이스
- [ ] API 훅 및 상태 관리 시스템

이 프론트엔드 로드맵은 기존 백엔드 로드맵과 완벽하게 통합되어, 사용자가 AI 에이전트 워크플로우를 직관적으로 상호작용하고 모니터링할 수 있는 인터페이스를 제공합니다.