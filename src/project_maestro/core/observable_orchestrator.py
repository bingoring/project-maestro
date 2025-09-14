"""Observable LangGraph 오케스트레이터 구현"""

import asyncio
import time
from typing import Dict, Any, AsyncIterator, List, Optional
import structlog
from dataclasses import dataclass
from enum import Enum
import uuid
from langfuse import Langfuse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from project_maestro.core.base_orchestrator import LangGraphOrchestrator
from project_maestro.models.workflow_models import WorkflowRequest, WorkflowResponse
from project_maestro.core.metrics import PrometheusMetrics


class EventType(Enum):
    """이벤트 타입 정의"""
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete" 
    AGENT_ERROR = "agent_error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    MEMORY_UPDATE = "memory_update"
    ROUTING_DECISION = "routing_decision"


@dataclass
class ObservabilityEvent:
    """관찰 가능성 이벤트"""
    id: str
    event_type: EventType
    agent_name: str
    timestamp: float
    data: Dict[str, Any]
    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    latency: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class ObservableLangGraphOrchestrator(LangGraphOrchestrator):
    """관찰 가능한 LangGraph 오케스트레이터"""
    
    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger()
        self.metrics = PrometheusMetrics()
        self.events: List[ObservabilityEvent] = []
        self.active_traces: Dict[str, Dict] = {}
        
        # Langfuse 초기화 (옵셔널)
        self.langfuse = None
        try:
            self.langfuse = Langfuse()
            self.logger.info("Langfuse initialized for tracing")
        except Exception as e:
            self.logger.warning(f"Langfuse initialization failed: {e}")
    
    async def execute_with_tracing(
        self, 
        request: WorkflowRequest
    ) -> AsyncIterator[ObservabilityEvent]:
        """추적 기능이 있는 워크플로우 실행"""
        
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Langfuse 추적 시작
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="workflow_execution",
                id=trace_id,
                input=request.dict(),
                user_id=request.user_id
            )
        
        self.active_traces[trace_id] = {
            'trace': trace,
            'start_time': start_time,
            'request': request
        }
        
        try:
            # 워크플로우 시작 이벤트
            start_event = ObservabilityEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.WORKFLOW_START,
                agent_name="orchestrator",
                timestamp=start_time,
                data={"request": request.dict()},
                trace_id=trace_id
            )
            self.events.append(start_event)
            yield start_event
            
            # 워크플로우 실행
            async for event in self._execute_with_events(request, trace_id):
                self.events.append(event)
                yield event
                
                # 메트릭 업데이트
                await self._update_metrics(event)
                
                # 로깅
                await self._log_event(event)
            
            # 워크플로우 완료
            end_time = time.perf_counter()
            total_latency = end_time - start_time
            
            complete_event = ObservabilityEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.WORKFLOW_COMPLETE,
                agent_name="orchestrator",
                timestamp=end_time,
                data={"total_latency": total_latency},
                trace_id=trace_id,
                latency=total_latency
            )
            self.events.append(complete_event)
            yield complete_event
            
            # Langfuse 완료
            if trace:
                trace.update(
                    output={"success": True},
                    end_time=end_time
                )
                
        except Exception as e:
            error_time = time.perf_counter()
            error_event = ObservabilityEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.WORKFLOW_ERROR,
                agent_name="orchestrator",
                timestamp=error_time,
                data={"error": str(e)},
                trace_id=trace_id,
                error=str(e)
            )
            self.events.append(error_event)
            yield error_event
            
            # Langfuse 에러 기록
            if trace:
                trace.update(
                    output={"error": str(e)},
                    end_time=error_time,
                    level="ERROR"
                )
            raise
            
        finally:
            # 추적 정리
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]
    
    async def _execute_with_events(
        self, 
        request: WorkflowRequest, 
        trace_id: str
    ) -> AsyncIterator[ObservabilityEvent]:
        """이벤트 생성을 포함한 워크플로우 실행"""
        
        # 의도 분석 및 라우팅
        routing_start = time.perf_counter()
        intent_result = await self.intent_analyzer.analyze_intent(request.prompt)
        routing_end = time.perf_counter()
        
        routing_event = ObservabilityEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.ROUTING_DECISION,
            agent_name="intent_analyzer",
            timestamp=routing_end,
            data={
                "intent": intent_result.intent,
                "complexity": intent_result.complexity,
                "confidence": intent_result.confidence,
                "selected_agents": intent_result.agents
            },
            trace_id=trace_id,
            latency=routing_end - routing_start
        )
        yield routing_event
        
        # 각 에이전트 실행 추적
        for agent_name in intent_result.agents:
            agent_start = time.perf_counter()
            
            # 에이전트 시작 이벤트
            start_event = ObservabilityEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.AGENT_START,
                agent_name=agent_name,
                timestamp=agent_start,
                data={"request": request.dict()},
                trace_id=trace_id
            )
            yield start_event
            
            try:
                # 에이전트 실행
                agent = self.agents[agent_name]
                
                # Langfuse 스팬 생성
                span = None
                if self.langfuse and trace_id in self.active_traces:
                    trace = self.active_traces[trace_id]['trace']
                    span = trace.span(
                        name=f"agent_{agent_name}",
                        input={"prompt": request.prompt},
                        start_time=agent_start
                    )
                
                # 에이전트 실행 및 스트리밍
                async for chunk in agent.astream(request):
                    # 중간 결과 이벤트
                    chunk_event = ObservabilityEvent(
                        id=str(uuid.uuid4()),
                        event_type=EventType.AGENT_START,  # 진행 중
                        agent_name=agent_name,
                        timestamp=time.perf_counter(),
                        data={"chunk": chunk},
                        trace_id=trace_id
                    )
                    yield chunk_event
                
                agent_end = time.perf_counter()
                agent_latency = agent_end - agent_start
                
                # 에이전트 완료 이벤트
                complete_event = ObservabilityEvent(
                    id=str(uuid.uuid4()),
                    event_type=EventType.AGENT_COMPLETE,
                    agent_name=agent_name,
                    timestamp=agent_end,
                    data={"success": True},
                    trace_id=trace_id,
                    latency=agent_latency
                )
                yield complete_event
                
                # Langfuse 스팬 완료
                if span:
                    span.update(
                        output={"success": True},
                        end_time=agent_end
                    )
                    
            except Exception as e:
                error_time = time.perf_counter()
                error_latency = error_time - agent_start
                
                # 에이전트 에러 이벤트
                error_event = ObservabilityEvent(
                    id=str(uuid.uuid4()),
                    event_type=EventType.AGENT_ERROR,
                    agent_name=agent_name,
                    timestamp=error_time,
                    data={"error": str(e)},
                    trace_id=trace_id,
                    latency=error_latency,
                    error=str(e)
                )
                yield error_event
                
                # Langfuse 스팬 에러
                if span:
                    span.update(
                        output={"error": str(e)},
                        end_time=error_time,
                        level="ERROR"
                    )
                
                # 에러 처리 - 다른 에이전트 계속 실행
                self.logger.error(f"Agent {agent_name} failed: {e}")
    
    async def _update_metrics(self, event: ObservabilityEvent):
        """메트릭 업데이트"""
        
        # 실행 시간 메트릭
        if event.latency:
            self.metrics.observe_histogram(
                'agent_execution_time',
                event.latency,
                labels={
                    'agent': event.agent_name,
                    'event_type': event.event_type.value
                }
            )
        
        # 이벤트 카운터
        self.metrics.increment_counter(
            'agent_events_total',
            labels={
                'agent': event.agent_name,
                'event_type': event.event_type.value,
                'status': 'error' if event.error else 'success'
            }
        )
        
        # 에러율 계산
        if event.event_type in [EventType.AGENT_ERROR, EventType.WORKFLOW_ERROR]:
            self.metrics.increment_counter(
                'agent_errors_total',
                labels={'agent': event.agent_name}
            )
    
    async def _log_event(self, event: ObservabilityEvent):
        """구조화된 로깅"""
        
        log_data = {
            "event_id": event.id,
            "event_type": event.event_type.value,
            "agent": event.agent_name,
            "timestamp": event.timestamp,
            "trace_id": event.trace_id,
            "latency": event.latency,
            "data": event.data
        }
        
        if event.error:
            self.logger.error("Agent execution error", **log_data, error=event.error)
        else:
            self.logger.info("Agent event", **log_data)
    
    def get_performance_summary(
        self, 
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """성능 요약 생성"""
        
        events = self.events
        if trace_id:
            events = [e for e in events if e.trace_id == trace_id]
        
        if not events:
            return {}
        
        # 에이전트별 통계
        agent_stats = {}
        for event in events:
            agent = event.agent_name
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'executions': 0,
                    'total_latency': 0,
                    'errors': 0,
                    'success_rate': 0
                }
            
            if event.event_type == EventType.AGENT_COMPLETE:
                agent_stats[agent]['executions'] += 1
                if event.latency:
                    agent_stats[agent]['total_latency'] += event.latency
            elif event.event_type == EventType.AGENT_ERROR:
                agent_stats[agent]['errors'] += 1
        
        # 성공률 계산
        for agent, stats in agent_stats.items():
            total = stats['executions'] + stats['errors']
            if total > 0:
                stats['success_rate'] = stats['executions'] / total
                stats['avg_latency'] = (
                    stats['total_latency'] / stats['executions'] 
                    if stats['executions'] > 0 else 0
                )
        
        return {
            'total_events': len(events),
            'unique_traces': len(set(e.trace_id for e in events)),
            'agent_statistics': agent_stats,
            'error_rate': len([e for e in events if e.error]) / len(events),
            'avg_workflow_time': self._calculate_avg_workflow_time(events)
        }
    
    def _calculate_avg_workflow_time(self, events: List[ObservabilityEvent]) -> float:
        """평균 워크플로우 실행 시간 계산"""
        
        workflow_times = []
        current_workflows = {}
        
        for event in events:
            trace_id = event.trace_id
            
            if event.event_type == EventType.WORKFLOW_START:
                current_workflows[trace_id] = event.timestamp
            elif event.event_type in [EventType.WORKFLOW_COMPLETE, EventType.WORKFLOW_ERROR]:
                if trace_id in current_workflows:
                    duration = event.timestamp - current_workflows[trace_id]
                    workflow_times.append(duration)
                    del current_workflows[trace_id]
        
        return sum(workflow_times) / len(workflow_times) if workflow_times else 0
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """실시간 메트릭 조회"""
        
        recent_events = [
            e for e in self.events 
            if time.time() - e.timestamp < 300  # 최근 5분
        ]
        
        return {
            'active_workflows': len(self.active_traces),
            'recent_events': len(recent_events),
            'current_throughput': len(recent_events) / 300,  # events per second
            'active_agents': list(set(e.agent_name for e in recent_events)),
            'error_rate': len([e for e in recent_events if e.error]) / max(len(recent_events), 1)
        }
    
    def clear_events(self, older_than_hours: int = 24):
        """오래된 이벤트 정리"""
        
        cutoff_time = time.time() - (older_than_hours * 3600)
        self.events = [
            event for event in self.events 
            if event.timestamp > cutoff_time
        ]
        
        self.logger.info(f"Cleared events older than {older_than_hours} hours")