"""관찰 가능성 API 엔드포인트"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime, timedelta

from project_maestro.core.observable_orchestrator import (
    ObservableLangGraphOrchestrator,
    ObservabilityEvent,
    EventType
)
from project_maestro.models.workflow_models import WorkflowRequest
from project_maestro.api.websocket_manager import ConnectionManager

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])
manager = ConnectionManager()

# 글로벌 오케스트레이터 인스턴스
observable_orchestrator = ObservableLangGraphOrchestrator()


@router.get("/metrics/summary")
async def get_metrics_summary(trace_id: Optional[str] = None) -> Dict[str, Any]:
    """성능 메트릭 요약 조회"""
    
    try:
        summary = observable_orchestrator.get_performance_summary(trace_id)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/realtime")
async def get_realtime_metrics() -> Dict[str, Any]:
    """실시간 메트릭 조회"""
    
    try:
        metrics = await observable_orchestrator.get_realtime_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events(
    trace_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """이벤트 조회 (필터링 지원)"""
    
    try:
        events = observable_orchestrator.events
        
        # 필터링
        if trace_id:
            events = [e for e in events if e.trace_id == trace_id]
        
        if agent_name:
            events = [e for e in events if e.agent_name == agent_name]
        
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        
        # 정렬 (최신순)
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)
        
        # 페이징
        total = len(events)
        events = events[offset:offset + limit]
        
        # 직렬화
        serialized_events = []
        for event in events:
            serialized_events.append({
                "id": event.id,
                "event_type": event.event_type.value,
                "agent_name": event.agent_name,
                "timestamp": event.timestamp,
                "data": event.data,
                "trace_id": event.trace_id,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "latency": event.latency,
                "error": event.error,
                "metadata": event.metadata
            })
        
        return JSONResponse(content={
            "events": serialized_events,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces")
async def get_traces(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None  # completed, running, error
) -> Dict[str, Any]:
    """추적 목록 조회"""
    
    try:
        # 추적 ID별로 그룹화
        traces = {}
        for event in observable_orchestrator.events:
            trace_id = event.trace_id
            
            if trace_id not in traces:
                traces[trace_id] = {
                    "trace_id": trace_id,
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "status": "running",
                    "agents": set(),
                    "events_count": 0,
                    "errors": 0
                }
            
            trace = traces[trace_id]
            trace["events_count"] += 1
            trace["agents"].add(event.agent_name)
            
            if event.event_type == EventType.WORKFLOW_START:
                trace["start_time"] = event.timestamp
            elif event.event_type == EventType.WORKFLOW_COMPLETE:
                trace["end_time"] = event.timestamp
                trace["status"] = "completed"
                if trace["start_time"]:
                    trace["duration"] = event.timestamp - trace["start_time"]
            elif event.event_type == EventType.WORKFLOW_ERROR:
                trace["end_time"] = event.timestamp
                trace["status"] = "error"
                if trace["start_time"]:
                    trace["duration"] = event.timestamp - trace["start_time"]
            
            if event.error:
                trace["errors"] += 1
        
        # 활성 추적 상태 업데이트
        for trace_id in observable_orchestrator.active_traces:
            if trace_id in traces:
                traces[trace_id]["status"] = "running"
        
        # 리스트로 변환
        trace_list = []
        for trace_id, trace_data in traces.items():
            trace_data["agents"] = list(trace_data["agents"])
            trace_list.append(trace_data)
        
        # 상태 필터링
        if status:
            trace_list = [t for t in trace_list if t["status"] == status]
        
        # 정렬 (최신순)
        trace_list.sort(key=lambda x: x.get("start_time", 0), reverse=True)
        
        # 페이징
        total = len(trace_list)
        trace_list = trace_list[offset:offset + limit]
        
        return JSONResponse(content={
            "traces": trace_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{trace_id}")
async def get_trace_details(trace_id: str) -> Dict[str, Any]:
    """특정 추적의 상세 정보"""
    
    try:
        # 추적의 모든 이벤트 조회
        trace_events = [
            e for e in observable_orchestrator.events 
            if e.trace_id == trace_id
        ]
        
        if not trace_events:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        # 이벤트 정렬
        trace_events.sort(key=lambda x: x.timestamp)
        
        # 추적 요약
        start_event = next((e for e in trace_events if e.event_type == EventType.WORKFLOW_START), None)
        end_event = next((e for e in trace_events if e.event_type in [EventType.WORKFLOW_COMPLETE, EventType.WORKFLOW_ERROR]), None)
        
        trace_summary = {
            "trace_id": trace_id,
            "start_time": start_event.timestamp if start_event else None,
            "end_time": end_event.timestamp if end_event else None,
            "status": "running",
            "total_events": len(trace_events),
            "agents_involved": list(set(e.agent_name for e in trace_events)),
            "error_count": len([e for e in trace_events if e.error])
        }
        
        if end_event:
            if end_event.event_type == EventType.WORKFLOW_COMPLETE:
                trace_summary["status"] = "completed"
            elif end_event.event_type == EventType.WORKFLOW_ERROR:
                trace_summary["status"] = "error"
            
            if start_event:
                trace_summary["duration"] = end_event.timestamp - start_event.timestamp
        elif trace_id in observable_orchestrator.active_traces:
            trace_summary["status"] = "running"
        
        # 에이전트별 통계
        agent_stats = {}
        for event in trace_events:
            agent = event.agent_name
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "events": 0,
                    "total_latency": 0,
                    "errors": 0,
                    "start_time": None,
                    "end_time": None
                }
            
            stats = agent_stats[agent]
            stats["events"] += 1
            
            if event.latency:
                stats["total_latency"] += event.latency
            
            if event.error:
                stats["errors"] += 1
            
            if event.event_type == EventType.AGENT_START:
                stats["start_time"] = event.timestamp
            elif event.event_type in [EventType.AGENT_COMPLETE, EventType.AGENT_ERROR]:
                stats["end_time"] = event.timestamp
        
        # 에이전트 통계 계산
        for agent, stats in agent_stats.items():
            if stats["events"] > 0:
                stats["avg_latency"] = stats["total_latency"] / stats["events"]
            if stats["start_time"] and stats["end_time"]:
                stats["total_duration"] = stats["end_time"] - stats["start_time"]
        
        # 이벤트 직렬화
        serialized_events = []
        for event in trace_events:
            serialized_events.append({
                "id": event.id,
                "event_type": event.event_type.value,
                "agent_name": event.agent_name,
                "timestamp": event.timestamp,
                "data": event.data,
                "latency": event.latency,
                "error": event.error
            })
        
        return JSONResponse(content={
            "trace_summary": trace_summary,
            "agent_statistics": agent_stats,
            "events": serialized_events
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traces/cleanup")
async def cleanup_old_traces(older_than_hours: int = 24) -> Dict[str, Any]:
    """오래된 추적 데이터 정리"""
    
    try:
        events_before = len(observable_orchestrator.events)
        observable_orchestrator.clear_events(older_than_hours)
        events_after = len(observable_orchestrator.events)
        
        cleaned_count = events_before - events_after
        
        return JSONResponse(content={
            "message": f"Cleaned {cleaned_count} events older than {older_than_hours} hours",
            "events_before": events_before,
            "events_after": events_after,
            "cleaned_count": cleaned_count
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/events/{user_id}")
async def websocket_events_stream(websocket: WebSocket, user_id: str):
    """실시간 이벤트 스트리밍"""
    
    await manager.connect(websocket, user_id)
    
    try:
        # 실시간 이벤트 스트리밍 태스크
        async def stream_events():
            last_event_id = None
            
            while True:
                try:
                    # 새 이벤트 확인
                    new_events = []
                    
                    for event in observable_orchestrator.events:
                        if last_event_id is None or event.id != last_event_id:
                            new_events.append(event)
                            last_event_id = event.id
                    
                    # 새 이벤트가 있으면 전송
                    if new_events:
                        for event in new_events[-10:]:  # 최근 10개만
                            await manager.send_personal_message({
                                "type": "observability_event",
                                "data": {
                                    "id": event.id,
                                    "event_type": event.event_type.value,
                                    "agent_name": event.agent_name,
                                    "timestamp": event.timestamp,
                                    "data": event.data,
                                    "trace_id": event.trace_id,
                                    "latency": event.latency,
                                    "error": event.error
                                }
                            }, user_id)
                    
                    await asyncio.sleep(1)  # 1초마다 체크
                    
                except Exception as e:
                    print(f"Error streaming events: {e}")
                    await asyncio.sleep(5)
        
        # 스트리밍 태스크 시작
        stream_task = asyncio.create_task(stream_events())
        
        # 클라이언트 메시지 수신
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                # 구독 설정 (필터 등)
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "data": {"filters": data.get("filters", {})}
                }, user_id)
            elif data.get("type") == "unsubscribe":
                # 구독 해제
                stream_task.cancel()
                break
                
    except WebSocketDisconnect:
        stream_task.cancel() if 'stream_task' in locals() else None
        manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        stream_task.cancel() if 'stream_task' in locals() else None
        manager.disconnect(websocket, user_id)