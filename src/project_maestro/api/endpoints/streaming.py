"""스트리밍 API 엔드포인트"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import uuid
from typing import Dict, Any, AsyncIterator

from project_maestro.core.streaming_handler import (
    StreamingResponseHandler,
    StreamConfig,
    BufferStrategy,
    StreamChunk
)
from project_maestro.core.observable_orchestrator import ObservableLangGraphOrchestrator
from project_maestro.models.workflow_models import WorkflowRequest
from project_maestro.api.websocket_manager import ConnectionManager

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])
manager = ConnectionManager()

# 글로벌 인스턴스
streaming_handler = StreamingResponseHandler()
observable_orchestrator = ObservableLangGraphOrchestrator()


@router.post("/workflow")
async def stream_workflow(request: WorkflowRequest) -> StreamingResponse:
    """Server-Sent Events를 통한 워크플로우 스트리밍"""
    
    async def generate_sse():
        """SSE 스트림 생성"""
        stream_id = str(uuid.uuid4())
        
        try:
            # 워크플로우 실행 스트림
            workflow_stream = observable_orchestrator.execute_with_tracing(request)
            
            # 이벤트를 StreamChunk로 변환하는 어댑터
            async def event_to_chunk_adapter():
                async for event in workflow_stream:
                    content = json.dumps({
                        "event_type": event.event_type.value,
                        "agent_name": event.agent_name,
                        "timestamp": event.timestamp,
                        "data": event.data,
                        "trace_id": event.trace_id
                    })
                    
                    yield StreamChunk(
                        content=content,
                        agent_name=event.agent_name,
                        timestamp=event.timestamp,
                        chunk_type="event",
                        priority=1 if event.error else 5
                    )
            
            # 버퍼링된 스트림 처리
            buffered_stream = streaming_handler.stream_with_buffering(
                stream_id=stream_id,
                agent_response=event_to_chunk_adapter(),
                strategy=BufferStrategy.ADAPTIVE
            )
            
            # SSE 형식으로 출력
            async for chunk in buffered_stream:
                yield f"data: {chunk}\n\n"
                
        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "timestamp": time.time()
            })
            yield f"data: {error_data}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.websocket("/ws/workflow/{user_id}")
async def websocket_streaming_workflow(websocket: WebSocket, user_id: str):
    """WebSocket을 통한 실시간 워크플로우 스트리밍"""
    
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # 클라이언트로부터 요청 수신
            data = await websocket.receive_json()
            
            if data.get("type") == "workflow_request":
                # 워크플로우 요청 처리
                request_data = data.get("data", {})
                workflow_request = WorkflowRequest(**request_data)
                
                stream_id = str(uuid.uuid4())
                
                # 스트리밍 태스크 시작
                async def stream_workflow_to_websocket():
                    try:
                        # 워크플로우 실행
                        workflow_stream = observable_orchestrator.execute_with_tracing(workflow_request)
                        
                        # 이벤트 스트리밍
                        async for event in workflow_stream:
                            message = {
                                "type": "workflow_event",
                                "stream_id": stream_id,
                                "data": {
                                    "event_type": event.event_type.value,
                                    "agent_name": event.agent_name,
                                    "timestamp": event.timestamp,
                                    "data": event.data,
                                    "trace_id": event.trace_id,
                                    "latency": event.latency,
                                    "error": event.error
                                }
                            }
                            
                            await manager.send_personal_message(message, user_id)
                    
                    except Exception as e:
                        error_message = {
                            "type": "workflow_error",
                            "stream_id": stream_id,
                            "data": {
                                "error": str(e),
                                "timestamp": time.time()
                            }
                        }
                        await manager.send_personal_message(error_message, user_id)
                    
                    finally:
                        complete_message = {
                            "type": "workflow_complete",
                            "stream_id": stream_id,
                            "data": {
                                "timestamp": time.time()
                            }
                        }
                        await manager.send_personal_message(complete_message, user_id)
                
                # 백그라운드에서 스트리밍 시작
                asyncio.create_task(stream_workflow_to_websocket())
                
                # 시작 확인 메시지
                await manager.send_personal_message({
                    "type": "workflow_started",
                    "stream_id": stream_id,
                    "data": {"request_id": workflow_request.request_id}
                }, user_id)
            
            elif data.get("type") == "stream_control":
                # 스트림 제어 (일시정지, 재시작 등)
                control_action = data.get("action")
                stream_id = data.get("stream_id")
                
                # TODO: 스트림 제어 로직 구현
                await manager.send_personal_message({
                    "type": "stream_control_ack",
                    "data": {
                        "action": control_action,
                        "stream_id": stream_id,
                        "status": "acknowledged"
                    }
                }, user_id)
            
            elif data.get("type") == "get_stream_stats":
                # 스트림 통계 조회
                stream_id = data.get("stream_id")
                stats = streaming_handler.get_stream_stats(stream_id)
                
                await manager.send_personal_message({
                    "type": "stream_stats",
                    "data": {
                        "stream_id": stream_id,
                        "stats": stats
                    }
                }, user_id)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket streaming error: {e}")
        manager.disconnect(websocket, user_id)


@router.post("/workflow/multi-agent")
async def stream_multi_agent_workflow(request: WorkflowRequest) -> StreamingResponse:
    """다중 에이전트 병렬 스트리밍"""
    
    async def generate_merged_stream():
        """병합된 스트림 생성"""
        
        try:
            # 각 에이전트별 스트림 생성
            agent_streams = []
            
            # 의도 분석으로 에이전트 선택
            intent_result = await observable_orchestrator.intent_analyzer.analyze_intent(request.prompt)
            
            for agent_name in intent_result.agents:
                agent = observable_orchestrator.agents.get(agent_name)
                if agent:
                    # 에이전트별 스트림을 StreamChunk 형태로 변환
                    async def agent_stream_adapter(agent_instance, agent_name):
                        async for chunk in agent_instance.astream(request):
                            yield StreamChunk(
                                content=json.dumps({
                                    "agent": agent_name,
                                    "content": chunk,
                                    "timestamp": time.time()
                                }),
                                agent_name=agent_name,
                                timestamp=time.time()
                            )
                    
                    agent_streams.append(agent_stream_adapter(agent, agent_name))
            
            # 스트림 병합
            merged_stream = streaming_handler.parallel_stream_merge(
                agent_streams,
                merge_strategy="priority"
            )
            
            # SSE 형식으로 출력
            async for merged_chunk in merged_stream:
                yield f"data: {merged_chunk}\n\n"
        
        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "timestamp": time.time(),
                "type": "multi_agent_error"
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_merged_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/config")
async def get_streaming_config() -> Dict[str, Any]:
    """현재 스트리밍 설정 조회"""
    
    return {
        "config": {
            "buffer_size": streaming_handler.config.buffer_size,
            "buffer_timeout": streaming_handler.config.buffer_timeout,
            "max_chunk_size": streaming_handler.config.max_chunk_size,
            "enable_compression": streaming_handler.config.enable_compression,
            "priority_threshold": streaming_handler.config.priority_threshold,
            "adaptive_buffer": streaming_handler.config.adaptive_buffer
        },
        "active_streams": len(streaming_handler.active_streams),
        "supported_strategies": [strategy.value for strategy in BufferStrategy]
    }


@router.post("/config")
async def update_streaming_config(config_update: Dict[str, Any]) -> Dict[str, Any]:
    """스트리밍 설정 업데이트"""
    
    try:
        # 설정 업데이트
        if "buffer_size" in config_update:
            streaming_handler.config.buffer_size = config_update["buffer_size"]
        
        if "buffer_timeout" in config_update:
            streaming_handler.config.buffer_timeout = config_update["buffer_timeout"]
        
        if "max_chunk_size" in config_update:
            streaming_handler.config.max_chunk_size = config_update["max_chunk_size"]
        
        if "enable_compression" in config_update:
            streaming_handler.config.enable_compression = config_update["enable_compression"]
            streaming_handler.compression_enabled = config_update["enable_compression"]
        
        if "priority_threshold" in config_update:
            streaming_handler.config.priority_threshold = config_update["priority_threshold"]
        
        if "adaptive_buffer" in config_update:
            streaming_handler.config.adaptive_buffer = config_update["adaptive_buffer"]
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "updated_config": {
                "buffer_size": streaming_handler.config.buffer_size,
                "buffer_timeout": streaming_handler.config.buffer_timeout,
                "max_chunk_size": streaming_handler.config.max_chunk_size,
                "enable_compression": streaming_handler.config.enable_compression,
                "priority_threshold": streaming_handler.config.priority_threshold,
                "adaptive_buffer": streaming_handler.config.adaptive_buffer
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/streams/{stream_id}/stats")
async def get_stream_statistics(stream_id: str) -> Dict[str, Any]:
    """특정 스트림의 통계 조회"""
    
    stats = streaming_handler.get_stream_stats(stream_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return {
        "stream_id": stream_id,
        "statistics": stats,
        "timestamp": time.time()
    }


@router.delete("/streams/{stream_id}")
async def terminate_stream(stream_id: str) -> Dict[str, Any]:
    """스트림 종료"""
    
    if stream_id in streaming_handler.active_streams:
        del streaming_handler.active_streams[stream_id]
        
        return {
            "success": True,
            "message": f"Stream {stream_id} terminated successfully",
            "timestamp": time.time()
        }
    else:
        raise HTTPException(status_code=404, detail="Stream not found")


@router.get("/health")
async def streaming_health_check() -> Dict[str, Any]:
    """스트리밍 시스템 헬스 체크"""
    
    return {
        "status": "healthy",
        "active_streams": len(streaming_handler.active_streams),
        "config": {
            "buffer_size": streaming_handler.config.buffer_size,
            "compression_enabled": streaming_handler.compression_enabled
        },
        "timestamp": time.time()
    }