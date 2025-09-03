"""Event streaming and management API endpoints."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse

from ...core.message_queue import EventType
from ...core.logging import get_logger
from ..models import EventResponse, PaginationParams, PaginatedResponse

router = APIRouter()
logger = get_logger("api.events")

# Store active WebSocket connections
active_websockets: List[WebSocket] = []


@router.get("/", response_model=PaginatedResponse)
async def list_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    source: Optional[str] = Query(None, description="Filter by event source"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    since: Optional[datetime] = Query(None, description="Events since this timestamp"),
    pagination: PaginationParams = Query()
):
    """List recent events with filtering and pagination."""
    
    # Mock event data
    mock_events = []
    event_types = [et.value for et in EventType]
    sources = ["orchestrator", "codex_agent", "canvas_agent", "sonata_agent", "labyrinth_agent", "builder_agent", "api"]
    
    for i in range(100):
        event_id = f"event_{i:03d}"
        timestamp = datetime.now() - timedelta(minutes=i * 2)
        
        # Skip events older than 'since' filter
        if since and timestamp < since:
            continue
            
        mock_event = EventResponse(
            event_id=event_id,
            event_type=event_types[i % len(event_types)],
            source=sources[i % len(sources)],
            data={
                "project_id": f"project_{i % 10}",
                "task_id": f"task_{i}",
                "status": "completed" if i % 3 == 0 else "in_progress"
            },
            timestamp=timestamp,
            correlation_id=f"corr_{i % 5}" if i % 5 == 0 else None
        )
        
        # Apply filters
        if event_type and mock_event.event_type != event_type:
            continue
        if source and mock_event.source != source:
            continue
        if correlation_id and mock_event.correlation_id != correlation_id:
            continue
            
        mock_events.append(mock_event)
    
    # Apply pagination
    start_idx = (pagination.page - 1) * pagination.size
    end_idx = start_idx + pagination.size
    page_items = mock_events[start_idx:end_idx]
    
    total = len(mock_events)
    pages = (total + pagination.size - 1) // pagination.size if total > 0 else 1
    
    return PaginatedResponse(
        items=page_items,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages,
        has_next=pagination.page < pages,
        has_prev=pagination.page > 1
    )


@router.get("/types", response_model=List[str])
async def list_event_types():
    """List all available event types."""
    return [event_type.value for event_type in EventType]


@router.get("/sources", response_model=List[str])
async def list_event_sources():
    """List all known event sources."""
    return [
        "orchestrator",
        "codex_agent", 
        "canvas_agent",
        "sonata_agent",
        "labyrinth_agent",
        "builder_agent",
        "api",
        "system"
    ]


@router.get("/stream")
async def stream_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    source: Optional[str] = Query(None, description="Filter by event source")
):
    """Stream events in real-time using Server-Sent Events (SSE)."""
    
    async def event_generator():
        """Generate events for SSE stream."""
        counter = 0
        
        while True:
            # In a real implementation, this would listen to the actual event bus
            # For demo purposes, we'll generate mock events
            
            event_data = {
                "event_id": f"stream_event_{counter}",
                "event_type": "task.completed",
                "source": "codex_agent",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "project_id": "demo_project",
                    "task_id": f"task_{counter}",
                    "status": "completed"
                }
            }
            
            # Apply filters
            if event_type and event_data["event_type"] != event_type:
                await asyncio.sleep(1)
                continue
                
            if source and event_data["source"] != source:
                await asyncio.sleep(1)
                continue
            
            yield {
                "event": "project_event",
                "data": f"{event_data}",
                "id": counter,
                "retry": 5000
            }
            
            counter += 1
            await asyncio.sleep(2)  # Send event every 2 seconds
    
    return EventSourceResponse(event_generator())


@router.websocket("/ws")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming."""
    
    await websocket.accept()
    active_websockets.append(websocket)
    
    logger.info("WebSocket client connected", client=websocket.client)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (optional filtering, subscriptions)
                message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                
                # Handle subscription requests
                if message.get("action") == "subscribe":
                    event_types = message.get("event_types", [])
                    sources = message.get("sources", [])
                    
                    await websocket.send_json({
                        "type": "subscription",
                        "status": "subscribed",
                        "event_types": event_types,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
            # In a real implementation, this would forward actual events
            # from the event bus to subscribed WebSocket clients
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", client=websocket.client)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), client=websocket.client)
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_event_to_websockets(event_data: dict):
    """Broadcast event to all active WebSocket connections."""
    
    if not active_websockets:
        return
        
    # Prepare event message
    message = {
        "type": "event",
        "event": event_data,
        "timestamp": datetime.now().isoformat()
    }
    
    # Send to all connected clients
    disconnected = []
    
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning("Failed to send event to WebSocket client", error=str(e))
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        active_websockets.remove(websocket)


@router.get("/statistics")
async def get_event_statistics(
    hours: int = Query(24, description="Number of hours to analyze")
):
    """Get event statistics and patterns."""
    
    return {
        "period_hours": hours,
        "total_events": 1247,
        "events_per_hour": 52,
        "event_type_distribution": [
            {"type": "task.created", "count": 342, "percentage": 27.4},
            {"type": "task.completed", "count": 298, "percentage": 23.9},
            {"type": "asset.generated", "count": 186, "percentage": 14.9},
            {"type": "build.started", "count": 125, "percentage": 10.0},
            {"type": "build.completed", "count": 118, "percentage": 9.5},
            {"type": "project.created", "count": 89, "percentage": 7.1},
            {"type": "task.failed", "count": 45, "percentage": 3.6},
            {"type": "agent.status_changed", "count": 44, "percentage": 3.5}
        ],
        "source_distribution": [
            {"source": "orchestrator", "count": 287, "percentage": 23.0},
            {"source": "codex_agent", "count": 245, "percentage": 19.6},
            {"source": "canvas_agent", "count": 198, "percentage": 15.9},
            {"source": "builder_agent", "count": 156, "percentage": 12.5},
            {"source": "sonata_agent", "count": 134, "percentage": 10.7},
            {"source": "labyrinth_agent", "count": 127, "percentage": 10.2},
            {"source": "api", "count": 100, "percentage": 8.0}
        ],
        "peak_hours": [
            {"hour": 14, "events": 89},
            {"hour": 15, "events": 87},
            {"hour": 10, "events": 82},
            {"hour": 11, "events": 79},
            {"hour": 16, "events": 74}
        ]
    }


@router.get("/correlation/{correlation_id}", response_model=List[EventResponse])
async def get_correlated_events(
    correlation_id: str,
    limit: int = Query(100, description="Maximum events to return")
):
    """Get all events with the same correlation ID."""
    
    # Mock correlated events
    mock_events = []
    
    # Generate a sequence of related events
    event_sequence = [
        ("project.created", "api"),
        ("task.created", "orchestrator"),
        ("task.started", "codex_agent"),
        ("asset.generated", "codex_agent"),
        ("task.completed", "codex_agent"),
        ("task.created", "orchestrator"),
        ("task.started", "canvas_agent"),
        ("asset.generated", "canvas_agent"),
        ("task.completed", "canvas_agent"),
        ("build.started", "builder_agent"),
        ("build.completed", "builder_agent")
    ]
    
    for i, (event_type, source) in enumerate(event_sequence[:limit]):
        mock_events.append(EventResponse(
            event_id=f"{correlation_id}_event_{i:03d}",
            event_type=event_type,
            source=source,
            data={
                "project_id": "demo_project",
                "step": i + 1,
                "total_steps": len(event_sequence)
            },
            timestamp=datetime.now() - timedelta(minutes=(len(event_sequence) - i) * 5),
            correlation_id=correlation_id
        ))
    
    return mock_events