"""WebSocket endpoints for real-time communication."""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..websocket_manager import connection_manager, handle_websocket_message
from ...core.logging import logger

router = APIRouter()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Main WebSocket endpoint for real-time communication."""
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                await handle_websocket_message(websocket, user_id, message_data)
                
            except json.JSONDecodeError:
                logger.error(
                    "Invalid JSON received from WebSocket client",
                    user_id=user_id,
                    data=data[:100]  # Log first 100 chars
                )
                await connection_manager.send_personal_message({
                    'type': 'error',
                    'data': {
                        'message': 'Invalid JSON format',
                        'error': 'Message must be valid JSON'
                    }
                }, user_id)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {user_id} disconnected")
        connection_manager.disconnect(websocket, user_id)
        
    except Exception as e:
        logger.error(
            "Unexpected error in WebSocket connection",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        connection_manager.disconnect(websocket, user_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    try:
        stats = connection_manager.get_connection_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error("Failed to get WebSocket stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get connection stats")


@router.post("/ws/broadcast")
async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    try:
        await connection_manager.broadcast(message)
        return JSONResponse(content={
            "success": True,
            "message": "Message broadcasted successfully",
            "recipients": len(connection_manager.active_connections)
        })
    except Exception as e:
        logger.error("Failed to broadcast message", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to broadcast message")


@router.post("/ws/send/{user_id}")
async def send_personal_message(user_id: str, message: dict):
    """Send a personal message to a specific user."""
    try:
        await connection_manager.send_personal_message(message, user_id)
        
        if user_id in connection_manager.user_sessions:
            return JSONResponse(content={
                "success": True,
                "message": f"Message sent to user {user_id}",
                "user_id": user_id
            })
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"User {user_id} not connected"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to send personal message",
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to send message")


# Health check for WebSocket functionality
@router.get("/ws/health")
async def websocket_health_check():
    """Health check endpoint for WebSocket functionality."""
    try:
        stats = connection_manager.get_connection_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "websocket_manager": "operational",
            "active_connections": stats["total_connections"],
            "timestamp": stats.get("timestamp", "unknown")
        })
    except Exception as e:
        logger.error("WebSocket health check failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "websocket_manager": "error",
                "error": str(e)
            }
        )