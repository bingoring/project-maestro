"""WebSocket manager for real-time communication."""

import json
import asyncio
from typing import Dict, List, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from ..core.logging import logger


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        self.active_connections.append(websocket)
        self.user_sessions[user_id] = websocket
        self.connection_metadata[user_id] = {
            'connected_at': datetime.now(),
            'user_id': user_id,
            'websocket': websocket
        }
        
        logger.info(
            "WebSocket connection established",
            user_id=user_id,
            total_connections=len(self.active_connections)
        )
        
        # Send initial connection confirmation
        await self.send_personal_message({
            'type': 'connection_established',
            'data': {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'WebSocket connection established'
            }
        }, user_id)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect and cleanup WebSocket connection."""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
                
            if user_id in self.connection_metadata:
                del self.connection_metadata[user_id]
                
            logger.info(
                "WebSocket connection closed",
                user_id=user_id,
                total_connections=len(self.active_connections)
            )
        except Exception as e:
            logger.error(
                "Error during WebSocket disconnect",
                user_id=user_id,
                error=str(e)
            )
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to a specific user."""
        if user_id in self.user_sessions:
            websocket = self.user_sessions[user_id]
            try:
                message_with_timestamp = {
                    **message,
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(message_with_timestamp))
            except Exception as e:
                logger.error(
                    "Failed to send personal message",
                    user_id=user_id,
                    error=str(e)
                )
                # Remove broken connection
                self.disconnect(websocket, user_id)
    
    async def broadcast(self, message: dict, exclude_users: Set[str] = None):
        """Broadcast message to all connected users."""
        if exclude_users is None:
            exclude_users = set()
        
        message_with_timestamp = {
            **message,
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected_users = []
        
        for user_id, websocket in self.user_sessions.items():
            if user_id not in exclude_users:
                try:
                    await websocket.send_text(json.dumps(message_with_timestamp))
                except Exception as e:
                    logger.error(
                        "Failed to send broadcast message",
                        user_id=user_id,
                        error=str(e)
                    )
                    disconnected_users.append((websocket, user_id))
        
        # Clean up disconnected users
        for websocket, user_id in disconnected_users:
            self.disconnect(websocket, user_id)
    
    async def send_workflow_update(self, workflow_data: dict):
        """Send workflow update to all connected clients."""
        await self.broadcast({
            'type': 'workflow_update',
            'data': workflow_data
        })
    
    async def send_agent_status(self, agent_data: dict):
        """Send agent status update to all connected clients."""
        await self.broadcast({
            'type': 'agent_status',
            'data': agent_data
        })
    
    async def send_log_entry(self, log_data: dict):
        """Send log entry to all connected clients."""
        await self.broadcast({
            'type': 'log_entry',
            'data': log_data
        })
    
    async def send_system_metrics(self, metrics_data: dict):
        """Send system metrics to all connected clients."""
        await self.broadcast({
            'type': 'system_metrics',
            'data': metrics_data
        })
    
    def get_connection_stats(self) -> dict:
        """Get current connection statistics."""
        return {
            'total_connections': len(self.active_connections),
            'active_users': list(self.user_sessions.keys()),
            'connection_details': {
                user_id: {
                    'connected_at': metadata['connected_at'].isoformat(),
                    'duration_seconds': (datetime.now() - metadata['connected_at']).total_seconds()
                }
                for user_id, metadata in self.connection_metadata.items()
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()


async def handle_websocket_message(websocket: WebSocket, user_id: str, message_data: dict):
    """Handle incoming WebSocket messages from clients."""
    try:
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        logger.info(
            "WebSocket message received",
            user_id=user_id,
            message_type=message_type
        )
        
        if message_type == 'prompt_submission':
            await handle_prompt_submission(user_id, data)
        elif message_type == 'agent_action':
            await handle_agent_action(user_id, data)
        elif message_type == 'workflow_control':
            await handle_workflow_control(user_id, data)
        elif message_type == 'ping':
            await connection_manager.send_personal_message({
                'type': 'pong',
                'data': {'timestamp': datetime.now().isoformat()}
            }, user_id)
        else:
            logger.warning(
                "Unknown message type received",
                user_id=user_id,
                message_type=message_type
            )
            
    except Exception as e:
        logger.error(
            "Error handling WebSocket message",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        await connection_manager.send_personal_message({
            'type': 'error',
            'data': {
                'message': 'Failed to process message',
                'error': str(e)
            }
        }, user_id)


async def handle_prompt_submission(user_id: str, data: dict):
    """Handle prompt submission from WebSocket client."""
    try:
        from ..core.langgraph_orchestrator import LangGraphOrchestrator
        
        prompt = data.get('prompt', '')
        context = data.get('context', {})
        complexity = data.get('complexity', 'moderate')
        
        # Create workflow
        orchestrator = LangGraphOrchestrator()
        workflow_id = await orchestrator.start_workflow(
            prompt=prompt,
            user_id=user_id,
            context=context,
            complexity=complexity
        )
        
        # Send confirmation to user
        await connection_manager.send_personal_message({
            'type': 'workflow_started',
            'data': {
                'workflow_id': workflow_id,
                'prompt': prompt,
                'complexity': complexity
            }
        }, user_id)
        
        # Start streaming workflow updates
        asyncio.create_task(
            stream_workflow_updates(workflow_id, user_id)
        )
        
    except Exception as e:
        logger.error(
            "Error handling prompt submission",
            user_id=user_id,
            error=str(e)
        )
        
        await connection_manager.send_personal_message({
            'type': 'error',
            'data': {
                'message': 'Failed to process prompt submission',
                'error': str(e)
            }
        }, user_id)


async def handle_agent_action(user_id: str, data: dict):
    """Handle agent action requests from WebSocket client."""
    try:
        action = data.get('action')
        agent_id = data.get('agent_id')
        parameters = data.get('parameters', {})
        
        # Process agent action
        from ..core.agent_framework import agent_registry
        
        if agent_id in agent_registry.agents:
            agent = agent_registry.agents[agent_id]
            result = await agent.handle_user_action(action, parameters)
            
            await connection_manager.send_personal_message({
                'type': 'agent_action_result',
                'data': {
                    'agent_id': agent_id,
                    'action': action,
                    'result': result
                }
            }, user_id)
        else:
            raise ValueError(f"Agent '{agent_id}' not found")
            
    except Exception as e:
        logger.error(
            "Error handling agent action",
            user_id=user_id,
            error=str(e)
        )
        
        await connection_manager.send_personal_message({
            'type': 'error',
            'data': {
                'message': 'Failed to process agent action',
                'error': str(e)
            }
        }, user_id)


async def handle_workflow_control(user_id: str, data: dict):
    """Handle workflow control commands from WebSocket client."""
    try:
        workflow_id = data.get('workflow_id')
        command = data.get('command')  # pause, resume, cancel
        
        from ..core.langgraph_orchestrator import LangGraphOrchestrator
        
        orchestrator = LangGraphOrchestrator()
        
        if command == 'pause':
            result = await orchestrator.pause_workflow(workflow_id)
        elif command == 'resume':
            result = await orchestrator.resume_workflow(workflow_id)
        elif command == 'cancel':
            result = await orchestrator.cancel_workflow(workflow_id)
        else:
            raise ValueError(f"Unknown workflow command: {command}")
        
        await connection_manager.send_personal_message({
            'type': 'workflow_control_result',
            'data': {
                'workflow_id': workflow_id,
                'command': command,
                'result': result
            }
        }, user_id)
        
    except Exception as e:
        logger.error(
            "Error handling workflow control",
            user_id=user_id,
            error=str(e)
        )
        
        await connection_manager.send_personal_message({
            'type': 'error',
            'data': {
                'message': 'Failed to process workflow control',
                'error': str(e)
            }
        }, user_id)


async def stream_workflow_updates(workflow_id: str, user_id: str):
    """Stream real-time workflow updates to a specific user."""
    try:
        from ..core.langgraph_orchestrator import LangGraphOrchestrator
        
        orchestrator = LangGraphOrchestrator()
        
        async for event in orchestrator.stream_workflow_events(workflow_id):
            await connection_manager.send_personal_message({
                'type': 'workflow_event',
                'data': {
                    'workflow_id': workflow_id,
                    'event': event
                }
            }, user_id)
            
    except Exception as e:
        logger.error(
            "Error streaming workflow updates",
            workflow_id=workflow_id,
            user_id=user_id,
            error=str(e)
        )