"""Event-driven message queue system for Project Maestro."""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel

from .config import settings
from .logging import get_logger


class EventType(str, Enum):
    """Types of events in the system."""
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    PROJECT_CREATED = "project.created"
    PROJECT_COMPLETED = "project.completed"
    BUILD_STARTED = "build.started"
    BUILD_COMPLETED = "build.completed"
    ASSET_GENERATED = "asset.generated"


@dataclass
class Event:
    """Base event structure."""
    id: str
    type: EventType
    source: str  # Agent or service that created the event
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For tracking related events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            source=data["source"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id")
        )


class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass
        
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if this handler can process the event type."""
        pass


class MessageQueue(ABC):
    """Abstract base class for message queue implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the message queue."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message queue."""
        pass
        
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event."""
        pass
        
    @abstractmethod
    async def subscribe(
        self, 
        event_types: List[EventType],
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe to event types."""
        pass
        
    @abstractmethod
    async def create_queue(self, name: str) -> None:
        """Create a named queue."""
        pass


class RedisMessageQueue(MessageQueue):
    """Redis-based message queue implementation."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = get_logger("message_queue")
        self._subscription_tasks: List[asyncio.Task] = []
        
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("Connected to Redis message queue")
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            # Cancel subscription tasks
            for task in self._subscription_tasks:
                task.cancel()
            
            try:
                await asyncio.gather(*self._subscription_tasks, return_exceptions=True)
            except Exception:
                pass  # Tasks were cancelled
                
            await self.redis_client.close()
            self.redis_client = None
            self.logger.info("Disconnected from Redis message queue")
            
    async def publish(self, event: Event) -> None:
        """Publish an event to Redis."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Publish to specific event type channel
            channel = f"events:{event.type.value}"
            message = json.dumps(event.to_dict())
            
            await self.redis_client.publish(channel, message)
            
            # Also publish to general events channel
            await self.redis_client.publish("events:all", message)
            
            self.logger.debug(
                "Published event",
                event_id=event.id,
                event_type=event.type.value,
                channel=channel
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to publish event",
                event_id=event.id,
                error=str(e)
            )
            raise
            
    async def subscribe(
        self, 
        event_types: List[EventType],
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe to event types."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
            
        # Create channels list
        channels = [f"events:{event_type.value}" for event_type in event_types]
        
        # Create subscription task
        task = asyncio.create_task(
            self._subscription_worker(channels, handler)
        )
        self._subscription_tasks.append(task)
        
        self.logger.info(
            "Created subscription",
            channels=channels,
            handler=handler.__name__ if hasattr(handler, '__name__') else str(handler)
        )
        
    async def _subscription_worker(
        self, 
        channels: List[str],
        handler: Callable[[Event], None]
    ) -> None:
        """Worker for handling subscriptions."""
        pubsub = self.redis_client.pubsub()
        
        try:
            # Subscribe to channels
            for channel in channels:
                await pubsub.subscribe(channel)
                
            self.logger.info("Subscription worker started", channels=channels)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse event
                        event_data = json.loads(message['data'])
                        event = Event.from_dict(event_data)
                        
                        # Handle event
                        await handler(event)
                        
                    except Exception as e:
                        self.logger.error(
                            "Error handling event",
                            channel=message['channel'],
                            error=str(e)
                        )
                        
        except asyncio.CancelledError:
            self.logger.info("Subscription worker cancelled", channels=channels)
        except Exception as e:
            self.logger.error(
                "Subscription worker error",
                channels=channels,
                error=str(e)
            )
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
            
    async def create_queue(self, name: str) -> None:
        """Create a named queue (Redis List)."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
            
        # Redis doesn't require explicit queue creation
        # Lists are created automatically when first item is pushed
        self.logger.info("Queue created", name=name)


class EventBus:
    """Central event bus for managing events and handlers."""
    
    def __init__(self, message_queue: MessageQueue):
        self.message_queue = message_queue
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.logger = get_logger("event_bus")
        
    async def initialize(self) -> None:
        """Initialize the event bus."""
        await self.message_queue.connect()
        self.logger.info("Event bus initialized")
        
    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        await self.message_queue.disconnect()
        self.logger.info("Event bus shutdown")
        
    def register_handler(
        self, 
        event_types: List[EventType],
        handler: EventHandler
    ) -> None:
        """Register an event handler."""
        for event_type in event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
        self.logger.info(
            "Handler registered",
            event_types=[et.value for et in event_types],
            handler=handler.__class__.__name__
        )
        
    async def publish_event(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """Publish an event."""
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            source=source,
            data=data,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        await self.message_queue.publish(event)
        
        self.logger.info(
            "Event published",
            event_id=event.id,
            event_type=event_type.value,
            source=source
        )
        
        return event.id
        
    async def start_listening(self) -> None:
        """Start listening for events."""
        # Subscribe to all event types that have handlers
        if self.handlers:
            event_types = list(self.handlers.keys())
            await self.message_queue.subscribe(
                event_types,
                self._handle_event
            )
            
        self.logger.info(
            "Started listening for events",
            event_types=[et.value for et in self.handlers.keys()]
        )
        
    async def _handle_event(self, event: Event) -> None:
        """Internal event handler."""
        handlers = self.handlers.get(event.type, [])
        
        self.logger.debug(
            "Handling event",
            event_id=event.id,
            event_type=event.type.value,
            handlers_count=len(handlers)
        )
        
        # Execute all handlers for this event type
        for handler in handlers:
            try:
                if handler.can_handle(event.type):
                    await handler.handle(event)
            except Exception as e:
                self.logger.error(
                    "Handler error",
                    event_id=event.id,
                    handler=handler.__class__.__name__,
                    error=str(e)
                )


# Global event bus instance
_event_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        message_queue = RedisMessageQueue()
        _event_bus = EventBus(message_queue)
        await _event_bus.initialize()
    return _event_bus


async def publish_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> str:
    """Convenience function to publish an event."""
    event_bus = await get_event_bus()
    return await event_bus.publish_event(event_type, source, data, correlation_id)


def create_event_handler(
    event_types: List[EventType],
    handler_func: Callable[[Event], None]
) -> EventHandler:
    """Create an event handler from a function."""
    
    class FunctionEventHandler(EventHandler):
        def __init__(self):
            self.event_types = event_types
            self.handler_func = handler_func
            
        async def handle(self, event: Event) -> None:
            if asyncio.iscoroutinefunction(self.handler_func):
                await self.handler_func(event)
            else:
                self.handler_func(event)
                
        def can_handle(self, event_type: EventType) -> bool:
            return event_type in self.event_types
            
    return FunctionEventHandler()