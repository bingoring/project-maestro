"""Structured logging configuration for Project Maestro."""

import logging
import sys
from typing import Any, Dict, Optional
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Configure structured logging with appropriate processors."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Define common processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]
    
    if settings.environment == "development":
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create application logger
    logger = structlog.get_logger("project_maestro")
    
    # Log startup information
    logger.info(
        "Logging configured",
        environment=settings.environment,
        log_level=settings.log_level,
        debug=settings.debug,
    )
    
    return logger


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)


class AgentLoggerMixin:
    """Mixin class to provide consistent logging for agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
        
    def log_agent_action(
        self, 
        action: str, 
        status: str = "started",
        **kwargs: Any
    ) -> None:
        """Log agent actions with consistent format."""
        self.logger.info(
            f"Agent {action}",
            agent=self.__class__.__name__,
            action=action,
            status=status,
            **kwargs
        )
        
    def log_agent_error(
        self,
        action: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Log agent errors with consistent format."""
        self.logger.error(
            f"Agent {action} failed",
            agent=self.__class__.__name__,
            action=action,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs,
            exc_info=True
        )
        
    def log_agent_success(
        self,
        action: str,
        result: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log successful agent actions."""
        log_data = {
            "agent": self.__class__.__name__,
            "action": action,
            "status": "completed",
            **kwargs
        }
        
        if result:
            log_data["result"] = result
            
        self.logger.info(f"Agent {action} completed", **log_data)


class TaskLoggerMixin:
    """Mixin class to provide consistent logging for tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"task.{self.__class__.__name__}")
        
    def log_task_start(
        self, 
        task_id: str,
        task_type: str,
        **kwargs: Any
    ) -> None:
        """Log task start with consistent format."""
        self.logger.info(
            "Task started",
            task_id=task_id,
            task_type=task_type,
            **kwargs
        )
        
    def log_task_progress(
        self,
        task_id: str,
        progress: float,
        message: str,
        **kwargs: Any
    ) -> None:
        """Log task progress updates."""
        self.logger.info(
            "Task progress",
            task_id=task_id,
            progress=progress,
            message=message,
            **kwargs
        )
        
    def log_task_complete(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log task completion."""
        log_data = {
            "task_id": task_id,
            "status": "completed",
            **kwargs
        }
        
        if result:
            log_data["result"] = result
            
        self.logger.info("Task completed", **log_data)
        
    def log_task_error(
        self,
        task_id: str,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Log task errors."""
        self.logger.error(
            "Task failed",
            task_id=task_id,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs,
            exc_info=True
        )


# Initialize logger
logger = configure_logging()