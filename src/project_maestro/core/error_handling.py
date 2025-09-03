"""Comprehensive error handling and recovery system for Project Maestro."""

import asyncio
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
import functools
import threading
from collections import defaultdict

from .logging import get_logger
from .message_queue import publish_event, EventType
from .monitoring import metrics_collector, agent_monitor

logger = get_logger("error_handling")


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    CUSTOM = "custom"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    component: str
    operation: str
    parameters: Dict[str, Any]
    severity: ErrorSeverity
    category: ErrorCategory
    recovery_strategy: RecoveryStrategy
    attempt_count: int = 0
    max_attempts: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempt_number: int
    recovery_time_ms: float
    error_message: Optional[str] = None
    fallback_data: Optional[Any] = None


class ErrorPattern:
    """Defines patterns for error classification and recovery."""
    
    def __init__(
        self,
        name: str,
        pattern: Union[str, Type[Exception]],
        category: ErrorCategory,
        severity: ErrorSeverity,
        recovery_strategy: RecoveryStrategy,
        max_attempts: int = 3,
        backoff_multiplier: float = 1.5,
        custom_handler: Optional[Callable] = None
    ):
        self.name = name
        self.pattern = pattern
        self.category = category
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.max_attempts = max_attempts
        self.backoff_multiplier = backoff_multiplier
        self.custom_handler = custom_handler
        
    def matches(self, error: Exception) -> bool:
        """Check if this pattern matches the given error."""
        if isinstance(self.pattern, type):
            return isinstance(error, self.pattern)
        elif isinstance(self.pattern, str):
            return self.pattern.lower() in str(error).lower()
        return False


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
            
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.timeout_seconds
    
    def _on_success(self):
        """Reset circuit breaker on successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure in circuit breaker."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    "Circuit breaker opened",
                    name=self.name,
                    failure_count=self.failure_count
                )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


class ErrorRecoveryEngine:
    """Main engine for error handling and recovery."""
    
    def __init__(self):
        self.error_patterns = []
        self.circuit_breakers = {}
        self.error_history = defaultdict(list)
        self.recovery_handlers = {}
        self.fallback_handlers = {}
        
        # Default error patterns
        self._setup_default_patterns()
        
    def _setup_default_patterns(self):
        """Set up default error patterns."""
        
        # Network errors
        self.add_error_pattern(ErrorPattern(
            name="connection_error",
            pattern=ConnectionError,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_attempts=3
        ))
        
        # Timeout errors
        self.add_error_pattern(ErrorPattern(
            name="timeout_error",
            pattern=TimeoutError,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_attempts=2
        ))
        
        # Resource errors
        self.add_error_pattern(ErrorPattern(
            name="memory_error",
            pattern=MemoryError,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            max_attempts=1
        ))
        
        # Validation errors
        self.add_error_pattern(ErrorPattern(
            name="validation_error",
            pattern=ValueError,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            max_attempts=1
        ))
        
    def add_error_pattern(self, pattern: ErrorPattern):
        """Add an error pattern for classification."""
        self.error_patterns.append(pattern)
        logger.debug("Added error pattern", pattern_name=pattern.name)
        
    def add_recovery_handler(
        self,
        strategy: RecoveryStrategy,
        handler: Callable[[ErrorContext], RecoveryResult]
    ):
        """Add a custom recovery handler."""
        self.recovery_handlers[strategy] = handler
        
    def add_fallback_handler(
        self,
        operation: str,
        handler: Callable[[ErrorContext], Any]
    ):
        """Add a fallback handler for specific operations."""
        self.fallback_handlers[operation] = handler
        
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name)
        return self.circuit_breakers[name]
        
    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        attempt_count: int = 0
    ) -> RecoveryResult:
        """Handle an error with appropriate recovery strategy."""
        
        # Create error context
        error_context = self._create_error_context(
            error, component, operation, parameters, attempt_count
        )
        
        # Classify error
        pattern = self._classify_error(error)
        if pattern:
            error_context.category = pattern.category
            error_context.severity = pattern.severity
            error_context.recovery_strategy = pattern.recovery_strategy
            error_context.max_attempts = pattern.max_attempts
            
        # Log error
        await self._log_error(error_context)
        
        # Record metrics
        agent_monitor.record_agent_error(
            component,
            error_context.category.value,
            error_context.error_message
        )
        
        # Store in history
        self.error_history[component].append(error_context)
        
        # Attempt recovery
        recovery_result = await self._attempt_recovery(error_context)
        
        # Publish error event
        await self._publish_error_event(error_context, recovery_result)
        
        return recovery_result
        
    def _create_error_context(
        self,
        error: Exception,
        component: str,
        operation: str,
        parameters: Optional[Dict[str, Any]],
        attempt_count: int
    ) -> ErrorContext:
        """Create error context from exception."""
        
        return ErrorContext(
            error_id=f"{component}:{operation}:{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            operation=operation,
            parameters=parameters or {},
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            recovery_strategy=RecoveryStrategy.RETRY,
            attempt_count=attempt_count
        )
        
    def _classify_error(self, error: Exception) -> Optional[ErrorPattern]:
        """Classify error using registered patterns."""
        
        for pattern in self.error_patterns:
            if pattern.matches(error):
                return pattern
                
        return None
        
    async def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        
        log_data = {
            "error_id": error_context.error_id,
            "component": error_context.component,
            "operation": error_context.operation,
            "error_type": error_context.error_type,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "attempt": error_context.attempt_count
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data, stack_trace=error_context.stack_trace)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error("High severity error", **log_data, error_message=error_context.error_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error", **log_data, error_message=error_context.error_message)
        else:
            logger.info("Low severity error", **log_data, error_message=error_context.error_message)
            
    async def _attempt_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt to recover from error using configured strategy."""
        
        start_time = datetime.now()
        
        try:
            if error_context.recovery_strategy == RecoveryStrategy.RETRY:
                return await self._retry_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.CUSTOM:
                return await self._custom_recovery(error_context)
            elif error_context.recovery_strategy == RecoveryStrategy.ESCALATE:
                return await self._escalate_recovery(error_context)
            else:  # IGNORE
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.IGNORE,
                    attempt_number=error_context.attempt_count,
                    recovery_time_ms=0.0,
                    error_message="Error ignored by strategy"
                )
                
        except Exception as e:
            recovery_time = (datetime.now() - start_time).total_seconds() * 1000
            return RecoveryResult(
                success=False,
                strategy_used=error_context.recovery_strategy,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=recovery_time,
                error_message=str(e)
            )
            
    async def _retry_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through retry."""
        
        if error_context.attempt_count >= error_context.max_attempts:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=0.0,
                error_message="Max retry attempts exceeded"
            )
            
        # Calculate delay with exponential backoff
        delay = min(1.0 * (1.5 ** error_context.attempt_count), 30.0)
        await asyncio.sleep(delay)
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            attempt_number=error_context.attempt_count + 1,
            recovery_time_ms=delay * 1000
        )
        
    async def _fallback_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through fallback."""
        
        fallback_handler = self.fallback_handlers.get(error_context.operation)
        
        if fallback_handler:
            try:
                fallback_data = await fallback_handler(error_context) if asyncio.iscoroutinefunction(fallback_handler) else fallback_handler(error_context)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    attempt_number=error_context.attempt_count,
                    recovery_time_ms=0.0,
                    fallback_data=fallback_data
                )
            except Exception as e:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    attempt_number=error_context.attempt_count,
                    recovery_time_ms=0.0,
                    error_message=f"Fallback failed: {str(e)}"
                )
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=0.0,
                error_message="No fallback handler available"
            )
            
    async def _circuit_breaker_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through circuit breaker."""
        
        circuit_breaker = self.get_circuit_breaker(error_context.component)
        
        if circuit_breaker.state == "OPEN":
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=0.0,
                error_message="Circuit breaker is open"
            )
        else:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=0.0
            )
            
    async def _custom_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery through custom handler."""
        
        handler = self.recovery_handlers.get(RecoveryStrategy.CUSTOM)
        
        if handler:
            return await handler(error_context) if asyncio.iscoroutinefunction(handler) else handler(error_context)
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CUSTOM,
                attempt_number=error_context.attempt_count,
                recovery_time_ms=0.0,
                error_message="No custom recovery handler available"
            )
            
    async def _escalate_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Escalate error to higher-level handling."""
        
        # In a real implementation, this might notify administrators,
        # trigger emergency procedures, etc.
        
        logger.critical(
            "Error escalated",
            error_id=error_context.error_id,
            component=error_context.component,
            operation=error_context.operation
        )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            attempt_number=error_context.attempt_count,
            recovery_time_ms=0.0,
            error_message="Error escalated for manual intervention"
        )
        
    async def _publish_error_event(
        self,
        error_context: ErrorContext,
        recovery_result: RecoveryResult
    ):
        """Publish error event to message queue."""
        
        await publish_event(
            EventType.TASK_FAILED if not recovery_result.success else EventType.TASK_COMPLETED,
            error_context.component,
            {
                "error_id": error_context.error_id,
                "error_type": error_context.error_type,
                "category": error_context.category.value,
                "severity": error_context.severity.value,
                "recovery_strategy": error_context.recovery_strategy.value,
                "recovery_success": recovery_result.success,
                "attempt_count": error_context.attempt_count,
                "operation": error_context.operation
            }
        )
        
    def get_error_statistics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics for analysis."""
        
        stats = {
            "total_errors": 0,
            "errors_by_category": defaultdict(int),
            "errors_by_severity": defaultdict(int),
            "recovery_success_rate": 0.0,
            "most_common_errors": [],
            "components_with_errors": []
        }
        
        all_errors = []
        if component:
            all_errors = self.error_history.get(component, [])
        else:
            for component_errors in self.error_history.values():
                all_errors.extend(component_errors)
                
        if not all_errors:
            return stats
            
        stats["total_errors"] = len(all_errors)
        
        # Count by category and severity
        for error in all_errors:
            stats["errors_by_category"][error.category.value] += 1
            stats["errors_by_severity"][error.severity.value] += 1
            
        # Component statistics
        stats["components_with_errors"] = list(self.error_history.keys())
        
        return stats


def with_error_handling(
    component: str,
    operation: str,
    max_attempts: int = 3,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
):
    """Decorator for adding error handling to functions."""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed, handle error
                        recovery_result = await error_recovery_engine.handle_error(
                            e, component, operation, {"args": args, "kwargs": kwargs}, attempt
                        )
                        
                        if not recovery_result.success:
                            raise e
                            
                        if recovery_result.fallback_data is not None:
                            return recovery_result.fallback_data
                        elif recovery_result.strategy_used == RecoveryStrategy.RETRY:
                            continue
                        else:
                            raise e
                    else:
                        # Not last attempt, just record the error
                        await error_recovery_engine.handle_error(
                            e, component, operation, {"args": args, "kwargs": kwargs}, attempt
                        )
                        
                        # Wait before retry
                        delay = min(1.0 * (1.5 ** attempt), 30.0)
                        await asyncio.sleep(delay)
                        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, create a simple wrapper
            # In a full implementation, you might want to use threading
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error synchronously
                logger.error(
                    "Synchronous function error",
                    component=component,
                    operation=operation,
                    error=str(e)
                )
                raise
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator


# Global error recovery engine
error_recovery_engine = ErrorRecoveryEngine()


def setup_default_fallbacks():
    """Set up default fallback handlers."""
    
    # Default fallback for data generation
    def default_data_fallback(error_context: ErrorContext) -> Dict[str, Any]:
        return {
            "status": "fallback",
            "message": "Using fallback data due to error",
            "error_id": error_context.error_id
        }
    
    error_recovery_engine.add_fallback_handler("generate_data", default_data_fallback)
    
    # Default fallback for asset generation
    def asset_generation_fallback(error_context: ErrorContext) -> Dict[str, Any]:
        return {
            "asset_url": "/fallback/placeholder.png",
            "asset_type": "placeholder",
            "message": "Using placeholder asset",
            "error_id": error_context.error_id
        }
    
    error_recovery_engine.add_fallback_handler("generate_asset", asset_generation_fallback)


# Initialize default fallbacks
setup_default_fallbacks()