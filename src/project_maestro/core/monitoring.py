"""Comprehensive monitoring, metrics, and alerting system for Project Maestro."""

import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading

from .logging import get_logger
from .message_queue import publish_event, EventType

logger = get_logger("monitoring")


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    """Alert definition and state."""
    id: str
    name: str
    description: str
    level: AlertLevel
    condition: str  # Expression to evaluate
    threshold: float
    evaluation_window: int  # seconds
    cooldown_period: int  # seconds to wait before re-alerting
    last_triggered: Optional[datetime] = None
    is_active: bool = False
    trigger_count: int = 0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    process_count: int
    timestamp: datetime


@dataclass
class AgentMetrics:
    """Agent-specific metrics."""
    agent_name: str
    agent_type: str
    tasks_completed: int
    tasks_failed: int
    average_execution_time: float
    success_rate: float
    current_status: str
    last_activity: datetime
    memory_usage_mb: float
    cpu_usage_percent: float


class MetricsCollector:
    """Collects and stores metrics data."""
    
    def __init__(self, max_points: int = 10000):
        self.metrics_store = defaultdict(lambda: deque(maxlen=max_points))
        self.max_points = max_points
        self.lock = threading.Lock()
        
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric data point."""
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        with self.lock:
            self.metrics_store[name].append(point)
            
        logger.debug(
            "Recorded metric",
            name=name,
            value=value,
            metric_type=metric_type.value,
            tags=tags
        )
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record_metric(name, duration_ms, MetricType.TIMER, tags)
    
    def get_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Retrieve metrics for a given name and time range."""
        
        with self.lock:
            points = list(self.metrics_store.get(name, []))
            
        if start_time or end_time:
            filtered_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
            return filtered_points
            
        return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self.lock:
            points = self.metrics_store.get(name)
            if points:
                return points[-1].value
        return None
    
    def calculate_aggregate(
        self,
        name: str,
        aggregation: str,
        window_seconds: int = 300
    ) -> Optional[float]:
        """Calculate aggregated value over time window."""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=window_seconds)
        
        points = self.get_metrics(name, start_time, end_time)
        if not points:
            return None
            
        values = [point.value for point in points]
        
        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "count":
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_interval = 30  # seconds
        self.monitor_task = None
        
    async def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                
                # Record metrics
                self.metrics.set_gauge("system.cpu_percent", system_metrics.cpu_percent)
                self.metrics.set_gauge("system.memory_percent", system_metrics.memory_percent)
                self.metrics.set_gauge("system.disk_usage_percent", system_metrics.disk_usage_percent)
                self.metrics.set_gauge("system.network_io_bytes", system_metrics.network_io_bytes)
                self.metrics.set_gauge("system.process_count", system_metrics.process_count)
                
                # Check for system health issues
                await self._check_system_health(system_metrics)
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self.monitor_interval)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_bytes = network_io.bytes_sent + network_io.bytes_recv
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io_bytes=network_io_bytes,
            process_count=process_count,
            timestamp=datetime.now()
        )
    
    async def _check_system_health(self, metrics: SystemMetrics):
        """Check system health and trigger alerts if needed."""
        
        # CPU usage alert
        if metrics.cpu_percent > 80:
            await publish_event(
                EventType.AGENT_STATUS_CHANGED,
                "system_monitor",
                {
                    "alert_type": "high_cpu_usage",
                    "cpu_percent": metrics.cpu_percent,
                    "level": "warning" if metrics.cpu_percent < 90 else "error"
                }
            )
        
        # Memory usage alert
        if metrics.memory_percent > 85:
            await publish_event(
                EventType.AGENT_STATUS_CHANGED,
                "system_monitor",
                {
                    "alert_type": "high_memory_usage",
                    "memory_percent": metrics.memory_percent,
                    "level": "warning" if metrics.memory_percent < 95 else "critical"
                }
            )
        
        # Disk usage alert
        if metrics.disk_usage_percent > 90:
            await publish_event(
                EventType.AGENT_STATUS_CHANGED,
                "system_monitor",
                {
                    "alert_type": "high_disk_usage",
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "level": "error"
                }
            )


class AgentMonitor:
    """Monitors agent performance and health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.agent_stats = {}
        self.monitoring = False
        
    def record_agent_task_start(self, agent_name: str, task_id: str):
        """Record when an agent task starts."""
        self.metrics.increment_counter(
            "agent.tasks.started",
            tags={"agent": agent_name, "task_id": task_id}
        )
        
        # Store start time for duration calculation
        self.agent_stats[f"{agent_name}:{task_id}"] = {
            "start_time": time.time(),
            "agent_name": agent_name
        }
    
    def record_agent_task_completion(self, agent_name: str, task_id: str, success: bool):
        """Record when an agent task completes."""
        
        # Calculate execution time
        key = f"{agent_name}:{task_id}"
        task_info = self.agent_stats.get(key)
        
        if task_info:
            duration = (time.time() - task_info["start_time"]) * 1000  # ms
            self.metrics.record_timing(
                "agent.task.duration",
                duration,
                tags={"agent": agent_name, "success": str(success)}
            )
            del self.agent_stats[key]
        
        # Record completion
        counter_name = "agent.tasks.completed" if success else "agent.tasks.failed"
        self.metrics.increment_counter(counter_name, tags={"agent": agent_name})
        
    def record_agent_error(self, agent_name: str, error_type: str, error_message: str):
        """Record an agent error."""
        self.metrics.increment_counter(
            "agent.errors",
            tags={
                "agent": agent_name,
                "error_type": error_type,
                "error_message": error_message[:100]  # Truncate long messages
            }
        )
    
    def get_agent_metrics(self, agent_name: str, window_seconds: int = 3600) -> AgentMetrics:
        """Get aggregated metrics for an agent."""
        
        # Calculate metrics over the time window
        completed = self.metrics.calculate_aggregate(
            "agent.tasks.completed", "count", window_seconds
        ) or 0
        
        failed = self.metrics.calculate_aggregate(
            "agent.tasks.failed", "count", window_seconds
        ) or 0
        
        avg_duration = self.metrics.calculate_aggregate(
            "agent.task.duration", "avg", window_seconds
        ) or 0
        
        success_rate = completed / (completed + failed) if (completed + failed) > 0 else 1.0
        
        return AgentMetrics(
            agent_name=agent_name,
            agent_type="unknown",  # Would need to be provided separately
            tasks_completed=int(completed),
            tasks_failed=int(failed),
            average_execution_time=avg_duration,
            success_rate=success_rate,
            current_status="active",
            last_activity=datetime.now(),
            memory_usage_mb=0.0,  # Would need process-specific monitoring
            cpu_usage_percent=0.0
        )


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts = {}
        self.alert_handlers = []
        self.monitoring = False
        self.check_interval = 60  # seconds
        
    def add_alert(self, alert: Alert):
        """Add an alert to be monitored."""
        self.alerts[alert.id] = alert
        logger.info("Added alert", alert_id=alert.id, name=alert.name)
    
    def add_alert_handler(self, handler: Callable[[Alert, Dict[str, Any]], None]):
        """Add a handler function for triggered alerts."""
        self.alert_handlers.append(handler)
    
    async def start_monitoring(self):
        """Start alert monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        asyncio.create_task(self._monitor_alerts())
        logger.info("Alert monitoring started")
    
    async def stop_monitoring(self):
        """Stop alert monitoring."""
        self.monitoring = False
        logger.info("Alert monitoring stopped")
    
    async def _monitor_alerts(self):
        """Monitor alerts and trigger when conditions are met."""
        
        while self.monitoring:
            try:
                for alert in self.alerts.values():
                    await self._evaluate_alert(alert)
                    
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("Error in alert monitoring", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _evaluate_alert(self, alert: Alert):
        """Evaluate a single alert condition."""
        
        try:
            # Skip if in cooldown period
            if alert.last_triggered:
                time_since_trigger = datetime.now() - alert.last_triggered
                if time_since_trigger.total_seconds() < alert.cooldown_period:
                    return
            
            # Evaluate the alert condition
            triggered = await self._evaluate_condition(alert)
            
            if triggered and not alert.is_active:
                # Alert just triggered
                alert.is_active = True
                alert.last_triggered = datetime.now()
                alert.trigger_count += 1
                
                await self._trigger_alert(alert)
                
            elif not triggered and alert.is_active:
                # Alert resolved
                alert.is_active = False
                await self._resolve_alert(alert)
                
        except Exception as e:
            logger.error(
                "Error evaluating alert",
                alert_id=alert.id,
                error=str(e)
            )
    
    async def _evaluate_condition(self, alert: Alert) -> bool:
        """Evaluate an alert condition expression."""
        
        # Simple threshold-based evaluation for now
        # In a full implementation, this could parse complex expressions
        
        if ">" in alert.condition:
            metric_name, threshold_str = alert.condition.split(">")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())
            
            current_value = self.metrics.get_latest_value(metric_name)
            if current_value is not None:
                return current_value > threshold
                
        elif "<" in alert.condition:
            metric_name, threshold_str = alert.condition.split("<")
            metric_name = metric_name.strip()
            threshold = float(threshold_str.strip())
            
            current_value = self.metrics.get_latest_value(metric_name)
            if current_value is not None:
                return current_value < threshold
        
        return False
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        
        logger.warning(
            "Alert triggered",
            alert_id=alert.id,
            name=alert.name,
            level=alert.level.value,
            trigger_count=alert.trigger_count
        )
        
        # Publish alert event
        await publish_event(
            EventType.AGENT_STATUS_CHANGED,
            "alert_manager",
            {
                "alert_id": alert.id,
                "alert_name": alert.name,
                "level": alert.level.value,
                "description": alert.description,
                "condition": alert.condition,
                "status": "triggered"
            }
        )
        
        # Call alert handlers
        alert_data = {
            "alert": alert,
            "timestamp": datetime.now(),
            "status": "triggered"
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert, alert_data)
            except Exception as e:
                logger.error("Error in alert handler", error=str(e))
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        
        logger.info(
            "Alert resolved",
            alert_id=alert.id,
            name=alert.name
        )
        
        # Publish resolution event
        await publish_event(
            EventType.AGENT_STATUS_CHANGED,
            "alert_manager",
            {
                "alert_id": alert.id,
                "alert_name": alert.name,
                "level": alert.level.value,
                "status": "resolved"
            }
        )


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.checks = {}
        
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function."""
        self.checks[name] = check_func
        
    async def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks and return results."""
        
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                response_time = (time.time() - start_time) * 1000
                
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
        return results


# Global monitoring instances
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)
agent_monitor = AgentMonitor(metrics_collector)
alert_manager = AlertManager(metrics_collector)
health_checker = HealthChecker()


def setup_default_alerts():
    """Set up default system alerts."""
    
    # High CPU usage alert
    alert_manager.add_alert(Alert(
        id="high_cpu_usage",
        name="High CPU Usage",
        description="CPU usage is above 80%",
        level=AlertLevel.WARNING,
        condition="system.cpu_percent > 80",
        threshold=80.0,
        evaluation_window=300,
        cooldown_period=600
    ))
    
    # High memory usage alert
    alert_manager.add_alert(Alert(
        id="high_memory_usage",
        name="High Memory Usage",
        description="Memory usage is above 85%",
        level=AlertLevel.ERROR,
        condition="system.memory_percent > 85",
        threshold=85.0,
        evaluation_window=300,
        cooldown_period=600
    ))
    
    # Agent failure rate alert
    alert_manager.add_alert(Alert(
        id="high_agent_failure_rate",
        name="High Agent Failure Rate",
        description="Agent failure rate is above 10%",
        level=AlertLevel.WARNING,
        condition="agent.failure_rate > 0.1",
        threshold=0.1,
        evaluation_window=600,
        cooldown_period=900
    ))


async def start_monitoring():
    """Start all monitoring services."""
    setup_default_alerts()
    await system_monitor.start_monitoring()
    await alert_manager.start_monitoring()
    logger.info("All monitoring services started")


async def stop_monitoring():
    """Stop all monitoring services."""
    await system_monitor.stop_monitoring()
    await alert_manager.stop_monitoring()
    logger.info("All monitoring services stopped")