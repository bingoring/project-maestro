"""Unit tests for monitoring system."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from src.project_maestro.core.monitoring import (
    MetricType, AlertLevel, MetricPoint, Alert, SystemMetrics, AgentMetrics,
    MetricsCollector, SystemMonitor, AgentMonitor, AlertManager, HealthChecker
)


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector(max_points=1000)
        
        assert collector.max_points == 1000
        assert len(collector.metrics_store) == 0
        
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("test.counter", 5.0, MetricType.COUNTER, {"tag": "value"})
        
        metrics = collector.get_metrics("test.counter")
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.name == "test.counter"
        assert metric.value == 5.0
        assert metric.metric_type == MetricType.COUNTER
        assert metric.tags == {"tag": "value"}
        assert isinstance(metric.timestamp, datetime)
        
    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment_counter("test.counter", 3.0, {"env": "test"})
        collector.increment_counter("test.counter", 2.0, {"env": "test"})
        
        metrics = collector.get_metrics("test.counter")
        assert len(metrics) == 2
        assert metrics[0].value == 3.0
        assert metrics[1].value == 2.0
        
    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        
        collector.set_gauge("test.gauge", 42.5, {"host": "server1"})
        
        latest_value = collector.get_latest_value("test.gauge")
        assert latest_value == 42.5
        
    def test_record_timing(self):
        """Test timing recording."""
        collector = MetricsCollector()
        
        collector.record_timing("test.timing", 123.45, {"operation": "api_call"})
        
        metrics = collector.get_metrics("test.timing")
        assert len(metrics) == 1
        assert metrics[0].value == 123.45
        assert metrics[0].metric_type == MetricType.TIMER
        
    def test_get_latest_value(self):
        """Test getting latest metric value."""
        collector = MetricsCollector()
        
        # No metrics recorded
        assert collector.get_latest_value("nonexistent") is None
        
        # Record multiple values
        collector.set_gauge("temperature", 20.0)
        collector.set_gauge("temperature", 25.0)
        collector.set_gauge("temperature", 22.5)
        
        assert collector.get_latest_value("temperature") == 22.5
        
    def test_time_range_filtering(self):
        """Test filtering metrics by time range."""
        collector = MetricsCollector()
        
        now = datetime.now()
        
        # Manually create metrics with specific timestamps
        old_metric = MetricPoint("test.metric", 1.0, now - timedelta(hours=2), {}, MetricType.GAUGE)
        recent_metric = MetricPoint("test.metric", 2.0, now - timedelta(minutes=30), {}, MetricType.GAUGE)
        new_metric = MetricPoint("test.metric", 3.0, now, {}, MetricType.GAUGE)
        
        collector.metrics_store["test.metric"].extend([old_metric, recent_metric, new_metric])
        
        # Get metrics from last hour
        start_time = now - timedelta(hours=1)
        filtered_metrics = collector.get_metrics("test.metric", start_time=start_time)
        
        assert len(filtered_metrics) == 2  # recent_metric and new_metric
        assert filtered_metrics[0].value == 2.0
        assert filtered_metrics[1].value == 3.0
        
    def test_calculate_aggregate(self):
        """Test aggregate calculations."""
        collector = MetricsCollector()
        
        # Record some values
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            collector.set_gauge("test.aggregate", value)
            
        # Test different aggregations
        assert collector.calculate_aggregate("test.aggregate", "avg", 3600) == 30.0  # Average
        assert collector.calculate_aggregate("test.aggregate", "min", 3600) == 10.0  # Minimum
        assert collector.calculate_aggregate("test.aggregate", "max", 3600) == 50.0  # Maximum
        assert collector.calculate_aggregate("test.aggregate", "sum", 3600) == 150.0  # Sum
        assert collector.calculate_aggregate("test.aggregate", "count", 3600) == 5  # Count
        
        # Test with no data
        assert collector.calculate_aggregate("nonexistent", "avg", 3600) is None
        
        # Test invalid aggregation
        with pytest.raises(ValueError):
            collector.calculate_aggregate("test.aggregate", "invalid", 3600)


class TestSystemMonitor:
    """Test system monitor functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        assert monitor.metrics == collector
        assert monitor.monitoring is False
        assert monitor.monitor_interval == 30
        assert monitor.monitor_task is None
        
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor.monitor_task is not None
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor.monitoring is False
        
    def test_collect_system_metrics(self):
        """Test system metrics collection."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        with patch('src.project_maestro.core.monitoring.psutil') as mock_psutil:
            # Mock psutil calls
            mock_psutil.cpu_percent.return_value = 45.5
            mock_psutil.virtual_memory.return_value = MagicMock(percent=67.8)
            mock_psutil.disk_usage.return_value = MagicMock(used=50, total=100)
            mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
            mock_psutil.pids.return_value = list(range(150))
            
            metrics = monitor.collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 45.5
            assert metrics.memory_percent == 67.8
            assert metrics.disk_usage_percent == 50.0
            assert metrics.network_io_bytes == 3000
            assert metrics.process_count == 150
            assert isinstance(metrics.timestamp, datetime)
            
    @pytest.mark.asyncio
    async def test_health_check_alerts(self):
        """Test system health check and alerting."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        # Mock high CPU usage
        high_cpu_metrics = SystemMetrics(
            cpu_percent=95.0,
            memory_percent=50.0,
            disk_usage_percent=30.0,
            network_io_bytes=1000,
            process_count=100,
            timestamp=datetime.now()
        )
        
        with patch('src.project_maestro.core.monitoring.publish_event') as mock_publish:
            await monitor._check_system_health(high_cpu_metrics)
            
            # Should publish high CPU alert
            mock_publish.assert_called()
            call_args = mock_publish.call_args
            assert call_args[0][1] == "system_monitor"  # source
            assert call_args[0][2]["alert_type"] == "high_cpu_usage"
            assert call_args[0][2]["level"] == "error"  # >90% is error level


class TestAgentMonitor:
    """Test agent monitor functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        collector = MetricsCollector()
        monitor = AgentMonitor(collector)
        
        assert monitor.metrics == collector
        assert monitor.agent_stats == {}
        assert monitor.monitoring is False
        
    def test_record_task_lifecycle(self):
        """Test recording agent task lifecycle."""
        collector = MetricsCollector()
        monitor = AgentMonitor(collector)
        
        # Record task start
        monitor.record_agent_task_start("test_agent", "task_001")
        
        assert "test_agent:task_001" in monitor.agent_stats
        task_info = monitor.agent_stats["test_agent:task_001"]
        assert "start_time" in task_info
        assert task_info["agent_name"] == "test_agent"
        
        # Record successful completion
        monitor.record_agent_task_completion("test_agent", "task_001", True)
        
        # Task should be removed from active stats
        assert "test_agent:task_001" not in monitor.agent_stats
        
        # Should have recorded metrics
        completed_metrics = collector.get_metrics("agent.tasks.completed")
        timing_metrics = collector.get_metrics("agent.task.duration")
        
        assert len(completed_metrics) == 1
        assert len(timing_metrics) == 1
        
    def test_record_agent_error(self):
        """Test recording agent errors."""
        collector = MetricsCollector()
        monitor = AgentMonitor(collector)
        
        monitor.record_agent_error("test_agent", "ValueError", "Invalid input provided")
        
        error_metrics = collector.get_metrics("agent.errors")
        assert len(error_metrics) == 1
        
        error_metric = error_metrics[0]
        assert error_metric.tags["agent"] == "test_agent"
        assert error_metric.tags["error_type"] == "ValueError"
        
    def test_get_agent_metrics(self):
        """Test getting aggregated agent metrics."""
        collector = MetricsCollector()
        monitor = AgentMonitor(collector)
        
        # Simulate some agent activity
        collector.increment_counter("agent.tasks.completed", 5, {"agent": "test_agent"})
        collector.increment_counter("agent.tasks.failed", 2, {"agent": "test_agent"})
        collector.record_timing("agent.task.duration", 150.0, {"agent": "test_agent"})
        collector.record_timing("agent.task.duration", 200.0, {"agent": "test_agent"})
        collector.record_timing("agent.task.duration", 100.0, {"agent": "test_agent"})
        
        # Get agent metrics (this will aggregate across all recorded metrics)
        with patch.object(collector, 'calculate_aggregate') as mock_aggregate:
            mock_aggregate.side_effect = lambda name, agg, window: {
                "agent.tasks.completed": 5.0 if agg == "count" else 0,
                "agent.tasks.failed": 2.0 if agg == "count" else 0,
                "agent.task.duration": 150.0 if agg == "avg" else 0
            }.get(name, 0)
            
            metrics = monitor.get_agent_metrics("test_agent")
            
            assert isinstance(metrics, AgentMetrics)
            assert metrics.agent_name == "test_agent"
            assert metrics.tasks_completed == 5
            assert metrics.tasks_failed == 2
            assert metrics.average_execution_time == 150.0
            assert metrics.success_rate == 5/(5+2)


class TestAlertManager:
    """Test alert manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        assert manager.metrics == collector
        assert manager.alerts == {}
        assert manager.alert_handlers == []
        assert manager.monitoring is False
        
    def test_add_alert(self):
        """Test adding alerts."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        alert = Alert(
            id="test_alert",
            name="Test Alert",
            description="Test alert description",
            level=AlertLevel.WARNING,
            condition="cpu_usage > 80",
            threshold=80.0,
            evaluation_window=300,
            cooldown_period=600
        )
        
        manager.add_alert(alert)
        
        assert "test_alert" in manager.alerts
        assert manager.alerts["test_alert"] == alert
        
    def test_add_alert_handler(self):
        """Test adding alert handlers."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        def test_handler(alert, data):
            pass
            
        manager.add_alert_handler(test_handler)
        
        assert test_handler in manager.alert_handlers
        
    @pytest.mark.asyncio
    async def test_evaluate_alert_condition(self):
        """Test alert condition evaluation."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        # Set up metric value
        collector.set_gauge("cpu_usage", 85.0)
        
        alert = Alert(
            id="cpu_alert",
            name="High CPU",
            description="CPU usage too high",
            level=AlertLevel.WARNING,
            condition="cpu_usage > 80",
            threshold=80.0,
            evaluation_window=300,
            cooldown_period=600
        )
        
        # Test condition evaluation
        triggered = await manager._evaluate_condition(alert)
        assert triggered is True
        
        # Test with lower value
        collector.set_gauge("cpu_usage", 75.0)
        triggered = await manager._evaluate_condition(alert)
        assert triggered is False
        
    @pytest.mark.asyncio
    async def test_alert_triggering_and_resolution(self):
        """Test alert triggering and resolution."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        # Mock alert handler
        handler_calls = []
        def mock_handler(alert, data):
            handler_calls.append((alert.id, data["status"]))
            
        manager.add_alert_handler(mock_handler)
        
        alert = Alert(
            id="test_alert",
            name="Test Alert",
            description="Test description",
            level=AlertLevel.WARNING,
            condition="test_metric > 50",
            threshold=50.0,
            evaluation_window=300,
            cooldown_period=60
        )
        
        manager.add_alert(alert)
        
        with patch('src.project_maestro.core.monitoring.publish_event') as mock_publish:
            # Trigger alert
            collector.set_gauge("test_metric", 75.0)
            await manager._evaluate_alert(alert)
            
            assert alert.is_active is True
            assert alert.trigger_count == 1
            assert len(handler_calls) == 1
            assert handler_calls[0] == ("test_alert", "triggered")
            
            # Resolve alert
            collector.set_gauge("test_metric", 25.0)
            await manager._evaluate_alert(alert)
            
            assert alert.is_active is False
            assert len(handler_calls) == 2
            assert handler_calls[1] == ("test_alert", "resolved")
            
            # Should have published events
            assert mock_publish.call_count == 2
            
    @pytest.mark.asyncio
    async def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        alert = Alert(
            id="cooldown_alert",
            name="Cooldown Test",
            description="Test cooldown",
            level=AlertLevel.WARNING,
            condition="test_metric > 50",
            threshold=50.0,
            evaluation_window=300,
            cooldown_period=1  # 1 second cooldown
        )
        
        manager.add_alert(alert)
        
        with patch('src.project_maestro.core.monitoring.publish_event'):
            # Trigger alert first time
            collector.set_gauge("test_metric", 75.0)
            await manager._evaluate_alert(alert)
            
            assert alert.is_active is True
            first_trigger_count = alert.trigger_count
            
            # Try to trigger again immediately (should be in cooldown)
            await manager._evaluate_alert(alert)
            
            # Trigger count should not increase due to cooldown
            assert alert.trigger_count == first_trigger_count
            
            # Wait for cooldown to expire
            await asyncio.sleep(1.1)
            
            # Reset alert state for re-triggering test
            alert.is_active = False
            
            # Trigger again after cooldown
            await manager._evaluate_alert(alert)
            
            assert alert.trigger_count == first_trigger_count + 1


class TestHealthChecker:
    """Test health checker functionality."""
    
    def test_checker_initialization(self):
        """Test checker initialization."""
        checker = HealthChecker()
        
        assert checker.checks == {}
        
    def test_add_health_check(self):
        """Test adding health checks."""
        checker = HealthChecker()
        
        def test_check():
            return True
            
        checker.add_health_check("test_component", test_check)
        
        assert "test_component" in checker.checks
        assert checker.checks["test_component"] == test_check
        
    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test running health checks."""
        checker = HealthChecker()
        
        # Add sync health check
        def healthy_check():
            return True
            
        def unhealthy_check():
            return False
            
        def error_check():
            raise Exception("Check failed")
            
        # Add async health check
        async def async_healthy_check():
            return True
            
        checker.add_health_check("healthy", healthy_check)
        checker.add_health_check("unhealthy", unhealthy_check)
        checker.add_health_check("error", error_check)
        checker.add_health_check("async_healthy", async_healthy_check)
        
        results = await checker.run_health_checks()
        
        assert len(results) == 4
        
        # Check healthy component
        assert results["healthy"]["status"] == "healthy"
        assert "response_time" in results["healthy"]
        
        # Check unhealthy component
        assert results["unhealthy"]["status"] == "unhealthy"
        
        # Check error component
        assert results["error"]["status"] == "unhealthy"
        assert "error" in results["error"]
        
        # Check async healthy component
        assert results["async_healthy"]["status"] == "healthy"


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        # Initialize components
        collector = MetricsCollector()
        system_monitor = SystemMonitor(collector)
        agent_monitor = AgentMonitor(collector)
        alert_manager = AlertManager(collector)
        health_checker = HealthChecker()
        
        # Set up alert
        cpu_alert = Alert(
            id="cpu_high",
            name="High CPU Usage",
            description="CPU usage above threshold",
            level=AlertLevel.WARNING,
            condition="system.cpu_percent > 80",
            threshold=80.0,
            evaluation_window=60,
            cooldown_period=300
        )
        alert_manager.add_alert(cpu_alert)
        
        # Set up health check
        def system_health():
            return collector.get_latest_value("system.cpu_percent") < 90
            
        health_checker.add_health_check("system", system_health)
        
        # Simulate system metrics
        with patch('src.project_maestro.core.monitoring.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 85.0  # High CPU
            mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0)
            mock_psutil.disk_usage.return_value = MagicMock(used=40, total=100)
            mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=1000, bytes_recv=1000)
            mock_psutil.pids.return_value = list(range(100))
            
            # Collect system metrics
            metrics = system_monitor.collect_system_metrics()
            collector.set_gauge("system.cpu_percent", metrics.cpu_percent)
            
            # Simulate agent activity
            agent_monitor.record_agent_task_start("test_agent", "task_001")
            agent_monitor.record_agent_task_completion("test_agent", "task_001", True)
            
            # Evaluate alert
            with patch('src.project_maestro.core.monitoring.publish_event') as mock_publish:
                await alert_manager._evaluate_alert(cpu_alert)
                
                # Alert should be triggered
                assert cpu_alert.is_active is True
                mock_publish.assert_called()
                
            # Run health checks
            health_results = await health_checker.run_health_checks()
            
            # System should be unhealthy due to high CPU
            assert health_results["system"]["status"] == "unhealthy"
            
        # Verify metrics were collected
        cpu_metrics = collector.get_metrics("system.cpu_percent")
        assert len(cpu_metrics) > 0
        assert cpu_metrics[0].value == 85.0
        
        completed_tasks = collector.get_metrics("agent.tasks.started")
        assert len(completed_tasks) > 0