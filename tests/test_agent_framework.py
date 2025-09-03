"""Unit tests for the agent framework."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.project_maestro.core.agent_framework import (
    AgentType, AgentStatus, AgentTask, AgentMetrics, 
    BaseAgent, AgentRegistry
)
from conftest import MockLLM, generate_agent_task


class TestAgentTask:
    """Test agent task functionality."""
    
    def test_agent_task_creation(self):
        """Test creating an agent task."""
        task = AgentTask(
            agent_type=AgentType.CODEX,
            action="generate_code",
            parameters={"language": "python"},
            priority=8,
            timeout=600
        )
        
        assert task.agent_type == AgentType.CODEX
        assert task.action == "generate_code"
        assert task.parameters["language"] == "python"
        assert task.priority == 8
        assert task.timeout == 600
        assert task.status == AgentStatus.IDLE
        assert task.id is not None
        assert task.created_at is not None
        
    def test_agent_task_defaults(self):
        """Test agent task default values."""
        task = AgentTask(
            agent_type=AgentType.CANVAS,
            action="generate_sprite"
        )
        
        assert task.priority == 5  # Default priority
        assert task.timeout is None  # Default timeout
        assert task.parameters == {}  # Default empty parameters


class TestAgentMetrics:
    """Test agent metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = AgentMetrics()
        
        assert metrics.task_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0
        assert metrics.success_rate == 1.0  # No tasks = 100% success rate
        
    def test_update_execution_success(self):
        """Test updating metrics with successful execution."""
        metrics = AgentMetrics()
        
        metrics.update_execution(1.5, True)
        
        assert metrics.task_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.total_execution_time == 1.5
        assert metrics.average_execution_time == 1.5
        assert metrics.success_rate == 1.0
        
    def test_update_execution_failure(self):
        """Test updating metrics with failed execution."""
        metrics = AgentMetrics()
        
        metrics.update_execution(2.0, False)
        
        assert metrics.task_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 2.0
        assert metrics.average_execution_time == 2.0
        assert metrics.success_rate == 0.0
        
    def test_multiple_executions(self):
        """Test metrics with multiple executions."""
        metrics = AgentMetrics()
        
        metrics.update_execution(1.0, True)
        metrics.update_execution(2.0, False)
        metrics.update_execution(1.5, True)
        
        assert metrics.task_count == 3
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert metrics.total_execution_time == 4.5
        assert metrics.average_execution_time == 1.5
        assert metrics.success_rate == 2/3


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def get_system_prompt(self) -> str:
        return "Test agent system prompt"
        
    async def execute_task(self, task: AgentTask) -> dict:
        """Mock task execution."""
        if task.action == "fail":
            raise ValueError("Test failure")
        return {"result": f"Completed {task.action}"}


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_initialization(self, mock_llm):
        """Test agent initialization."""
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        assert agent.name == "test_agent"
        assert agent.agent_type == AgentType.CODEX
        assert agent.llm == mock_llm
        assert agent.status == AgentStatus.IDLE
        assert agent.current_task is None
        assert isinstance(agent.metrics, AgentMetrics)
        
    def test_agent_default_llm_creation(self):
        """Test default LLM creation."""
        with patch('src.project_maestro.core.agent_framework.settings') as mock_settings:
            mock_settings.openai_api_key = "test_key"
            mock_settings.anthropic_api_key = None
            
            agent = ConcreteAgent(
                name="test_agent",
                agent_type=AgentType.CODEX
            )
            
            assert agent.llm is not None
            
    @pytest.mark.asyncio
    async def test_successful_task_processing(self, mock_llm):
        """Test successful task processing."""
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        task = generate_agent_task(
            agent_type=AgentType.CODEX,
            action="generate_code"
        )
        
        # Mock monitoring functions
        with patch('src.project_maestro.core.agent_framework.agent_monitor') as mock_monitor, \
             patch('src.project_maestro.core.agent_framework.metrics_collector') as mock_metrics, \
             patch('src.project_maestro.core.agent_framework.error_recovery_engine') as mock_recovery:
            
            result_task = await agent.process_task(task)
            
            assert result_task.status == AgentStatus.COMPLETED
            assert result_task.result["result"] == "Completed generate_code"
            assert result_task.completed_at is not None
            assert agent.status == AgentStatus.IDLE
            assert agent.current_task is None
            
            # Verify monitoring calls
            mock_monitor.record_agent_task_start.assert_called_once()
            mock_monitor.record_agent_task_completion.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_failed_task_processing(self, mock_llm):
        """Test failed task processing."""
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        task = generate_agent_task(
            agent_type=AgentType.CODEX,
            action="fail"  # This will trigger the failure case
        )
        
        # Mock monitoring functions
        with patch('src.project_maestro.core.agent_framework.agent_monitor') as mock_monitor, \
             patch('src.project_maestro.core.agent_framework.metrics_collector') as mock_metrics, \
             patch('src.project_maestro.core.agent_framework.error_recovery_engine') as mock_recovery:
            
            result_task = await agent.process_task(task)
            
            assert result_task.status == AgentStatus.FAILED
            assert result_task.error is not None
            assert "Test failure" in result_task.error
            assert result_task.completed_at is not None
            assert agent.status == AgentStatus.IDLE
            
            # Verify error handling calls
            mock_monitor.record_agent_task_completion.assert_called_with(
                agent.name, task.id, False
            )
            mock_recovery.handle_error.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_task_timeout(self, mock_llm):
        """Test task timeout handling."""
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        # Create a task that will timeout
        task = generate_agent_task(
            agent_type=AgentType.CODEX,
            action="slow_task",
            timeout=0.1  # Very short timeout
        )
        
        # Mock the execute_task to be slow
        original_execute = agent.execute_task
        async def slow_execute(task):
            await asyncio.sleep(0.2)  # Longer than timeout
            return await original_execute(task)
        agent.execute_task = slow_execute
        
        with patch('src.project_maestro.core.agent_framework.agent_monitor') as mock_monitor, \
             patch('src.project_maestro.core.agent_framework.metrics_collector') as mock_metrics, \
             patch('src.project_maestro.core.agent_framework.error_recovery_engine') as mock_recovery:
            
            result_task = await agent.process_task(task)
            
            assert result_task.status == AgentStatus.FAILED
            assert result_task.error is not None
            assert agent.status == AgentStatus.IDLE
            
    def test_can_handle_task(self, mock_llm):
        """Test task handling capability check."""
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        codex_task = generate_agent_task(agent_type=AgentType.CODEX)
        canvas_task = generate_agent_task(agent_type=AgentType.CANVAS)
        
        assert agent.can_handle_task(codex_task) is True
        assert agent.can_handle_task(canvas_task) is False


class TestAgentRegistry:
    """Test agent registry functionality."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()
        
        assert isinstance(registry.agents, dict)
        assert len(registry.agents) == 0
        
    def test_agent_registration(self, mock_llm):
        """Test agent registration."""
        registry = AgentRegistry()
        agent = ConcreteAgent(
            name="test_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        registry.register_agent(agent)
        
        assert "test_agent" in registry.agents
        assert registry.agents["test_agent"] == agent
        
    def test_agent_retrieval(self, mock_llm):
        """Test agent retrieval methods."""
        registry = AgentRegistry()
        
        codex_agent = ConcreteAgent("codex_agent", AgentType.CODEX, mock_llm)
        canvas_agent = ConcreteAgent("canvas_agent", AgentType.CANVAS, mock_llm)
        
        registry.register_agent(codex_agent)
        registry.register_agent(canvas_agent)
        
        # Test get_agent
        assert registry.get_agent("codex_agent") == codex_agent
        assert registry.get_agent("nonexistent") is None
        
        # Test get_agents_by_type
        codex_agents = registry.get_agents_by_type(AgentType.CODEX)
        assert len(codex_agents) == 1
        assert codex_agents[0] == codex_agent
        
        canvas_agents = registry.get_agents_by_type(AgentType.CANVAS)
        assert len(canvas_agents) == 1
        assert canvas_agents[0] == canvas_agent
        
        # Test non-existent type
        orchestrator_agents = registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        assert len(orchestrator_agents) == 0
        
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, mock_llm):
        """Test agent shutdown functionality."""
        registry = AgentRegistry()
        
        agent = ConcreteAgent("test_agent", AgentType.CODEX, mock_llm)
        agent.shutdown = AsyncMock()  # Mock the shutdown method
        
        registry.register_agent(agent)
        
        await registry.shutdown_all()
        
        agent.shutdown.assert_called_once()


@pytest.mark.performance
class TestAgentPerformance:
    """Test agent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_task_processing_performance(self, mock_llm, performance_tracker):
        """Test task processing performance."""
        agent = ConcreteAgent(
            name="perf_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        task = generate_agent_task(
            agent_type=AgentType.CODEX,
            action="performance_test"
        )
        
        with patch('src.project_maestro.core.agent_framework.agent_monitor'), \
             patch('src.project_maestro.core.agent_framework.metrics_collector'), \
             patch('src.project_maestro.core.agent_framework.error_recovery_engine'):
            
            performance_tracker.start()
            await agent.process_task(task)
            performance_tracker.stop()
            
            # Task processing should complete within reasonable time
            assert performance_tracker.duration < 1.0  # Less than 1 second
            
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, mock_llm):
        """Test concurrent task processing performance."""
        agent = ConcreteAgent(
            name="concurrent_agent",
            agent_type=AgentType.CODEX,
            llm=mock_llm
        )
        
        tasks = [
            generate_agent_task(
                agent_type=AgentType.CODEX,
                action=f"concurrent_test_{i}"
            ) for i in range(5)
        ]
        
        with patch('src.project_maestro.core.agent_framework.agent_monitor'), \
             patch('src.project_maestro.core.agent_framework.metrics_collector'), \
             patch('src.project_maestro.core.agent_framework.error_recovery_engine'):
            
            start_time = datetime.now()
            
            # Process tasks concurrently
            results = await asyncio.gather(*[
                agent.process_task(task) for task in tasks
            ])
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # All tasks should complete successfully
            assert len(results) == 5
            assert all(result.status == AgentStatus.COMPLETED for result in results)
            
            # Concurrent processing should be faster than sequential
            # (This is a simplified test - in reality, mock execution is very fast)
            assert total_time < 5.0  # Should complete within 5 seconds