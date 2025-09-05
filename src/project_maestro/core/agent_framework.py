"""Base agent framework for Project Maestro using LangChain."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# Modern LangChain imports
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .config import settings
from .logging import AgentLoggerMixin, get_logger
from .monitoring import agent_monitor, metrics_collector
from .error_handling import error_recovery_engine, with_error_handling, RecoveryStrategy


class AgentType(str, Enum):
    """Types of agents in the Project Maestro system."""
    ORCHESTRATOR = "orchestrator"
    CODEX = "codex"
    CANVAS = "canvas"
    SONATA = "sonata"
    LABYRINTH = "labyrinth"
    BUILDER = "builder"
    QUERY = "query"  # Enterprise knowledge query agent


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking."""
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    
    def update_execution(self, execution_time: float, success: bool):
        """Update metrics after task execution."""
        self.task_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.task_count
        self.last_execution_time = datetime.now()


class AgentTask(BaseModel):
    """Task model for agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5)  # 1-10, higher is more priority
    timeout: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    parent_task_id: Optional[str] = None
    child_tasks: List[str] = Field(default_factory=list)


class MaestroCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Project Maestro agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(f"agent.{agent_name}")
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str],
        **kwargs: Any
    ) -> Any:
        """Called when LLM starts generating."""
        self.logger.info(
            "LLM started",
            agent=self.agent_name,
            prompts_count=len(prompts)
        )
        
    def on_llm_end(
        self, 
        response,
        **kwargs: Any
    ) -> Any:
        """Called when LLM finishes generating."""
        self.logger.info(
            "LLM finished",
            agent=self.agent_name,
            response_length=len(str(response))
        )
        
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts execution."""
        self.logger.info(
            "Tool started",
            agent=self.agent_name,
            tool=serialized.get("name", "unknown"),
            input=input_str
        )
        
    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends execution."""
        self.logger.info(
            "Tool completed",
            agent=self.agent_name,
            output=output
        )
        
    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> Any:
        """Called when tool encounters an error."""
        self.logger.error(
            "Tool error",
            agent=self.agent_name,
            error=str(error),
            error_type=type(error).__name__
        )


class BaseAgent(ABC, AgentLoggerMixin):
    """
    Modern base class for all Project Maestro agents using LCEL and Runnable patterns.
    
    This class provides a foundation for building agents that leverage LangChain's
    Expression Language (LCEL) for composable, streaming-capable AI workflows.
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.agent_type = agent_type
        self.tools = tools or []
        self.metrics = AgentMetrics()
        self.current_task: Optional[AgentTask] = None
        self.status = AgentStatus.IDLE
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = self._create_default_llm()
            
        # Create callback handler
        self.callback_handler = MaestroCallbackHandler(self.name)
        
        # Initialize modern agent graph (replaces AgentExecutor)
        self._agent_graph = None
        self._runnable_chain = None
        
    def _create_default_llm(self) -> BaseLanguageModel:
        """Create default LLM with modern configuration."""
        if settings.openai_api_key:
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o",  # Updated to latest model
                temperature=0.1,
                callbacks=[self.callback_handler],
                streaming=True  # Enable streaming by default
            )
        elif settings.anthropic_api_key:
            return ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model="claude-3-5-sonnet-20241022",  # Updated to latest model
                temperature=0.1,
                callbacks=[self.callback_handler],
                streaming=True  # Enable streaming by default
            )
        else:
            raise ValueError("No LLM API key configured")
            
    @property
    def agent_graph(self):
        """Get or create modern agent graph using LangGraph."""
        if not self._agent_graph:
            self._agent_graph = self._create_agent_graph()
        return self._agent_graph
        
    @property  
    def runnable_chain(self):
        """Get or create LCEL runnable chain."""
        if not self._runnable_chain:
            self._runnable_chain = self._create_runnable_chain()
        return self._runnable_chain
        
    @abstractmethod
    def _create_agent_graph(self):
        """Create modern LangGraph agent (replaces AgentExecutor)."""
        pass
        
    @abstractmethod
    def _create_runnable_chain(self):
        """Create LCEL runnable chain for this agent."""
        pass
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
        
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task using modern LCEL patterns."""
        pass
        
    async def stream_task(self, task: AgentTask):
        """Stream task execution for real-time feedback."""
        async for chunk in self.agent_graph.astream(
            {"messages": [("human", f"Task: {task.action} with params: {task.parameters}")]},
            config={"configurable": {"thread_id": task.id}}
        ):
            yield chunk
            
    @with_error_handling("base_agent", "process_task", max_attempts=2, recovery_strategy=RecoveryStrategy.RETRY)
    async def process_task(self, task: AgentTask) -> AgentTask:
        """Process a task with modern error handling and metrics."""
        self.current_task = task
        task.started_at = datetime.now()
        task.status = AgentStatus.RUNNING
        self.status = AgentStatus.RUNNING
        
        # Record task start in monitoring
        agent_monitor.record_agent_task_start(self.name, task.id)
        metrics_collector.increment_counter(
            "agent.tasks.started",
            tags={"agent_name": self.name, "agent_type": self.agent_type.value, "action": task.action}
        )
        
        self.log_agent_action(
            "process_task", 
            "started",
            task_id=task.id,
            action=task.action
        )
        
        start_time = datetime.now()
        
        try:
            # Use modern async execution with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self.execute_task(task),
                    timeout=task.timeout
                )
            else:
                result = await self.execute_task(task)
                
            # Update task with result
            task.result = result
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_execution(execution_time, True)
            
            # Record successful completion in monitoring
            agent_monitor.record_agent_task_completion(self.name, task.id, True)
            metrics_collector.record_timing(
                "agent.task.duration",
                execution_time * 1000,  # Convert to milliseconds
                tags={"agent_name": self.name, "agent_type": self.agent_type.value, "success": "true"}
            )
            
            self.log_agent_success(
                "process_task",
                result=result,
                task_id=task.id,
                execution_time=execution_time
            )
            
        except Exception as e:
            # Handle task failure with enhanced error context
            task.error = str(e)
            task.status = AgentStatus.FAILED
            task.completed_at = datetime.now()
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_execution(execution_time, False)
            
            # Record failure in monitoring
            agent_monitor.record_agent_task_completion(self.name, task.id, False)
            agent_monitor.record_agent_error(self.name, type(e).__name__, str(e))
            metrics_collector.record_timing(
                "agent.task.duration",
                execution_time * 1000,
                tags={"agent_name": self.name, "agent_type": self.agent_type.value, "success": "false"}
            )
            metrics_collector.increment_counter(
                "agent.errors",
                tags={"agent_name": self.name, "agent_type": self.agent_type.value, "error_type": type(e).__name__}
            )
            
            self.log_agent_error(
                "process_task",
                e,
                task_id=task.id,
                execution_time=execution_time
            )
            
            # Handle error through error recovery engine
            await error_recovery_engine.handle_error(
                e, self.name, "process_task", 
                {"task_id": task.id, "action": task.action}
            )
            
        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None
            
        return task
        
    def can_handle_task(self, task: AgentTask) -> bool:
        """Check if this agent can handle the given task."""
        return task.agent_type == self.agent_type
        
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status with enhanced metrics."""
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "current_task": self.current_task.id if self.current_task else None,
            "capabilities": {
                "streaming": True,
                "lcel_enabled": True,
                "runnable_interface": True,
                "langgraph_powered": True
            },
            "metrics": {
                "task_count": self.metrics.task_count,
                "success_rate": (
                    self.metrics.success_count / max(self.metrics.task_count, 1)
                ),
                "average_execution_time": self.metrics.average_execution_time,
                "last_execution": self.metrics.last_execution_time.isoformat() 
                    if self.metrics.last_execution_time else None
            }
        }


class AgentRegistry:
    """Registry for managing agent instances."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[AgentType, List[BaseAgent]] = {}
        self.logger = get_logger("agent_registry")
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance."""
        self._agents[agent.name] = agent
        
        if agent.agent_type not in self._agents_by_type:
            self._agents_by_type[agent.agent_type] = []
        self._agents_by_type[agent.agent_type].append(agent)
        
        self.logger.info(
            "Agent registered",
            name=agent.name,
            type=agent.agent_type.value
        )
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(name)
        
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        return self._agents_by_type.get(agent_type, [])
        
    def get_available_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get an available agent of the specified type."""
        agents = self.get_agents_by_type(agent_type)
        for agent in agents:
            if agent.status == AgentStatus.IDLE:
                return agent
        return None
        
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
        
    def get_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            name: agent.get_status() 
            for name, agent in self._agents.items()
        }
        
    async def shutdown_all(self):
        """Shutdown all agents gracefully."""
        self.logger.info("Shutting down all agents")
        # Add cleanup logic if needed
        for agent in self._agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()


# Global agent registry
agent_registry = AgentRegistry()