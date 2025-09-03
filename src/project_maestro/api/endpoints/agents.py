"""Agent management API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from ...core.agent_framework import agent_registry, AgentType, AgentTask
from ...core.logging import get_logger
from ..models import BaseResponse, AgentStatus, AgentTaskRequest

router = APIRouter()
logger = get_logger("api.agents")


@router.get("/", response_model=List[AgentStatus])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by agent status")
):
    """List all agents and their current status."""
    
    try:
        agents_status = agent_registry.get_agents_status()
        
        # Convert to AgentStatus models
        agent_list = []
        for name, status_data in agents_status.items():
            agent_list.append(AgentStatus(
                name=name,
                type=status_data["type"],
                status=status_data["status"],
                current_task=status_data["current_task"],
                metrics=status_data["metrics"]
            ))
        
        # Apply filters
        if agent_type:
            agent_list = [a for a in agent_list if a.type == agent_type]
            
        if status:
            agent_list = [a for a in agent_list if a.status == status]
        
        return agent_list
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_name}", response_model=AgentStatus)
async def get_agent_status(agent_name: str):
    """Get detailed status of a specific agent."""
    
    try:
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        status_data = agent.get_status()
        
        return AgentStatus(
            name=status_data["name"],
            type=status_data["type"],
            status=status_data["status"],
            current_task=status_data["current_task"],
            metrics=status_data["metrics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.post("/{agent_name}/tasks", response_model=BaseResponse)
async def create_agent_task(agent_name: str, task_request: AgentTaskRequest):
    """Create a new task for a specific agent."""
    
    try:
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # Verify agent can handle this task type
        try:
            agent_type = AgentType(task_request.agent_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type: {task_request.agent_type}"
            )
        
        if not agent.can_handle_task(AgentTask(agent_type=agent_type, action="", parameters={})):
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{agent_name}' cannot handle tasks of type '{task_request.agent_type}'"
            )
        
        # Create and submit task
        task = AgentTask(
            agent_type=agent_type,
            action=task_request.action,
            parameters=task_request.parameters,
            priority=task_request.priority,
            timeout=task_request.timeout
        )
        
        # In a real implementation, this would be queued for processing
        logger.info(
            "Task created for agent",
            agent_name=agent_name,
            task_id=task.id,
            action=task.action
        )
        
        return BaseResponse(
            success=True,
            message=f"Task '{task.id}' created for agent '{agent_name}'"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create agent task", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent task: {str(e)}"
        )


@router.get("/types/", response_model=List[str])
async def list_agent_types():
    """List all available agent types."""
    return [agent_type.value for agent_type in AgentType]


@router.post("/{agent_name}/pause", response_model=BaseResponse) 
async def pause_agent(agent_name: str):
    """Pause an agent (stop accepting new tasks)."""
    
    try:
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # In real implementation, would pause the agent
        logger.info("Agent paused", agent_name=agent_name)
        
        return BaseResponse(
            success=True,
            message=f"Agent '{agent_name}' paused successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause agent", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause agent: {str(e)}"
        )


@router.post("/{agent_name}/resume", response_model=BaseResponse)
async def resume_agent(agent_name: str):
    """Resume a paused agent."""
    
    try:
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # In real implementation, would resume the agent
        logger.info("Agent resumed", agent_name=agent_name)
        
        return BaseResponse(
            success=True,
            message=f"Agent '{agent_name}' resumed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume agent", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume agent: {str(e)}"
        )


@router.get("/{agent_name}/metrics")
async def get_agent_metrics(agent_name: str):
    """Get detailed metrics for a specific agent."""
    
    try:
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        status = agent.get_status()
        metrics = status["metrics"]
        
        # Add additional computed metrics
        detailed_metrics = {
            **metrics,
            "uptime_seconds": 3600,  # Mock uptime
            "memory_usage_mb": 256,   # Mock memory usage
            "cpu_usage_percent": 15.5, # Mock CPU usage
            "tasks_per_minute": metrics.get("task_count", 0) / 60,
            "error_rate": 1 - metrics.get("success_rate", 1.0)
        }
        
        return detailed_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent metrics: {str(e)}"
        )