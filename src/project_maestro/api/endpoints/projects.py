"""Project management API endpoints."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from sqlalchemy.orm import Session

from ...core.agent_framework import agent_registry, AgentType, AgentTask
from ...core.message_queue import publish_event, EventType
from ...core.logging import get_logger
from ..models import (
    BaseResponse, ErrorResponse, ProjectCreateRequest, ProjectResponse,
    ProjectStatusResponse, PaginationParams, PaginatedResponse
)

router = APIRouter()
logger = get_logger("api.projects")

# Mock database session dependency
def get_db():
    """Get database session (mock implementation)."""
    # In a real implementation, this would return actual DB session
    return None


@router.post("/", response_model=ProjectResponse)
async def create_project(
    request: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new game project from design document."""
    
    try:
        logger.info(
            "Creating new project",
            title=request.title,
            doc_length=len(request.game_design_document)
        )
        
        # Generate project ID
        project_id = str(uuid.uuid4())
        
        # Get orchestrator agent
        orchestrator = agent_registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        if not orchestrator:
            raise HTTPException(
                status_code=503,
                detail="Orchestrator agent not available"
            )
        
        orchestrator_agent = orchestrator[0]
        
        # Create orchestrator task
        task = AgentTask(
            agent_type=AgentType.ORCHESTRATOR,
            action="process_game_document",
            parameters={
                "document": request.game_design_document,
                "project_id": project_id,
                "title": request.title,
                "description": request.description
            },
            timeout=1800  # 30 minutes
        )
        
        # Process task in background
        background_tasks.add_task(
            _process_project_creation_task,
            orchestrator_agent,
            task,
            project_id
        )
        
        # Publish project creation event
        await publish_event(
            EventType.PROJECT_CREATED,
            "api",
            {
                "project_id": project_id,
                "title": request.title,
                "description": request.description
            }
        )
        
        return ProjectResponse(
            project_id=project_id,
            title=request.title,
            description=request.description,
            status="created",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            progress=0.0
        )
        
    except Exception as e:
        logger.error("Failed to create project", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create project: {str(e)}"
        )


async def _process_project_creation_task(
    orchestrator_agent,
    task: AgentTask,
    project_id: str
):
    """Process project creation task in background."""
    try:
        logger.info("Processing project creation task", project_id=project_id)
        
        # Execute the orchestrator task
        result = await orchestrator_agent.process_task(task)
        
        if result.status == "completed":
            logger.info(
                "Project creation task completed successfully",
                project_id=project_id,
                result=result.result
            )
        else:
            logger.error(
                "Project creation task failed",
                project_id=project_id,
                error=result.error
            )
            
    except Exception as e:
        logger.error(
            "Error processing project creation task",
            project_id=project_id,
            error=str(e)
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get project information."""
    
    # Mock implementation - in real app would query database
    return ProjectResponse(
        project_id=project_id,
        title="Sample Project",
        description="Sample game project",
        status="in_progress",
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now() - timedelta(minutes=30),
        progress=0.65,
        estimated_completion=datetime.now() + timedelta(hours=1)
    )


@router.get("/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str, db: Session = Depends(get_db)):
    """Get detailed project status including workflow and agents."""
    
    try:
        # Get orchestrator status for this project
        orchestrator_agents = agent_registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        
        project_info = ProjectResponse(
            project_id=project_id,
            title="Sample Project",
            description="Sample game project",
            status="in_progress",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now(),
            progress=0.45
        )
        
        # Mock workflow status
        workflow_status = {
            "current_phase": "asset_generation",
            "phases_completed": 2,
            "total_phases": 5,
            "estimated_remaining_time": 3600
        }
        
        # Get current agent tasks
        current_tasks = []
        completed_tasks = []
        
        for agent in agent_registry.get_all_agents():
            agent_status = agent.get_status()
            if agent_status["current_task"]:
                current_tasks.append({
                    "agent": agent_status["name"],
                    "task_id": agent_status["current_task"],
                    "status": agent_status["status"]
                })
        
        # Mock completed tasks
        completed_tasks = [
            {
                "agent": "codex_agent",
                "task": "generate_player_controller",
                "completed_at": datetime.now() - timedelta(minutes=45),
                "duration": 180
            },
            {
                "agent": "canvas_agent", 
                "task": "generate_character_sprites",
                "completed_at": datetime.now() - timedelta(minutes=30),
                "duration": 420
            }
        ]
        
        agents_status = agent_registry.get_agents_status()
        
        return ProjectStatusResponse(
            project=project_info,
            workflow_status=workflow_status,
            current_tasks=current_tasks,
            completed_tasks=completed_tasks,
            agents_status=agents_status
        )
        
    except Exception as e:
        logger.error("Failed to get project status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project status: {str(e)}"
        )


@router.get("/", response_model=PaginatedResponse)
async def list_projects(
    pagination: PaginationParams = Depends(),
    status: Optional[str] = Query(None, description="Filter by project status"),
    db: Session = Depends(get_db)
):
    """List all projects with pagination and filtering."""
    
    # Mock implementation
    mock_projects = [
        ProjectResponse(
            project_id=str(uuid.uuid4()),
            title=f"Game Project {i}",
            description=f"Description for game project {i}",
            status="completed" if i % 3 == 0 else "in_progress",
            created_at=datetime.now() - timedelta(days=i),
            updated_at=datetime.now() - timedelta(hours=i),
            progress=min(1.0, 0.1 * i)
        )
        for i in range(1, 26)  # 25 mock projects
    ]
    
    # Apply status filter
    if status:
        mock_projects = [p for p in mock_projects if p.status == status]
    
    # Apply pagination
    start_idx = (pagination.page - 1) * pagination.size
    end_idx = start_idx + pagination.size
    page_items = mock_projects[start_idx:end_idx]
    
    total = len(mock_projects)
    pages = (total + pagination.size - 1) // pagination.size
    
    return PaginatedResponse(
        items=page_items,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages,
        has_next=pagination.page < pages,
        has_prev=pagination.page > 1
    )


@router.delete("/{project_id}", response_model=BaseResponse)
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project and all its assets."""
    
    try:
        logger.info("Deleting project", project_id=project_id)
        
        # In real implementation:
        # 1. Stop any running tasks for this project
        # 2. Delete all assets from storage
        # 3. Delete project from database
        # 4. Publish deletion event
        
        await publish_event(
            EventType.PROJECT_CREATED,  # Would be PROJECT_DELETED
            "api",
            {"project_id": project_id, "action": "deleted"}
        )
        
        return BaseResponse(
            success=True,
            message=f"Project {project_id} deleted successfully"
        )
        
    except Exception as e:
        logger.error("Failed to delete project", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete project: {str(e)}"
        )


@router.post("/{project_id}/pause", response_model=BaseResponse)
async def pause_project(project_id: str, db: Session = Depends(get_db)):
    """Pause project processing."""
    
    try:
        logger.info("Pausing project", project_id=project_id)
        
        # In real implementation, would pause all running tasks
        
        return BaseResponse(
            success=True,
            message=f"Project {project_id} paused successfully"
        )
        
    except Exception as e:
        logger.error("Failed to pause project", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause project: {str(e)}"
        )


@router.post("/{project_id}/resume", response_model=BaseResponse)
async def resume_project(project_id: str, db: Session = Depends(get_db)):
    """Resume paused project processing."""
    
    try:
        logger.info("Resuming project", project_id=project_id)
        
        # In real implementation, would resume paused tasks
        
        return BaseResponse(
            success=True,
            message=f"Project {project_id} resumed successfully"
        )
        
    except Exception as e:
        logger.error("Failed to resume project", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume project: {str(e)}"
        )


@router.get("/{project_id}/logs")
async def get_project_logs(
    project_id: str,
    level: str = Query("INFO", description="Log level filter"),
    limit: int = Query(100, description="Maximum number of logs to return")
):
    """Get project processing logs."""
    
    # Mock logs implementation
    mock_logs = [
        {
            "timestamp": datetime.now() - timedelta(minutes=i),
            "level": "INFO" if i % 3 != 0 else "WARNING",
            "agent": ["orchestrator", "codex", "canvas", "sonata"][i % 4],
            "message": f"Processing step {i} for project {project_id}",
            "details": {"step": i, "progress": i * 0.1}
        }
        for i in range(min(limit, 50))
    ]
    
    return {"logs": mock_logs, "total": len(mock_logs)}


@router.post("/{project_id}/regenerate", response_model=BaseResponse)
async def regenerate_project_assets(
    project_id: str,
    asset_types: List[str] = Query([], description="Asset types to regenerate"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Regenerate specific project assets."""
    
    try:
        logger.info(
            "Regenerating project assets",
            project_id=project_id,
            asset_types=asset_types
        )
        
        if not asset_types:
            asset_types = ["code", "sprites", "audio", "levels"]
            
        # In real implementation, would create tasks for each asset type
        for asset_type in asset_types:
            await publish_event(
                EventType.TASK_CREATED,
                "api",
                {
                    "project_id": project_id,
                    "action": f"regenerate_{asset_type}",
                    "asset_type": asset_type
                }
            )
        
        return BaseResponse(
            success=True,
            message=f"Asset regeneration started for {len(asset_types)} asset types"
        )
        
    except Exception as e:
        logger.error("Failed to regenerate assets", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate assets: {str(e)}"
        )