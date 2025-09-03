"""Build management API endpoints."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends

from ...core.agent_framework import agent_registry, AgentType, AgentTask
from ...core.message_queue import publish_event, EventType
from ...core.logging import get_logger
from ..models import BaseResponse, BuildRequest, BuildResponse, PaginationParams, PaginatedResponse

router = APIRouter()
logger = get_logger("api.builds")


@router.post("/", response_model=BuildResponse)
async def create_build(
    request: BuildRequest,
    background_tasks: BackgroundTasks
):
    """Create a new build for a project."""
    
    try:
        logger.info(
            "Creating new build",
            project_id=request.project_id,
            target=request.build_target
        )
        
        # Generate build ID
        build_id = str(uuid.uuid4())
        
        # Get builder agent
        builder_agents = agent_registry.get_agents_by_type(AgentType.BUILDER)
        if not builder_agents:
            raise HTTPException(
                status_code=503,
                detail="Builder agent not available"
            )
        
        builder_agent = builder_agents[0]
        
        # Create build task
        task = AgentTask(
            agent_type=AgentType.BUILDER,
            action="build_game",
            parameters={
                "project_id": request.project_id,
                "build_target": request.build_target,
                "build_options": request.build_options or {},
                "build_id": build_id
            },
            timeout=3600  # 1 hour
        )
        
        # Process build in background
        background_tasks.add_task(
            _process_build_task,
            builder_agent,
            task,
            build_id
        )
        
        # Publish build started event
        await publish_event(
            EventType.BUILD_STARTED,
            "api",
            {
                "build_id": build_id,
                "project_id": request.project_id,
                "build_target": request.build_target
            }
        )
        
        return BuildResponse(
            build_id=build_id,
            project_id=request.project_id,
            build_target=request.build_target,
            status="started",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error("Failed to create build", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create build: {str(e)}"
        )


async def _process_build_task(
    builder_agent,
    task: AgentTask,
    build_id: str
):
    """Process build task in background."""
    try:
        logger.info("Processing build task", build_id=build_id)
        
        # Execute the builder task
        result = await builder_agent.process_task(task)
        
        if result.status == "completed":
            logger.info(
                "Build task completed successfully",
                build_id=build_id,
                result=result.result
            )
            
            # Publish build completed event
            await publish_event(
                EventType.BUILD_COMPLETED,
                "builder_agent",
                {
                    "build_id": build_id,
                    "success": True,
                    "build_path": result.result.get("build_path") if result.result else None
                }
            )
        else:
            logger.error(
                "Build task failed",
                build_id=build_id,
                error=result.error
            )
            
            await publish_event(
                EventType.BUILD_COMPLETED,
                "builder_agent",
                {
                    "build_id": build_id,
                    "success": False,
                    "error": result.error
                }
            )
            
    except Exception as e:
        logger.error(
            "Error processing build task",
            build_id=build_id,
            error=str(e)
        )


@router.get("/{build_id}", response_model=BuildResponse)
async def get_build(build_id: str):
    """Get build information."""
    
    # Mock implementation - in real app would query database
    return BuildResponse(
        build_id=build_id,
        project_id="sample-project-id",
        build_target="Android",
        status="completed",
        created_at=datetime.now() - timedelta(minutes=30),
        completed_at=datetime.now() - timedelta(minutes=5),
        build_path="/builds/sample-game.apk",
        download_url="https://storage.example.com/builds/sample-game.apk"
    )


@router.get("/", response_model=PaginatedResponse)
async def list_builds(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    build_target: Optional[str] = Query(None, description="Filter by build target"),
    status: Optional[str] = Query(None, description="Filter by build status"),
    pagination: PaginationParams = Depends()
):
    """List builds with pagination and filtering."""
    
    # Mock implementation
    mock_builds = [
        BuildResponse(
            build_id=str(uuid.uuid4()),
            project_id=project_id or f"project-{i}",
            build_target="Android" if i % 2 == 0 else "iOS",
            status="completed" if i % 3 == 0 else "in_progress",
            created_at=datetime.now() - timedelta(hours=i),
            completed_at=datetime.now() - timedelta(hours=i-1) if i % 3 == 0 else None,
            build_path=f"/builds/game-{i}.apk" if i % 3 == 0 else None,
            download_url=f"https://storage.example.com/builds/game-{i}.apk" if i % 3 == 0 else None
        )
        for i in range(1, 16)  # 15 mock builds
    ]
    
    # Apply filters
    if build_target:
        mock_builds = [b for b in mock_builds if b.build_target == build_target]
    
    if status:
        mock_builds = [b for b in mock_builds if b.status == status]
    
    # Apply pagination
    start_idx = (pagination.page - 1) * pagination.size
    end_idx = start_idx + pagination.size
    page_items = mock_builds[start_idx:end_idx]
    
    total = len(mock_builds)
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


@router.delete("/{build_id}", response_model=BaseResponse)
async def delete_build(build_id: str):
    """Delete a build and its artifacts."""
    
    try:
        logger.info("Deleting build", build_id=build_id)
        
        # In real implementation:
        # 1. Delete build artifacts from storage
        # 2. Delete build record from database
        # 3. Publish deletion event
        
        return BaseResponse(
            success=True,
            message=f"Build {build_id} deleted successfully"
        )
        
    except Exception as e:
        logger.error("Failed to delete build", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete build: {str(e)}"
        )


@router.get("/{build_id}/download")
async def get_build_download_url(build_id: str, expiry_seconds: int = Query(3600)):
    """Get download URL for build artifacts."""
    
    try:
        # In real implementation, would generate signed URL for build artifacts
        download_url = f"https://storage.example.com/builds/{build_id}.apk"
        
        return {
            "download_url": download_url,
            "expires_in": expiry_seconds,
            "file_size": 25600000,  # 25.6 MB
            "build_target": "Android"
        }
        
    except Exception as e:
        logger.error("Failed to get build download URL", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get build download URL: {str(e)}"
        )


@router.get("/{build_id}/logs")
async def get_build_logs(
    build_id: str,
    level: str = Query("INFO", description="Log level filter")
):
    """Get build process logs."""
    
    # Mock build logs
    mock_logs = [
        {
            "timestamp": datetime.now() - timedelta(minutes=i),
            "level": "INFO" if i % 4 != 0 else "WARNING",
            "message": f"Build step {10-i}: {['Unity project setup', 'Asset import', 'Script compilation', 'Scene building', 'Platform build'][i % 5]}",
            "details": {"step": 10-i, "progress": (10-i) * 0.1}
        }
        for i in range(10)
    ]
    
    return {"logs": mock_logs, "build_id": build_id}


@router.post("/{build_id}/cancel", response_model=BaseResponse)
async def cancel_build(build_id: str):
    """Cancel a running build."""
    
    try:
        logger.info("Cancelling build", build_id=build_id)
        
        # In real implementation, would stop the build process
        
        return BaseResponse(
            success=True,
            message=f"Build {build_id} cancelled successfully"
        )
        
    except Exception as e:
        logger.error("Failed to cancel build", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel build: {str(e)}"
        )


@router.get("/targets/", response_model=List[str])
async def list_build_targets():
    """List all supported build targets."""
    return ["Android", "iOS", "WebGL", "Windows", "macOS", "Linux"]