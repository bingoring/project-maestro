"""API models for Project Maestro."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Project Models
class ProjectCreateRequest(BaseModel):
    """Request to create a new project."""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    game_design_document: str = Field(..., description="Game design document content")


class ProjectResponse(BaseModel):
    """Project information response."""
    project_id: str
    title: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: float = Field(ge=0.0, le=1.0)
    estimated_completion: Optional[datetime] = None


class ProjectStatusResponse(BaseModel):
    """Detailed project status response."""
    project: ProjectResponse
    workflow_status: Dict[str, Any]
    current_tasks: List[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]
    agents_status: Dict[str, Any]


# Agent Models
class AgentStatus(BaseModel):
    """Agent status information."""
    name: str
    type: str
    status: str
    current_task: Optional[str] = None
    metrics: Dict[str, Any]


class AgentTaskRequest(BaseModel):
    """Request to create an agent task."""
    agent_type: str
    action: str
    parameters: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    timeout: Optional[int] = None


# Asset Models
class AssetUploadRequest(BaseModel):
    """Request to upload an asset."""
    project_id: str
    asset_type: str
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AssetResponse(BaseModel):
    """Asset information response."""
    id: str
    project_id: str
    filename: str
    asset_type: str
    mime_type: str
    file_size: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    download_url: Optional[str] = None


# Build Models
class BuildRequest(BaseModel):
    """Request to build a project."""
    project_id: str
    build_target: str = Field(default="Android", description="Target platform")
    build_options: Optional[Dict[str, Any]] = None


class BuildResponse(BaseModel):
    """Build result response."""
    build_id: str
    project_id: str
    build_target: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    build_path: Optional[str] = None
    download_url: Optional[str] = None
    error_message: Optional[str] = None


# Workflow Models
class WorkflowStepResponse(BaseModel):
    """Workflow step information."""
    step_id: str
    name: str
    agent_type: str
    status: str
    progress: float
    estimated_duration: int
    actual_duration: Optional[int] = None
    dependencies: List[str]
    error_message: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Workflow information response."""
    project_id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps: List[WorkflowStepResponse]
    estimated_total_duration: int
    actual_duration: Optional[int] = None


# Event Models
class EventResponse(BaseModel):
    """Event information response."""
    event_id: str
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None


# Analytics Models
class ProjectAnalytics(BaseModel):
    """Project analytics data."""
    project_id: str
    generation_time: float
    asset_counts: Dict[str, int]
    build_times: Dict[str, float]
    success_rate: float
    performance_metrics: Dict[str, Any]


class SystemAnalytics(BaseModel):
    """System-wide analytics."""
    total_projects: int
    total_assets: int
    total_builds: int
    average_generation_time: float
    success_rate: float
    agent_utilization: Dict[str, float]
    popular_genres: List[Dict[str, Any]]
    system_health: Dict[str, Any]


# Health Check Models
class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    dependencies: Dict[str, Dict[str, Any]]


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


# Configuration Models
class SystemConfigResponse(BaseModel):
    """System configuration response."""
    environment: str
    debug: bool
    api_version: str
    supported_platforms: List[str]
    max_file_size: int
    supported_formats: Dict[str, List[str]]
    rate_limits: Dict[str, int]