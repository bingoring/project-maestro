"""Main FastAPI application for Project Maestro."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.config import settings
from ..core.logging import logger
from ..core.message_queue import get_event_bus
from ..core.agent_framework import agent_registry
from ..core.storage import get_asset_manager
from ..core.monitoring import (
    start_monitoring, stop_monitoring, system_monitor, 
    agent_monitor, metrics_collector, health_checker
)
from ..core.error_handling import error_recovery_engine
from .models import HealthCheckResponse, ErrorResponse
from .endpoints import projects, agents, assets, builds, analytics, events, conversations, privacy, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Project Maestro API server")
    
    # Initialize event bus
    event_bus = await get_event_bus()
    await event_bus.start_listening()
    
    # Initialize agents (this would be done by a separate service in production)
    await _initialize_agents()
    
    # Initialize asset manager
    asset_manager = get_asset_manager()
    
    # Start monitoring services
    await start_monitoring()
    
    # Set up health checks
    health_checker.add_health_check("event_bus", lambda: event_bus.is_healthy())
    health_checker.add_health_check("asset_manager", lambda: asset_manager.is_healthy())
    health_checker.add_health_check("agent_registry", lambda: len(agent_registry.agents) > 0)
    
    logger.info(
        "API server startup complete",
        environment=settings.environment,
        debug=settings.debug
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Project Maestro API server")
    
    # Stop monitoring services
    await stop_monitoring()
    
    # Cleanup event bus
    await event_bus.shutdown()
    
    # Cleanup agents
    await agent_registry.shutdown_all()
    
    logger.info("API server shutdown complete")


async def _initialize_agents():
    """Initialize all agent instances."""
    from ..agents.orchestrator import OrchestratorAgent
    from ..agents.codex_agent import CodexAgent
    from ..agents.canvas_agent import CanvasAgent
    from ..agents.sonata_agent import SonataAgent
    from ..agents.labyrinth_agent import LabyrinthAgent
    from ..agents.builder_agent import BuilderAgent
    
    # Create agent instances
    orchestrator = OrchestratorAgent()
    codex = CodexAgent()
    canvas = CanvasAgent()
    sonata = SonataAgent()
    labyrinth = LabyrinthAgent()
    builder = BuilderAgent()
    
    # Register agents
    agent_registry.register_agent(orchestrator)
    agent_registry.register_agent(codex)
    agent_registry.register_agent(canvas)
    agent_registry.register_agent(sonata)
    agent_registry.register_agent(labyrinth)
    agent_registry.register_agent(builder)
    
    logger.info("All agents initialized and registered")


# Create FastAPI application
app = FastAPI(
    title="Project Maestro API",
    description="AI Agent-based Game Prototyping Automation System",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://maestro.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for web interface)
if (settings.project_root / "static").exists():
    app.mount("/static", StaticFiles(directory=settings.project_root / "static"), name="static")

# Include API routers
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(builds.router, prefix="/api/v1/builds", tags=["builds"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
app.include_router(conversations.router, tags=["conversations"])
app.include_router(privacy.router, tags=["privacy"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    if settings.debug:
        # Include full error details in debug mode
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="Internal server error",
                error_code="INTERNAL_ERROR",
                details={"error": str(exc), "type": type(exc).__name__}
            ).dict()
        )
    else:
        # Generic error message in production
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message="Internal server error",
                error_code="INTERNAL_ERROR"
            ).dict()
        )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Enhanced health check endpoint with comprehensive monitoring."""
    
    # Run all registered health checks
    components = await health_checker.run_health_checks()
    
    # Add system metrics
    system_metrics = system_monitor.collect_system_metrics()
    components["system"] = {
        "status": "healthy" if system_metrics.cpu_percent < 80 and system_metrics.memory_percent < 85 else "degraded",
        "cpu_percent": system_metrics.cpu_percent,
        "memory_percent": system_metrics.memory_percent,
        "disk_usage_percent": system_metrics.disk_usage_percent,
        "response_time": 0
    }
    
    # Add error statistics
    error_stats = error_recovery_engine.get_error_statistics()
    components["error_handling"] = {
        "status": "healthy" if error_stats["total_errors"] < 50 else "degraded",
        "total_errors": error_stats["total_errors"],
        "response_time": 0
    }
    
    # Legacy checks for backward compatibility
    # Check database connection
    try:
        # This would actually test the Redis connection
        components["redis"] = {"status": "healthy", "response_time": 5}
    except Exception as e:
        components["redis"] = {"status": "unhealthy", "error": str(e)}
        
    # Check agents status
    try:
        agents_status = agent_registry.get_agents_status()
        components["agents"] = {
            "status": "healthy",
            "agent_count": len(agents_status),
            "details": agents_status
        }
    except Exception as e:
        components["agents"] = {"status": "unhealthy", "error": str(e)}
    
    # Check storage backend
    try:
        asset_manager = get_asset_manager()
        # This would actually test the storage connection
        components["storage"] = {"status": "healthy"}
    except Exception as e:
        components["storage"] = {"status": "unhealthy", "error": str(e)}
    
    # Determine overall status
    overall_status = "healthy"
    if any(comp.get("status") == "unhealthy" for comp in components.values()):
        overall_status = "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="0.1.0",
        timestamp=datetime.now(),
        components=components,
        dependencies={
            "langchain": {"version": "0.1.0", "status": "healthy"},
            "openai": {"version": "1.6.0", "status": "healthy"},
            "anthropic": {"version": "0.8.0", "status": "healthy"}
        }
    )


# Monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    
    # Get latest system metrics
    system_metrics = system_monitor.collect_system_metrics()
    
    # Get agent metrics
    agent_metrics = {}
    for agent_name in agent_registry.agents.keys():
        agent_metrics[agent_name] = agent_monitor.get_agent_metrics(agent_name).dict() if hasattr(agent_monitor.get_agent_metrics(agent_name), 'dict') else agent_monitor.get_agent_metrics(agent_name).__dict__
    
    # Get error statistics
    error_stats = error_recovery_engine.get_error_statistics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": system_metrics.cpu_percent,
            "memory_percent": system_metrics.memory_percent,
            "disk_usage_percent": system_metrics.disk_usage_percent,
            "network_io_bytes": system_metrics.network_io_bytes,
            "process_count": system_metrics.process_count
        },
        "agents": agent_metrics,
        "errors": error_stats
    }


@app.get("/metrics/agents/{agent_name}")
async def get_agent_metrics(agent_name: str):
    """Get detailed metrics for a specific agent."""
    
    if agent_name not in agent_registry.agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    agent_metrics = agent_monitor.get_agent_metrics(agent_name)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "agent_name": agent_name,
        "metrics": agent_metrics.dict() if hasattr(agent_metrics, 'dict') else agent_metrics.__dict__
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Project Maestro API",
        "version": "0.1.0",
        "description": "AI Agent-based Game Prototyping Automation System",
        "docs": "/docs" if settings.debug else "Documentation available to authenticated users",
        "status": "operational"
    }


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # This would return Prometheus-formatted metrics
    # For now, return basic metrics
    return {
        "http_requests_total": 100,
        "projects_created_total": 10,
        "builds_completed_total": 8,
        "agent_tasks_processed_total": 45,
        "system_uptime_seconds": 3600
    }


# Development endpoints (only available in debug mode)
if settings.debug:
    
    @app.post("/dev/trigger-event")
    async def trigger_event(event_type: str, data: Dict[str, Any]):
        """Development endpoint to trigger events."""
        from ..core.message_queue import publish_event, EventType
        
        try:
            event_type_enum = EventType(event_type)
            event_id = await publish_event(
                event_type_enum,
                "dev_api",
                data
            )
            return {"event_id": event_id, "message": "Event triggered successfully"}
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type: {event_type}"
            )
    
    @app.post("/dev/reset-agents")
    async def reset_agents():
        """Development endpoint to reset all agents."""
        await agent_registry.shutdown_all()
        await _initialize_agents()
        return {"message": "All agents reset successfully"}


# Run server function
def run_server():
    """Run the API server."""
    uvicorn.run(
        "project_maestro.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=settings.debug
    )


if __name__ == "__main__":
    run_server()