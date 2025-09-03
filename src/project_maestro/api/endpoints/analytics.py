"""Analytics API endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Query

from ...core.logging import get_logger
from ..models import ProjectAnalytics, SystemAnalytics

router = APIRouter()
logger = get_logger("api.analytics")


@router.get("/projects/{project_id}", response_model=ProjectAnalytics)
async def get_project_analytics(project_id: str):
    """Get analytics data for a specific project."""
    
    # Mock analytics data
    return ProjectAnalytics(
        project_id=project_id,
        generation_time=450.5,  # seconds
        asset_counts={
            "code": 15,
            "sprites": 25,
            "audio": 12,
            "levels": 5
        },
        build_times={
            "Android": 180.2,
            "iOS": 220.8,
            "WebGL": 95.3
        },
        success_rate=0.92,
        performance_metrics={
            "avg_compile_time": 45.2,
            "avg_asset_generation_time": 120.8,
            "memory_usage_peak_mb": 1024,
            "cpu_usage_avg_percent": 35.5
        }
    )


@router.get("/system", response_model=SystemAnalytics)
async def get_system_analytics(
    days: int = Query(30, description="Number of days to include in analytics")
):
    """Get system-wide analytics data."""
    
    # Mock system analytics
    return SystemAnalytics(
        total_projects=152,
        total_assets=3840,
        total_builds=428,
        average_generation_time=520.3,
        success_rate=0.89,
        agent_utilization={
            "orchestrator": 0.75,
            "codex": 0.68,
            "canvas": 0.82,
            "sonata": 0.45,
            "labyrinth": 0.53,
            "builder": 0.71
        },
        popular_genres=[
            {"genre": "Platformer", "count": 45, "percentage": 29.6},
            {"genre": "Puzzle", "count": 38, "percentage": 25.0},
            {"genre": "Action", "count": 32, "percentage": 21.1},
            {"genre": "RPG", "count": 22, "percentage": 14.5},
            {"genre": "Strategy", "count": 15, "percentage": 9.9}
        ],
        system_health={
            "api_response_time_ms": 125,
            "database_connections": 8,
            "redis_memory_usage_mb": 64,
            "storage_usage_gb": 128.5,
            "error_rate_24h": 0.02
        }
    )


@router.get("/projects/trends")
async def get_project_trends(
    days: int = Query(30, description="Number of days to analyze"),
    metric: str = Query("count", description="Metric to analyze (count, success_rate, duration)")
):
    """Get project creation and completion trends."""
    
    # Generate mock trend data
    trends = []
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        
        if metric == "count":
            value = 3 + (i % 7) + (1 if i % 3 == 0 else 0)
        elif metric == "success_rate":
            value = 0.85 + (0.1 * (i % 5) / 5) + (0.05 if i % 7 == 0 else 0)
        else:  # duration
            value = 450 + (50 * (i % 3)) + (100 if i % 5 == 0 else 0)
            
        trends.append({
            "date": date.isoformat(),
            "value": round(value, 2)
        })
    
    return {
        "metric": metric,
        "period_days": days,
        "trends": trends
    }


@router.get("/agents/performance")
async def get_agent_performance_metrics(
    days: int = Query(7, description="Number of days to analyze")
):
    """Get performance metrics for all agents."""
    
    # Mock agent performance data
    agent_metrics = {
        "orchestrator": {
            "tasks_completed": 45,
            "average_execution_time": 25.5,
            "success_rate": 0.96,
            "utilization": 0.75,
            "peak_memory_mb": 512,
            "errors_24h": 2
        },
        "codex": {
            "tasks_completed": 128,
            "average_execution_time": 180.2,
            "success_rate": 0.92,
            "utilization": 0.68,
            "peak_memory_mb": 256,
            "errors_24h": 5
        },
        "canvas": {
            "tasks_completed": 95,
            "average_execution_time": 420.8,
            "success_rate": 0.88,
            "utilization": 0.82,
            "peak_memory_mb": 1024,
            "errors_24h": 8
        },
        "sonata": {
            "tasks_completed": 67,
            "average_execution_time": 310.5,
            "success_rate": 0.85,
            "utilization": 0.45,
            "peak_memory_mb": 512,
            "errors_24h": 3
        },
        "labyrinth": {
            "tasks_completed": 78,
            "average_execution_time": 95.2,
            "success_rate": 0.94,
            "utilization": 0.53,
            "peak_memory_mb": 128,
            "errors_24h": 1
        },
        "builder": {
            "tasks_completed": 52,
            "average_execution_time": 650.3,
            "success_rate": 0.90,
            "utilization": 0.71,
            "peak_memory_mb": 2048,
            "errors_24h": 4
        }
    }
    
    return {
        "period_days": days,
        "agents": agent_metrics
    }


@router.get("/assets/distribution")
async def get_asset_distribution():
    """Get distribution of asset types across all projects."""
    
    return {
        "total_assets": 3840,
        "distribution": [
            {"type": "sprites", "count": 1250, "percentage": 32.6},
            {"type": "code", "count": 980, "percentage": 25.5},
            {"type": "audio", "count": 756, "percentage": 19.7},
            {"type": "levels", "count": 432, "percentage": 11.3},
            {"type": "ui_elements", "count": 278, "percentage": 7.2},
            {"type": "materials", "count": 144, "percentage": 3.8}
        ]
    }


@router.get("/builds/statistics")
async def get_build_statistics(
    days: int = Query(30, description="Number of days to analyze")
):
    """Get build success rates and performance statistics."""
    
    return {
        "period_days": days,
        "total_builds": 428,
        "successful_builds": 381,
        "failed_builds": 47,
        "success_rate": 0.89,
        "average_build_time_seconds": 245.8,
        "platform_statistics": {
            "Android": {
                "builds": 198,
                "success_rate": 0.91,
                "avg_time_seconds": 220.5,
                "avg_size_mb": 28.6
            },
            "iOS": {
                "builds": 145,
                "success_rate": 0.87,
                "avg_time_seconds": 285.2,
                "avg_size_mb": 32.4
            },
            "WebGL": {
                "builds": 85,
                "success_rate": 0.89,
                "avg_time_seconds": 195.8,
                "avg_size_mb": 15.2
            }
        }
    }


@router.get("/errors/summary")
async def get_error_summary(
    hours: int = Query(24, description="Number of hours to analyze")
):
    """Get error summary and patterns."""
    
    return {
        "period_hours": hours,
        "total_errors": 23,
        "error_rate": 0.03,
        "error_categories": [
            {"category": "Generation Timeout", "count": 8, "percentage": 34.8},
            {"category": "Build Failure", "count": 6, "percentage": 26.1},
            {"category": "Asset Processing", "count": 4, "percentage": 17.4},
            {"category": "API Rate Limit", "count": 3, "percentage": 13.0},
            {"category": "Storage Error", "count": 2, "percentage": 8.7}
        ],
        "error_agents": {
            "canvas": 8,
            "builder": 6,
            "codex": 4,
            "sonata": 3,
            "orchestrator": 2
        },
        "recent_errors": [
            {
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "agent": "canvas",
                "category": "Generation Timeout",
                "message": "Stable Diffusion API timeout after 600 seconds"
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "agent": "builder",
                "category": "Build Failure",
                "message": "Unity compilation failed: missing script reference"
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
                "agent": "codex",
                "category": "API Rate Limit",
                "message": "OpenAI API rate limit exceeded"
            }
        ]
    }


@router.get("/performance/real-time")
async def get_real_time_metrics():
    """Get real-time system performance metrics."""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_load": {
            "cpu_usage_percent": 42.5,
            "memory_usage_percent": 68.2,
            "disk_usage_percent": 35.8,
            "network_io_mbps": 15.3
        },
        "queue_depths": {
            "project_creation": 3,
            "asset_generation": 12,
            "builds": 2,
            "events": 0
        },
        "active_agents": {
            "orchestrator": 1,
            "codex": 2,
            "canvas": 3,
            "sonata": 1,
            "labyrinth": 2,
            "builder": 1
        },
        "response_times_ms": {
            "api_avg": 125,
            "database_avg": 45,
            "storage_avg": 85,
            "ai_services_avg": 2500
        }
    }