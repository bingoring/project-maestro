"""Command Line Interface for Project Maestro."""

import asyncio
import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from .core.config import settings
from .core.logging import logger
from .api.main import run_server

app = typer.Typer(
    name="maestro",
    help="Project Maestro: AI Agent-based Game Prototyping Automation System",
    rich_markup_mode="markdown"
)

console = Console()

# Server commands
server_app = typer.Typer(name="server", help="Server management commands")
app.add_typer(server_app)

# Project commands  
project_app = typer.Typer(name="project", help="Project management commands")
app.add_typer(project_app)

# Agent commands
agent_app = typer.Typer(name="agent", help="Agent management commands")
app.add_typer(agent_app)


@app.command()
def version():
    """Show version information."""
    console.print(f"Project Maestro v0.1.0", style="bold green")
    console.print(f"Environment: {settings.environment}")
    console.print(f"Debug: {settings.debug}")


@server_app.command("start")
def start_server(
    host: str = typer.Option(settings.api_host, help="Host to bind to"),
    port: int = typer.Option(settings.api_port, help="Port to bind to"),
    workers: int = typer.Option(settings.api_workers, help="Number of workers"),
    reload: bool = typer.Option(settings.debug, help="Enable auto-reload")
):
    """Start the API server."""
    console.print("🚀 Starting Project Maestro API server...", style="bold blue")
    
    # Update settings
    settings.api_host = host
    settings.api_port = port
    settings.api_workers = workers
    
    console.print(f"Server will start at http://{host}:{port}")
    console.print(f"Workers: {workers}")
    console.print(f"Reload: {reload}")
    console.print(f"Environment: {settings.environment}")
    
    run_server()


@server_app.command("status")
def server_status():
    """Check server status."""
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.get(f"http://{settings.api_host}:{settings.api_port}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                console.print("✅ Server is running", style="bold green")
                
                # Display component status
                table = Table(title="Component Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="magenta")
                table.add_column("Details", style="green")
                
                for name, info in health_data.get("components", {}).items():
                    status = "✅ Healthy" if info.get("status") == "healthy" else "❌ Unhealthy"
                    details = info.get("error", f"Response time: {info.get('response_time', 'N/A')}ms")
                    table.add_row(name.title(), status, details)
                
                console.print(table)
            else:
                console.print("❌ Server is not responding correctly", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Server is not running: {str(e)}", style="bold red")


@project_app.command("create")
def create_project(
    title: str = typer.Argument(..., help="Project title"),
    description: str = typer.Option("", help="Project description"),
    document_file: Optional[Path] = typer.Option(None, help="Game design document file"),
    document_text: Optional[str] = typer.Option(None, help="Game design document text")
):
    """Create a new game project."""
    
    # Get game design document content
    if document_file:
        if not document_file.exists():
            console.print(f"❌ File not found: {document_file}", style="bold red")
            return
        game_design_document = document_file.read_text()
    elif document_text:
        game_design_document = document_text
    else:
        console.print("❌ Please provide either --document-file or --document-text", style="bold red")
        return
    
    console.print(f"🎮 Creating project: [bold]{title}[/bold]")
    
    # Make API request
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"http://{settings.api_host}:{settings.api_port}/api/v1/projects/",
                json={
                    "title": title,
                    "description": description,
                    "game_design_document": game_design_document
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                project_data = response.json()
                project_id = project_data["project_id"]
                
                console.print(f"✅ Project created successfully!", style="bold green")
                console.print(f"Project ID: [bold]{project_id}[/bold]")
                console.print(f"Status: {project_data['status']}")
                console.print(f"Progress: {project_data['progress']:.1%}")
                
                # Monitor progress
                if typer.confirm("Monitor progress?"):
                    monitor_project(project_id)
                    
            else:
                console.print(f"❌ Failed to create project: {response.text}", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Error creating project: {str(e)}", style="bold red")


@project_app.command("list")
def list_projects(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    page: int = typer.Option(1, help="Page number"),
    size: int = typer.Option(10, help="Page size")
):
    """List all projects."""
    
    import httpx
    
    try:
        params = {"page": page, "size": size}
        if status:
            params["status"] = status
            
        with httpx.Client() as client:
            response = client.get(
                f"http://{settings.api_host}:{settings.api_port}/api/v1/projects/",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                projects = data["items"]
                
                if not projects:
                    console.print("No projects found.", style="yellow")
                    return
                
                # Display projects table
                table = Table(title=f"Projects (Page {page}/{data['pages']})")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Title", style="magenta")
                table.add_column("Status", style="green")
                table.add_column("Progress", style="blue")
                table.add_column("Created", style="dim")
                
                for project in projects:
                    table.add_row(
                        project["project_id"][:8] + "...",
                        project["title"],
                        project["status"],
                        f"{project['progress']:.1%}",
                        project["created_at"][:16]
                    )
                
                console.print(table)
                console.print(f"Total: {data['total']} projects")
                
            else:
                console.print(f"❌ Failed to list projects: {response.text}", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Error listing projects: {str(e)}", style="bold red")


@project_app.command("status")
def project_status(project_id: str = typer.Argument(..., help="Project ID")):
    """Get detailed project status."""
    
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"http://{settings.api_host}:{settings.api_port}/api/v1/projects/{project_id}/status"
            )
            
            if response.status_code == 200:
                data = response.json()
                project = data["project"]
                workflow = data["workflow_status"]
                
                # Project info panel
                project_info = f"""
**Title:** {project['title']}
**Status:** {project['status']}
**Progress:** {project['progress']:.1%}
**Created:** {project['created_at']}
**Updated:** {project['updated_at']}
                """
                
                console.print(Panel(project_info.strip(), title="Project Information"))
                
                # Workflow status
                workflow_info = f"""
**Current Phase:** {workflow['current_phase']}
**Phases Completed:** {workflow['phases_completed']}/{workflow['total_phases']}
**Estimated Remaining:** {workflow['estimated_remaining_time']//60} minutes
                """
                
                console.print(Panel(workflow_info.strip(), title="Workflow Status"))
                
                # Current tasks
                if data["current_tasks"]:
                    table = Table(title="Current Tasks")
                    table.add_column("Agent", style="cyan")
                    table.add_column("Task ID", style="magenta")
                    table.add_column("Status", style="green")
                    
                    for task in data["current_tasks"]:
                        table.add_row(task["agent"], task["task_id"], task["status"])
                    
                    console.print(table)
                
            else:
                console.print(f"❌ Failed to get project status: {response.text}", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Error getting project status: {str(e)}", style="bold red")


def monitor_project(project_id: str):
    """Monitor project progress in real-time."""
    
    import httpx
    import time
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        task = progress.add_task("Monitoring project progress...", total=None)
        
        try:
            with httpx.Client() as client:
                while True:
                    response = client.get(
                        f"http://{settings.api_host}:{settings.api_port}/api/v1/projects/{project_id}"
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data["status"]
                        prog = data["progress"]
                        
                        progress.update(task, description=f"Status: {status} - Progress: {prog:.1%}")
                        
                        if status in ["completed", "failed"]:
                            break
                            
                    time.sleep(5)  # Check every 5 seconds
                    
        except KeyboardInterrupt:
            console.print("\n⏸️ Monitoring stopped by user")
        except Exception as e:
            console.print(f"\n❌ Error monitoring project: {str(e)}", style="bold red")


@agent_app.command("list")
def list_agents():
    """List all agents and their status."""
    
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"http://{settings.api_host}:{settings.api_port}/api/v1/agents/"
            )
            
            if response.status_code == 200:
                agents = response.json()
                
                if not agents:
                    console.print("No agents found.", style="yellow")
                    return
                
                # Display agents table
                table = Table(title="Agents Status")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="magenta")
                table.add_column("Status", style="green")
                table.add_column("Tasks", style="blue")
                table.add_column("Success Rate", style="yellow")
                
                for agent in agents:
                    metrics = agent["metrics"]
                    table.add_row(
                        agent["name"],
                        agent["type"],
                        agent["status"],
                        str(metrics.get("task_count", 0)),
                        f"{metrics.get('success_rate', 0):.1%}"
                    )
                
                console.print(table)
                
            else:
                console.print(f"❌ Failed to list agents: {response.text}", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Error listing agents: {str(e)}", style="bold red")


@agent_app.command("status")
def agent_status(agent_name: str = typer.Argument(..., help="Agent name")):
    """Get detailed agent status."""
    
    import httpx
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"http://{settings.api_host}:{settings.api_port}/api/v1/agents/{agent_name}"
            )
            
            if response.status_code == 200:
                agent = response.json()
                metrics = agent["metrics"]
                
                agent_info = f"""
**Name:** {agent['name']}
**Type:** {agent['type']}
**Status:** {agent['status']}
**Current Task:** {agent['current_task'] or 'None'}
**Total Tasks:** {metrics.get('task_count', 0)}
**Success Rate:** {metrics.get('success_rate', 0):.1%}
**Average Execution Time:** {metrics.get('average_execution_time', 0):.1f}s
                """
                
                console.print(Panel(agent_info.strip(), title=f"Agent: {agent_name}"))
                
            else:
                console.print(f"❌ Agent not found: {agent_name}", style="bold red")
                
    except Exception as e:
        console.print(f"❌ Error getting agent status: {str(e)}", style="bold red")


@app.command("config")
def show_config():
    """Show current configuration."""
    
    config_data = {
        "Environment": settings.environment,
        "Debug": settings.debug,
        "API Host": settings.api_host,
        "API Port": settings.api_port,
        "Database URL": settings.database_url,
        "Redis URL": settings.redis_url,
        "Storage Type": settings.storage_type,
        "Unity Path": settings.unity_path,
    }
    
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config_data.items():
        # Hide sensitive values
        if "key" in key.lower() or "password" in key.lower():
            value = "*" * 8 if value else "Not set"
        table.add_row(key, str(value))
    
    console.print(table)


@app.command("init")
def init_project(
    path: Path = typer.Option(Path.cwd(), help="Project directory"),
    template: str = typer.Option("basic", help="Project template")
):
    """Initialize a new Project Maestro project."""
    
    if not path.exists():
        path.mkdir(parents=True)
    
    # Create basic project structure
    (path / "game_design_documents").mkdir(exist_ok=True)
    (path / "generated_assets").mkdir(exist_ok=True)
    (path / "builds").mkdir(exist_ok=True)
    
    # Create sample game design document
    sample_gdd = """# Sample Game Design Document

## Game Overview
Title: Simple Platformer Game
Genre: 2D Platformer
Platform: Mobile (Android/iOS)

## Core Gameplay
- Player controls a character that can move left/right and jump
- Collect coins scattered throughout levels
- Avoid enemies and obstacles
- Reach the end of each level to progress

## Characters
- Player: A small, colorful character (sprite-based)
- Enemies: Simple moving obstacles

## Audio
- Background music: Upbeat, cheerful tune
- Sound effects: Jump sound, collect coin sound, damage sound

## Levels
- 3 levels with increasing difficulty
- Each level has platforms, enemies, and collectibles
"""
    
    (path / "game_design_documents" / "sample.md").write_text(sample_gdd)
    
    # Create configuration file
    config_content = f"""# Project Maestro Configuration
project_name = "My Game Project"
description = "A game created with Project Maestro"

[build]
target_platforms = ["Android", "iOS"]
unity_version = "2023.2.0f1"

[assets]
art_style = "pixel art"
audio_quality = "high"
"""
    
    (path / "maestro.toml").write_text(config_content)
    
    console.print(f"✅ Project initialized at: [bold]{path}[/bold]", style="green")
    console.print("📁 Created directories: game_design_documents, generated_assets, builds")
    console.print("📝 Created sample game design document: game_design_documents/sample.md")
    console.print("⚙️ Created configuration file: maestro.toml")


if __name__ == "__main__":
    app()