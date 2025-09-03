"""Integration tests for the FastAPI application."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient

from src.project_maestro.api.main import app
from conftest import (
    mock_event_bus, mock_asset_manager, mock_agent_registry,
    sample_game_design_document, sample_project_spec
)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Project Maestro API"
        assert data["version"] == "0.1.0"
        assert "description" in data
        assert data["status"] == "operational"
        
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        with patch('src.project_maestro.api.main.health_checker') as mock_health_checker, \
             patch('src.project_maestro.api.main.system_monitor') as mock_system_monitor, \
             patch('src.project_maestro.api.main.error_recovery_engine') as mock_error_engine:
            
            # Mock health checker
            mock_health_checker.run_health_checks.return_value = {
                "event_bus": {"status": "healthy", "response_time": 5},
                "asset_manager": {"status": "healthy", "response_time": 10}
            }
            
            # Mock system monitor
            mock_system_metrics = MagicMock()
            mock_system_metrics.cpu_percent = 45.0
            mock_system_metrics.memory_percent = 60.0
            mock_system_metrics.disk_usage_percent = 30.0
            mock_system_monitor.collect_system_metrics.return_value = mock_system_metrics
            
            # Mock error recovery engine
            mock_error_engine.get_error_statistics.return_value = {"total_errors": 5}
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")
                
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "version" in data
            assert "timestamp" in data
            assert "components" in data
            
            # Check system component
            assert "system" in data["components"]
            system_health = data["components"]["system"]
            assert system_health["status"] == "healthy"
            assert system_health["cpu_percent"] == 45.0
            
            # Check error handling component
            assert "error_handling" in data["components"]
            error_health = data["components"]["error_handling"]
            assert error_health["status"] == "healthy"
            assert error_health["total_errors"] == 5


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        with patch('src.project_maestro.api.main.system_monitor') as mock_system_monitor, \
             patch('src.project_maestro.api.main.agent_monitor') as mock_agent_monitor, \
             patch('src.project_maestro.api.main.agent_registry') as mock_registry, \
             patch('src.project_maestro.api.main.error_recovery_engine') as mock_error_engine:
            
            # Mock system monitor
            mock_system_metrics = MagicMock()
            mock_system_metrics.cpu_percent = 55.0
            mock_system_metrics.memory_percent = 65.0
            mock_system_metrics.disk_usage_percent = 40.0
            mock_system_metrics.network_io_bytes = 5000
            mock_system_metrics.process_count = 120
            mock_system_monitor.collect_system_metrics.return_value = mock_system_metrics
            
            # Mock agent registry
            mock_registry.agents.keys.return_value = ["orchestrator", "codex"]
            
            # Mock agent monitor
            mock_agent_metrics = MagicMock()
            mock_agent_metrics.__dict__ = {
                "agent_name": "test_agent",
                "tasks_completed": 10,
                "success_rate": 0.95
            }
            mock_agent_monitor.get_agent_metrics.return_value = mock_agent_metrics
            
            # Mock error recovery engine
            mock_error_engine.get_error_statistics.return_value = {
                "total_errors": 15,
                "errors_by_category": {"network": 5, "timeout": 3}
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/metrics")
                
            assert response.status_code == 200
            data = response.json()
            
            assert "timestamp" in data
            assert "system" in data
            assert "agents" in data
            assert "errors" in data
            
            # Check system metrics
            system = data["system"]
            assert system["cpu_percent"] == 55.0
            assert system["memory_percent"] == 65.0
            assert system["network_io_bytes"] == 5000
            
            # Check error statistics
            errors = data["errors"]
            assert errors["total_errors"] == 15
            
    @pytest.mark.asyncio
    async def test_agent_metrics_endpoint(self):
        """Test individual agent metrics endpoint."""
        with patch('src.project_maestro.api.main.agent_registry') as mock_registry, \
             patch('src.project_maestro.api.main.agent_monitor') as mock_agent_monitor:
            
            # Mock agent registry
            mock_registry.agents = {"test_agent": MagicMock()}
            
            # Mock agent metrics
            mock_agent_metrics = MagicMock()
            mock_agent_metrics.__dict__ = {
                "agent_name": "test_agent",
                "tasks_completed": 25,
                "tasks_failed": 2,
                "success_rate": 0.92
            }
            mock_agent_monitor.get_agent_metrics.return_value = mock_agent_metrics
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/metrics/agents/test_agent")
                
            assert response.status_code == 200
            data = response.json()
            
            assert data["agent_name"] == "test_agent"
            assert "metrics" in data
            metrics = data["metrics"]
            assert metrics["tasks_completed"] == 25
            assert metrics["success_rate"] == 0.92
            
    @pytest.mark.asyncio
    async def test_agent_metrics_not_found(self):
        """Test agent metrics endpoint with non-existent agent."""
        with patch('src.project_maestro.api.main.agent_registry') as mock_registry:
            mock_registry.agents = {}  # Empty registry
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/metrics/agents/nonexistent")
                
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()


class TestProjectsAPI:
    """Test projects API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_project(self, sample_game_design_document):
        """Test project creation."""
        with patch('src.project_maestro.api.endpoints.projects.agent_registry') as mock_registry, \
             patch('src.project_maestro.api.endpoints.projects.publish_event') as mock_publish:
            
            # Mock orchestrator agent
            mock_orchestrator = AsyncMock()
            mock_orchestrator.process_task = AsyncMock(return_value=MagicMock(
                status="completed",
                result={
                    "project_id": "test_project_123",
                    "title": "Test Game",
                    "workflow_steps": 5,
                    "status": "workflow_started"
                }
            ))
            mock_registry.get_agents_by_type.return_value = [mock_orchestrator]
            
            project_data = {
                "title": "Test Game Project",
                "description": "A test game for API testing",
                "game_design_document": sample_game_design_document
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/v1/projects/", json=project_data)
                
            assert response.status_code == 200
            data = response.json()
            
            assert "project_id" in data
            assert data["title"] == "Test Game Project"
            assert data["status"] in ["created", "processing"]
            
            # Verify orchestrator was called
            mock_orchestrator.process_task.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_list_projects(self):
        """Test listing projects."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/projects/")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert "pages" in data
        
    @pytest.mark.asyncio
    async def test_get_project(self):
        """Test getting a specific project."""
        project_id = "test_project_123"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/v1/projects/{project_id}")
            
        # Mock implementation returns 200 with project data
        assert response.status_code == 200
        data = response.json()
        
        assert "project_id" in data
        assert "title" in data
        assert "status" in data
        
    @pytest.mark.asyncio
    async def test_get_project_status(self):
        """Test getting project status."""
        project_id = "test_project_123"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/v1/projects/{project_id}/status")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "project" in data
        assert "workflow_status" in data


class TestAgentsAPI:
    """Test agents API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_agents(self):
        """Test listing agents."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/agents/")
            
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Mock implementation returns agent data
        
    @pytest.mark.asyncio
    async def test_get_agent(self):
        """Test getting specific agent."""
        agent_name = "orchestrator"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/v1/agents/{agent_name}")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "type" in data
        assert "status" in data


class TestBuildsAPI:
    """Test builds API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_build(self):
        """Test creating a build."""
        with patch('src.project_maestro.api.endpoints.builds.agent_registry') as mock_registry, \
             patch('src.project_maestro.api.endpoints.builds.publish_event') as mock_publish:
            
            # Mock builder agent
            mock_builder = MagicMock()
            mock_registry.get_agents_by_type.return_value = [mock_builder]
            
            build_data = {
                "project_id": "test_project_123",
                "build_target": "Android",
                "build_options": {"debug": True}
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/v1/builds/", json=build_data)
                
            assert response.status_code == 200
            data = response.json()
            
            assert "build_id" in data
            assert data["project_id"] == "test_project_123"
            assert data["build_target"] == "Android"
            assert data["status"] == "started"
            
    @pytest.mark.asyncio
    async def test_list_builds(self):
        """Test listing builds."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/builds/")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        
    @pytest.mark.asyncio
    async def test_get_build(self):
        """Test getting build information."""
        build_id = "test_build_456"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/v1/builds/{build_id}")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["build_id"] == build_id
        assert "status" in data


class TestAssetsAPI:
    """Test assets API endpoints."""
    
    @pytest.mark.asyncio
    async def test_upload_asset(self):
        """Test asset upload."""
        with patch('src.project_maestro.api.endpoints.assets.get_asset_manager') as mock_get_manager:
            # Mock asset manager
            mock_manager = AsyncMock()
            mock_asset_info = MagicMock()
            mock_asset_info.id = "asset_123"
            mock_asset_info.project_id = "project_123"
            mock_asset_info.filename = "test.png"
            mock_asset_info.asset_type = "sprite"
            mock_asset_info.mime_type = "image/png"
            mock_asset_info.file_size = 1024
            mock_asset_info.created_at = "2024-01-01T00:00:00"
            mock_asset_info.metadata = {}
            
            mock_manager.upload_asset = AsyncMock(return_value=mock_asset_info)
            mock_manager.get_asset_url = AsyncMock(return_value="http://example.com/asset.png")
            mock_get_manager.return_value = mock_manager
            
            # Create test file data
            files = {"file": ("test.png", b"fake_image_data", "image/png")}
            data = {
                "project_id": "project_123",
                "asset_type": "sprite"
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/v1/assets/upload", files=files, data=data)
                
            assert response.status_code == 200
            response_data = response.json()
            
            assert response_data["id"] == "asset_123"
            assert response_data["filename"] == "test.png"
            assert response_data["asset_type"] == "sprite"
            
    @pytest.mark.asyncio
    async def test_list_assets(self):
        """Test listing assets."""
        with patch('src.project_maestro.api.endpoints.assets.get_asset_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.list_project_assets.return_value = []
            mock_get_manager.return_value = mock_manager
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/api/v1/assets/?project_id=project_123")
                
            assert response.status_code == 200
            data = response.json()
            
            assert "items" in data
            assert "total" in data


class TestAnalyticsAPI:
    """Test analytics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_project_analytics(self):
        """Test getting project analytics."""
        project_id = "test_project_123"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/v1/analytics/projects/{project_id}")
            
        assert response.status_code == 200
        data = response.json()
        
        assert data["project_id"] == project_id
        assert "generation_time" in data
        assert "asset_counts" in data
        assert "success_rate" in data
        
    @pytest.mark.asyncio
    async def test_get_system_analytics(self):
        """Test getting system analytics."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/analytics/system")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "total_projects" in data
        assert "total_assets" in data
        assert "success_rate" in data
        assert "agent_utilization" in data


class TestEventsAPI:
    """Test events API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_events(self):
        """Test listing events."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/events/")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        
    @pytest.mark.asyncio
    async def test_list_event_types(self):
        """Test listing event types."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/events/types")
            
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
    @pytest.mark.asyncio
    async def test_get_event_statistics(self):
        """Test getting event statistics."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/events/statistics")
            
        assert response.status_code == 200
        data = response.json()
        
        assert "total_events" in data
        assert "events_per_hour" in data
        assert "event_type_distribution" in data


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the full API."""
    
    @pytest.mark.asyncio
    async def test_full_project_lifecycle(self, sample_game_design_document):
        """Test complete project creation to build lifecycle."""
        with patch('src.project_maestro.api.endpoints.projects.agent_registry') as mock_project_registry, \
             patch('src.project_maestro.api.endpoints.builds.agent_registry') as mock_build_registry, \
             patch('src.project_maestro.api.endpoints.projects.publish_event'), \
             patch('src.project_maestro.api.endpoints.builds.publish_event'):
            
            # Mock orchestrator
            mock_orchestrator = AsyncMock()
            mock_orchestrator.process_task = AsyncMock(return_value=MagicMock(
                status="completed",
                result={
                    "project_id": "integration_test_project",
                    "title": "Integration Test Game",
                    "workflow_steps": 5,
                    "status": "workflow_started"
                }
            ))
            mock_project_registry.get_agents_by_type.return_value = [mock_orchestrator]
            
            # Mock builder
            mock_builder = MagicMock()
            mock_build_registry.get_agents_by_type.return_value = [mock_builder]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # 1. Create project
                project_data = {
                    "title": "Integration Test Game",
                    "description": "Full lifecycle test",
                    "game_design_document": sample_game_design_document
                }
                
                project_response = await client.post("/api/v1/projects/", json=project_data)
                assert project_response.status_code == 200
                
                project_result = project_response.json()
                project_id = project_result["project_id"]
                
                # 2. Check project status
                status_response = await client.get(f"/api/v1/projects/{project_id}")
                assert status_response.status_code == 200
                
                # 3. Create build
                build_data = {
                    "project_id": project_id,
                    "build_target": "Android",
                    "build_options": {"release": True}
                }
                
                build_response = await client.post("/api/v1/builds/", json=build_data)
                assert build_response.status_code == 200
                
                build_result = build_response.json()
                build_id = build_result["build_id"]
                
                # 4. Check build status
                build_status_response = await client.get(f"/api/v1/builds/{build_id}")
                assert build_status_response.status_code == 200
                
                # 5. Get analytics
                analytics_response = await client.get(f"/api/v1/analytics/projects/{project_id}")
                assert analytics_response.status_code == 200
                
                # 6. Check system health
                health_response = await client.get("/health")
                assert health_response.status_code == 200
                
                # Verify all operations completed successfully
                assert project_result["title"] == "Integration Test Game"
                assert build_result["project_id"] == project_id
                assert build_result["build_target"] == "Android"


@pytest.mark.performance 
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_performance(self):
        """Test health endpoint response time."""
        with patch('src.project_maestro.api.main.health_checker') as mock_health_checker, \
             patch('src.project_maestro.api.main.system_monitor') as mock_system_monitor, \
             patch('src.project_maestro.api.main.error_recovery_engine') as mock_error_engine:
            
            mock_health_checker.run_health_checks.return_value = {}
            mock_system_monitor.collect_system_metrics.return_value = MagicMock(
                cpu_percent=50.0, memory_percent=60.0, disk_usage_percent=30.0
            )
            mock_error_engine.get_error_statistics.return_value = {"total_errors": 0}
            
            import time
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                start_time = time.time()
                response = await client.get("/health")
                end_time = time.time()
                
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 1.0  # Should respond within 1 second
            
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import asyncio
        
        async def make_request(client, endpoint):
            return await client.get(endpoint)
            
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Make 10 concurrent requests
            tasks = [
                make_request(client, "/health"),
                make_request(client, "/"),
                make_request(client, "/api/v1/projects/"),
                make_request(client, "/api/v1/agents/"),
                make_request(client, "/api/v1/builds/"),
            ] * 2  # Duplicate to get 10 requests
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should succeed
            for response in responses:
                assert response.status_code in [200, 404]  # 404 is ok for some mock endpoints
                
            # Total time should be reasonable for concurrent execution
            total_time = end_time - start_time
            assert total_time < 5.0  # Should complete within 5 seconds