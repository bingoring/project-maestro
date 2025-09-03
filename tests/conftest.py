"""Pytest configuration and fixtures for Project Maestro tests."""

import asyncio
import os
import tempfile
import uuid
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseMessage, AIMessage

# Test configuration
os.environ["MAESTRO_ENVIRONMENT"] = "test"
os.environ["MAESTRO_DEBUG"] = "true"
os.environ["MAESTRO_STORAGE_TYPE"] = "local"


class MockLLM(BaseLanguageModel):
    """Mock language model for testing."""
    
    def __init__(self, responses: list = None):
        self.responses = responses or ["Test response"]
        self.call_count = 0
        
    def _generate(self, messages, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)
    
    async def _agenerate(self, messages, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)
    
    @property
    def _llm_type(self) -> str:
        return "mock"


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    return MockLLM([
        "Mock response 1",
        "Mock response 2", 
        "Mock response 3"
    ])


@pytest.fixture
def temp_storage_path():
    """Provide a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_game_design_document():
    """Provide a sample game design document for testing."""
    return """
# Simple Platformer Game

## Game Overview
- **Title**: Super Jump Adventure
- **Genre**: Platformer
- **Platform**: Mobile (Android/iOS)
- **Art Style**: Pixel art

## Core Gameplay
- Player controls a character that can move left/right and jump
- Collect coins scattered throughout levels
- Avoid enemies and obstacles
- Reach the end of each level to progress

## Characters
### Player Character
- **Name**: Hero
- **Role**: Player
- **Description**: A small, colorful character
- **Abilities**: Move, Jump, Collect items
- **Animations**: Idle, Walk, Jump, Fall, Collect

### Enemies
- **Name**: Goomba
- **Role**: Enemy
- **Description**: Small walking enemy
- **Abilities**: Move left/right, Damage player on contact

## Levels
### Level 1: Green Hills
- **Theme**: Grassland
- **Objectives**: Collect 10 coins, Reach the flag
- **Enemies**: 3 Goombas
- **Difficulty**: Easy

### Level 2: Desert Canyon
- **Theme**: Desert
- **Objectives**: Collect 15 coins, Avoid cacti, Reach the flag
- **Enemies**: 5 Goombas, 2 Cacti
- **Difficulty**: Medium

## Audio Requirements
- **Background Music**: Upbeat, cheerful tune that loops
- **Sound Effects**: Jump sound, coin collect sound, damage sound

## Technical Requirements
- Unity 2023.2.0f1
- Target 60 FPS on mobile devices
- Touch controls for mobile
"""


@pytest.fixture
def sample_project_spec():
    """Provide a sample project specification for testing."""
    return {
        "id": str(uuid.uuid4()),
        "title": "Super Jump Adventure",
        "description": "A simple platformer game",
        "genre": "platformer",
        "platform": "mobile",
        "art_style": "pixel art",
        "gameplay_mechanics": ["movement", "jumping", "collecting"],
        "characters": [
            {
                "name": "Hero",
                "role": "player",
                "description": "A small, colorful character",
                "abilities": ["move", "jump", "collect"]
            }
        ],
        "environments": [
            {
                "name": "Green Hills",
                "theme": "grassland",
                "description": "Rolling green hills with platforms"
            }
        ],
        "sounds": [
            {
                "name": "jump_sound",
                "type": "sfx",
                "description": "Sound when player jumps"
            },
            {
                "name": "background_music",
                "type": "bgm",
                "description": "Upbeat background music"
            }
        ],
        "levels": [
            {
                "name": "Level 1",
                "theme": "Green Hills",
                "objectives": ["Collect 10 coins", "Reach the flag"],
                "difficulty": "easy"
            }
        ],
        "technical_requirements": {
            "unity_version": "2023.2.0f1",
            "target_fps": 60,
            "platforms": ["Android", "iOS"]
        },
        "estimated_complexity": 5
    }


@pytest.fixture
async def mock_event_bus():
    """Provide a mock event bus for testing."""
    mock_bus = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus.subscribe = AsyncMock()
    mock_bus.start_listening = AsyncMock()
    mock_bus.shutdown = AsyncMock()
    mock_bus.is_healthy = MagicMock(return_value=True)
    return mock_bus


@pytest.fixture
async def mock_asset_manager():
    """Provide a mock asset manager for testing."""
    mock_manager = AsyncMock()
    mock_manager.upload_asset = AsyncMock(return_value=MagicMock(
        id="test_asset_id",
        filename="test.png",
        asset_type="sprite"
    ))
    mock_manager.get_asset_info = MagicMock(return_value=MagicMock(
        id="test_asset_id",
        filename="test.png"
    ))
    mock_manager.is_healthy = MagicMock(return_value=True)
    return mock_manager


@pytest.fixture
def mock_agent_registry():
    """Provide a mock agent registry for testing."""
    from src.project_maestro.core.agent_framework import AgentType
    
    mock_registry = MagicMock()
    mock_registry.agents = {
        "orchestrator": MagicMock(agent_type=AgentType.ORCHESTRATOR),
        "codex": MagicMock(agent_type=AgentType.CODEX),
        "canvas": MagicMock(agent_type=AgentType.CANVAS)
    }
    mock_registry.get_agents_by_type = MagicMock(
        return_value=[mock_registry.agents["orchestrator"]]
    )
    mock_registry.shutdown_all = AsyncMock()
    return mock_registry


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_client():
    """Provide a test client for API testing."""
    from httpx import AsyncClient
    from src.project_maestro.api.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Test data generators
def generate_agent_task(agent_type=None, action="test_action", **kwargs):
    """Generate a test agent task."""
    from src.project_maestro.core.agent_framework import AgentTask, AgentType
    
    return AgentTask(
        agent_type=agent_type or AgentType.CODEX,
        action=action,
        parameters=kwargs.get("parameters", {}),
        priority=kwargs.get("priority", 5),
        timeout=kwargs.get("timeout", 300)
    )


def generate_test_project_id():
    """Generate a unique test project ID."""
    return f"test_project_{uuid.uuid4().hex[:8]}"


# Mock external services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Mock OpenAI response")
                )
            ]
        )
    )
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text="Mock Anthropic response")]
        )
    )
    return mock_client


@pytest.fixture
def mock_stable_diffusion_client():
    """Mock Stable Diffusion client."""
    mock_client = AsyncMock()
    mock_client.generate_image = AsyncMock(
        return_value=b"mock_image_data"
    )
    return mock_client


# Environment setup and teardown
@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Set up test environment before each test."""
    # Clear any cached instances
    # Reset monitoring state
    from src.project_maestro.core.monitoring import metrics_collector
    metrics_collector.metrics_store.clear()
    
    yield
    
    # Cleanup after test
    pass


# Performance testing utilities
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
            
        def record_metric(self, name, value):
            self.metrics[name] = value
            
    return PerformanceTracker()


# Assertion helpers
def assert_agent_task_valid(task):
    """Assert that an agent task is valid."""
    from src.project_maestro.core.agent_framework import AgentTask, AgentType
    
    assert isinstance(task, AgentTask)
    assert isinstance(task.agent_type, AgentType)
    assert task.action is not None
    assert task.id is not None
    assert task.created_at is not None


def assert_api_response_valid(response, expected_status=200):
    """Assert that an API response is valid."""
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    
    if expected_status == 200:
        data = response.json()
        assert isinstance(data, dict)
        

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow