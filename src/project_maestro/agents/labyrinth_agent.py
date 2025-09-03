"""Labyrinth Agent - specialized in procedural level design and gameplay progression."""

import asyncio
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.message_queue import EventType, publish_event
from ..core.config import settings


@dataclass
class LevelElement:
    """Represents an element in a level."""
    element_type: str  # platform, enemy, collectible, hazard, checkpoint
    position: Tuple[float, float]
    size: Tuple[float, float]
    properties: Dict[str, Any]


@dataclass
class LevelSection:
    """Represents a section of a level."""
    section_id: str
    difficulty: float  # 0.0 to 1.0
    elements: List[LevelElement]
    connections: List[str]  # Connected section IDs
    theme: str
    estimated_time: float  # seconds to complete


class LevelData(BaseModel):
    """Complete level data structure."""
    level_id: str
    name: str
    difficulty: float
    estimated_duration: float
    sections: List[Dict[str, Any]]  # LevelSection as dicts
    metadata: Dict[str, Any]
    generation_params: Dict[str, Any]


class DifficultyProgression:
    """Handles difficulty curve and progression."""
    
    def __init__(self):
        self.difficulty_factors = {
            "enemy_count": 0.3,
            "platform_complexity": 0.2,
            "timing_precision": 0.2,
            "hazard_density": 0.15,
            "collectible_distance": 0.15
        }
        
    def calculate_section_difficulty(
        self,
        section_index: int,
        total_sections: int,
        base_difficulty: float = 0.1
    ) -> float:
        """Calculate difficulty for a specific section."""
        # Gradual increase with some variation
        progress = section_index / max(total_sections - 1, 1)
        difficulty = base_difficulty + (0.8 * progress)
        
        # Add some wave pattern for interesting pacing
        wave = 0.1 * np.sin(progress * np.pi * 2)
        difficulty = max(0.05, min(0.95, difficulty + wave))
        
        return difficulty
        
    def apply_difficulty_to_elements(
        self,
        elements: List[LevelElement],
        difficulty: float
    ) -> List[LevelElement]:
        """Modify elements based on difficulty level."""
        for element in elements:
            if element.element_type == "enemy":
                # More enemies, faster movement at higher difficulty
                element.properties["speed"] = 1.0 + difficulty * 2.0
                element.properties["health"] = int(1 + difficulty * 3)
                
            elif element.element_type == "platform":
                # Smaller, more precise platforms at higher difficulty
                base_width = element.properties.get("width", 2.0)
                element.properties["width"] = base_width * (1.2 - difficulty * 0.5)
                
            elif element.element_type == "collectible":
                # Harder to reach collectibles
                element.properties["bonus_points"] = int(10 * (1 + difficulty))
                
        return elements


class ProceduralGenerator:
    """Generates level content procedurally."""
    
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.noise_cache: Dict[str, np.ndarray] = {}
        
    def generate_platform_layout(
        self,
        width: int,
        height: int,
        density: float = 0.3
    ) -> List[LevelElement]:
        """Generate platform layout using procedural methods."""
        platforms = []
        
        # Generate base terrain
        terrain_height = self._generate_terrain_height(width)
        
        # Add platforms based on terrain
        for x in range(0, width, 2):
            if self.random.random() < density:
                y = terrain_height[x] + self.random.randint(1, 4)
                platform_width = self.random.choice([1, 2, 3, 4])
                
                platform = LevelElement(
                    element_type="platform",
                    position=(x, y),
                    size=(platform_width, 0.5),
                    properties={
                        "material": self.random.choice(["stone", "wood", "metal"]),
                        "width": platform_width,
                        "solid": True
                    }
                )
                platforms.append(platform)
                
        return platforms
        
    def generate_enemy_placement(
        self,
        platforms: List[LevelElement],
        difficulty: float
    ) -> List[LevelElement]:
        """Generate enemy placements based on platforms and difficulty."""
        enemies = []
        enemy_count = int(len(platforms) * 0.3 * (1 + difficulty))
        
        enemy_types = ["goomba", "koopa", "spiky", "flying"]
        
        for _ in range(enemy_count):
            if platforms:
                platform = self.random.choice(platforms)
                enemy_type = self.random.choice(enemy_types)
                
                enemy = LevelElement(
                    element_type="enemy",
                    position=(
                        platform.position[0] + self.random.uniform(-1, 1),
                        platform.position[1] + 1
                    ),
                    size=(1, 1),
                    properties={
                        "enemy_type": enemy_type,
                        "speed": 1.0 + difficulty * 0.5,
                        "health": int(1 + difficulty * 2),
                        "patrol_range": self.random.uniform(2, 6),
                        "aggressive": difficulty > 0.5
                    }
                )
                enemies.append(enemy)
                
        return enemies
        
    def generate_collectibles(
        self,
        platforms: List[LevelElement],
        count: int = 10
    ) -> List[LevelElement]:
        """Generate collectible items."""
        collectibles = []
        collectible_types = ["coin", "gem", "powerup", "health"]
        
        for _ in range(count):
            if platforms:
                platform = self.random.choice(platforms)
                collectible_type = self.random.choice(collectible_types)
                
                collectible = LevelElement(
                    element_type="collectible",
                    position=(
                        platform.position[0],
                        platform.position[1] + 1.5
                    ),
                    size=(0.5, 0.5),
                    properties={
                        "collectible_type": collectible_type,
                        "points": 10 if collectible_type == "coin" else 50,
                        "effect": self._get_collectible_effect(collectible_type)
                    }
                )
                collectibles.append(collectible)
                
        return collectibles
        
    def generate_hazards(
        self,
        width: int,
        height: int,
        difficulty: float
    ) -> List[LevelElement]:
        """Generate hazards and obstacles."""
        hazards = []
        hazard_count = int(width * 0.1 * (1 + difficulty))
        
        hazard_types = ["spikes", "pit", "lava", "saw", "falling_block"]
        
        for _ in range(hazard_count):
            hazard_type = self.random.choice(hazard_types)
            
            hazard = LevelElement(
                element_type="hazard",
                position=(
                    self.random.uniform(5, width - 5),
                    self.random.uniform(0, height * 0.7)
                ),
                size=(1, 1),
                properties={
                    "hazard_type": hazard_type,
                    "damage": int(1 + difficulty * 2),
                    "active": True,
                    "animation_speed": 1.0 + difficulty
                }
            )
            hazards.append(hazard)
            
        return hazards
        
    def _generate_terrain_height(self, width: int) -> List[int]:
        """Generate base terrain height using Perlin noise."""
        if "terrain" not in self.noise_cache:
            # Simple noise generation (in real implementation, use proper Perlin noise)
            terrain = []
            height = 3
            for x in range(width):
                # Simple random walk
                height += self.random.randint(-1, 1)
                height = max(1, min(8, height))
                terrain.append(height)
            self.noise_cache["terrain"] = terrain
            
        return self.noise_cache["terrain"][:width]
        
    def _get_collectible_effect(self, collectible_type: str) -> Dict[str, Any]:
        """Get effect properties for collectible types."""
        effects = {
            "coin": {"type": "points", "value": 10},
            "gem": {"type": "points", "value": 50},
            "powerup": {"type": "speed_boost", "duration": 5},
            "health": {"type": "heal", "value": 1}
        }
        return effects.get(collectible_type, {"type": "points", "value": 10})


class LevelValidator:
    """Validates generated levels for playability."""
    
    def __init__(self):
        self.min_platform_distance = 3.0
        self.max_platform_gap = 6.0
        self.min_collectible_spacing = 2.0
        
    def validate_level(self, sections: List[LevelSection]) -> Tuple[bool, List[str]]:
        """Validate a complete level."""
        issues = []
        
        for i, section in enumerate(sections):
            section_issues = self.validate_section(section, i)
            issues.extend(section_issues)
            
        # Check overall level flow
        flow_issues = self._validate_level_flow(sections)
        issues.extend(flow_issues)
        
        return len(issues) == 0, issues
        
    def validate_section(self, section: LevelSection, index: int) -> List[str]:
        """Validate a single level section."""
        issues = []
        
        platforms = [e for e in section.elements if e.element_type == "platform"]
        enemies = [e for e in section.elements if e.element_type == "enemy"]
        collectibles = [e for e in section.elements if e.element_type == "collectible"]
        
        # Check platform connectivity
        if not self._validate_platform_connectivity(platforms):
            issues.append(f"Section {index}: Platforms not properly connected")
            
        # Check enemy placement
        if not self._validate_enemy_placement(enemies, platforms):
            issues.append(f"Section {index}: Enemies placed in unreachable locations")
            
        # Check collectible accessibility
        if not self._validate_collectible_accessibility(collectibles, platforms):
            issues.append(f"Section {index}: Some collectibles are unreachable")
            
        return issues
        
    def _validate_platform_connectivity(self, platforms: List[LevelElement]) -> bool:
        """Check if platforms form a connected path."""
        if len(platforms) < 2:
            return True
            
        # Simple connectivity check - more sophisticated pathfinding needed in practice
        platforms_sorted = sorted(platforms, key=lambda p: p.position[0])
        
        for i in range(len(platforms_sorted) - 1):
            current = platforms_sorted[i]
            next_platform = platforms_sorted[i + 1]
            
            distance = abs(next_platform.position[0] - current.position[0])
            height_diff = abs(next_platform.position[1] - current.position[1])
            
            if distance > self.max_platform_gap or height_diff > 4:
                return False
                
        return True
        
    def _validate_enemy_placement(
        self, 
        enemies: List[LevelElement],
        platforms: List[LevelElement]
    ) -> bool:
        """Check if enemies are placed on or near platforms."""
        for enemy in enemies:
            near_platform = any(
                abs(enemy.position[0] - platform.position[0]) < 3 and
                abs(enemy.position[1] - platform.position[1]) < 2
                for platform in platforms
            )
            if not near_platform:
                return False
        return True
        
    def _validate_collectible_accessibility(
        self,
        collectibles: List[LevelElement],
        platforms: List[LevelElement]
    ) -> bool:
        """Check if collectibles are accessible from platforms."""
        for collectible in collectibles:
            accessible = any(
                abs(collectible.position[0] - platform.position[0]) < 2 and
                collectible.position[1] >= platform.position[1]
                for platform in platforms
            )
            if not accessible:
                return False
        return True
        
    def _validate_level_flow(self, sections: List[LevelSection]) -> List[str]:
        """Validate overall level flow and pacing."""
        issues = []
        
        if not sections:
            issues.append("Level has no sections")
            return issues
            
        # Check difficulty progression
        difficulties = [section.difficulty for section in sections]
        for i in range(1, len(difficulties)):
            if difficulties[i] < difficulties[i-1] - 0.3:
                issues.append(f"Difficulty drops too sharply between sections {i-1} and {i}")
                
        # Check estimated time distribution
        times = [section.estimated_time for section in sections]
        if max(times) > min(times) * 5:
            issues.append("Section completion times vary too widely")
            
        return issues


class LabyrinthTools:
    """Tools available to the Labyrinth Agent."""
    
    @staticmethod
    def create_generate_platformer_level_tool(agent: "LabyrinthAgent") -> BaseTool:
        """Tool for generating platformer levels."""
        
        class PlatformerLevelInput(LangchainBaseModel):
            name: str = Field(description="Level name")
            theme: str = Field(description="Level theme (forest, castle, city, space)")
            difficulty: float = Field(description="Base difficulty (0.0 to 1.0)")
            section_count: int = Field(description="Number of level sections")
            mechanics: List[str] = Field(description="Required game mechanics")
            
        async def generate_platformer_level(
            name: str,
            theme: str,
            difficulty: float,
            section_count: int,
            mechanics: List[str]
        ) -> str:
            try:
                level_data = await agent._generate_platformer_level(
                    name, theme, difficulty, section_count, mechanics
                )
                return f"Generated {name} platformer level with {section_count} sections"
            except Exception as e:
                return f"Error generating platformer level: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_platformer_level,
            name="generate_platformer_level",
            description="Generate a complete platformer level",
            args_schema=PlatformerLevelInput
        )
        
    @staticmethod
    def create_generate_puzzle_level_tool(agent: "LabyrinthAgent") -> BaseTool:
        """Tool for generating puzzle levels."""
        
        class PuzzleLevelInput(LangchainBaseModel):
            name: str = Field(description="Level name")
            puzzle_type: str = Field(description="Type of puzzle (switch, block, key)")
            complexity: int = Field(description="Puzzle complexity (1-5)")
            
        async def generate_puzzle_level(
            name: str,
            puzzle_type: str,
            complexity: int
        ) -> str:
            try:
                level_data = await agent._generate_puzzle_level(
                    name, puzzle_type, complexity
                )
                return f"Generated {name} puzzle level (type: {puzzle_type})"
            except Exception as e:
                return f"Error generating puzzle level: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_puzzle_level,
            name="generate_puzzle_level",
            description="Generate a puzzle-based level",
            args_schema=PuzzleLevelInput
        )
        
    @staticmethod
    def create_optimize_level_difficulty_tool(agent: "LabyrinthAgent") -> BaseTool:
        """Tool for optimizing level difficulty curves."""
        
        class OptimizeDifficultyInput(LangchainBaseModel):
            level_id: str = Field(description="Level ID to optimize")
            target_completion_time: float = Field(description="Target completion time in seconds")
            
        async def optimize_level_difficulty(
            level_id: str,
            target_completion_time: float
        ) -> str:
            try:
                result = await agent._optimize_level_difficulty(
                    level_id, target_completion_time
                )
                return f"Optimized difficulty for level {level_id}"
            except Exception as e:
                return f"Error optimizing difficulty: {str(e)}"
                
        return StructuredTool.from_function(
            func=optimize_level_difficulty,
            name="optimize_level_difficulty",
            description="Optimize level difficulty curve",
            args_schema=OptimizeDifficultyInput
        )


class LabyrinthAgent(BaseAgent):
    """Specialist agent for procedural level design and gameplay progression."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="labyrinth_agent",
            agent_type=AgentType.LABYRINTH,
            **kwargs
        )
        
        self.generator = ProceduralGenerator()
        self.difficulty = DifficultyProgression()
        self.validator = LevelValidator()
        self.generated_levels: Dict[str, LevelData] = {}
        
        # Create tools
        self.tools = [
            LabyrinthTools.create_generate_platformer_level_tool(self),
            LabyrinthTools.create_generate_puzzle_level_tool(self),
            LabyrinthTools.create_optimize_level_difficulty_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for level design."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Labyrinth agent."""
        return """
        You are the Labyrinth Agent, a specialist in procedural level design and gameplay progression.
        
        Your expertise includes:
        - Procedural content generation for game levels
        - Difficulty curve design and balancing
        - Player progression and pacing
        - Level connectivity and flow
        - Gameplay mechanic integration
        - Performance optimization for mobile platforms
        
        Level design capabilities:
        - 2D platformer levels with varied layouts
        - Puzzle levels with logical progression  
        - Enemy placement and AI behaviors
        - Collectible distribution and rewards
        - Environmental storytelling through level design
        - Accessibility considerations
        
        Design principles:
        - Player-centric design focused on fun and engagement
        - Clear visual communication of interactive elements
        - Balanced risk-reward mechanics
        - Smooth difficulty progression
        - Replayability through varied paths and secrets
        - Mobile-friendly controls and screen space usage
        
        Technical considerations:
        - Memory-efficient level data structures
        - Streaming and loading optimization
        - Collision detection optimization
        - Visual clarity at small screen sizes
        - Performance scaling for different device capabilities
        
        Always design levels that are intuitive, engaging, and technically sound.
        Consider the target audience and ensure appropriate challenge without frustration.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a level design task."""
        
        action = task.action
        params = task.parameters
        
        if action == "generate_level":
            return await self._handle_level_generation(params)
        elif action == "optimize_difficulty":
            return await self._handle_difficulty_optimization(params)
        elif action == "validate_level":
            return await self._handle_level_validation(params)
        else:
            # Use agent executor for complex level design tasks
            result = await self.agent_executor.ainvoke({
                "input": f"Design level for: {action} with requirements: {params}"
            })
            return {"result": result["output"]}
            
    async def _handle_level_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle level generation requests."""
        
        level = params.get("level", {})
        gameplay_mechanics = params.get("gameplay_mechanics", [])
        project_id = params.get("project_id", "")
        
        name = level.get("name", "level")
        theme = level.get("theme", "forest")
        difficulty = level.get("difficulty", 0.5)
        level_type = level.get("type", "platformer")
        
        self.log_agent_action("generate_level", "started",
                            level=name, project_id=project_id)
        
        try:
            if level_type == "platformer":
                level_data = await self._generate_platformer_level(
                    name, theme, difficulty, 5, gameplay_mechanics
                )
            elif level_type == "puzzle":
                level_data = await self._generate_puzzle_level(
                    name, "switch", int(difficulty * 5) + 1
                )
            else:
                # Default to platformer
                level_data = await self._generate_platformer_level(
                    name, theme, difficulty, 5, gameplay_mechanics
                )
                
            # Store level data
            asset_key = f"{project_id}_{name}_level"
            self.generated_levels[asset_key] = level_data
            
            # Save level data to file
            await self._save_level_data(level_data, project_id)
            
            # Publish asset generated event
            await publish_event(
                EventType.ASSET_GENERATED,
                "labyrinth_agent",
                {
                    "asset_type": "level",
                    "filename": f"{name}_level.json",
                    "level": name,
                    "project_id": project_id,
                    "difficulty": difficulty,
                    "sections": len(level_data.sections)
                }
            )
            
            return {
                "level_id": level_data.level_id,
                "name": level_data.name,
                "difficulty": level_data.difficulty,
                "sections": len(level_data.sections),
                "estimated_duration": level_data.estimated_duration
            }
            
        except Exception as e:
            self.log_agent_error("generate_level", e,
                               level=name, project_id=project_id)
            raise
            
    async def _generate_platformer_level(
        self,
        name: str,
        theme: str,
        difficulty: float,
        section_count: int,
        mechanics: List[str]
    ) -> LevelData:
        """Generate a complete platformer level."""
        
        sections = []
        total_duration = 0.0
        
        for i in range(section_count):
            section_difficulty = self.difficulty.calculate_section_difficulty(
                i, section_count, difficulty
            )
            
            # Generate section elements
            platforms = self.generator.generate_platform_layout(20, 10, 0.4)
            enemies = self.generator.generate_enemy_placement(platforms, section_difficulty)
            collectibles = self.generator.generate_collectibles(platforms, 8)
            hazards = self.generator.generate_hazards(20, 10, section_difficulty)
            
            # Apply difficulty scaling
            all_elements = platforms + enemies + collectibles + hazards
            all_elements = self.difficulty.apply_difficulty_to_elements(
                all_elements, section_difficulty
            )
            
            # Estimate completion time
            estimated_time = self._estimate_section_time(all_elements, section_difficulty)
            total_duration += estimated_time
            
            section = LevelSection(
                section_id=f"{name}_section_{i}",
                difficulty=section_difficulty,
                elements=all_elements,
                connections=[f"{name}_section_{j}" for j in [i-1, i+1] if 0 <= j < section_count and j != i],
                theme=theme,
                estimated_time=estimated_time
            )
            
            sections.append(section)
            
        # Validate level
        is_valid, issues = self.validator.validate_level(sections)
        if not is_valid:
            self.logger.warning(f"Level validation issues: {issues}")
            # In a full implementation, we'd fix the issues or regenerate
            
        level_data = LevelData(
            level_id=f"{name}_{theme}_{int(difficulty*10)}",
            name=name,
            difficulty=difficulty,
            estimated_duration=total_duration,
            sections=[asdict(section) for section in sections],
            metadata={
                "theme": theme,
                "mechanics": mechanics,
                "section_count": section_count,
                "validation_issues": issues
            },
            generation_params={
                "generator_seed": self.generator.random.getstate()[1][0],
                "difficulty_base": difficulty,
                "theme": theme
            }
        )
        
        return level_data
        
    async def _generate_puzzle_level(
        self,
        name: str,
        puzzle_type: str,
        complexity: int
    ) -> LevelData:
        """Generate a puzzle-based level."""
        
        # Simplified puzzle level generation
        section = LevelSection(
            section_id=f"{name}_puzzle",
            difficulty=complexity / 5.0,
            elements=[],
            connections=[],
            theme="puzzle",
            estimated_time=60.0 * complexity
        )
        
        # Add puzzle elements based on type
        if puzzle_type == "switch":
            section.elements.extend(self._generate_switch_puzzle(complexity))
        elif puzzle_type == "block":
            section.elements.extend(self._generate_block_puzzle(complexity))
        elif puzzle_type == "key":
            section.elements.extend(self._generate_key_puzzle(complexity))
            
        level_data = LevelData(
            level_id=f"{name}_puzzle_{complexity}",
            name=name,
            difficulty=complexity / 5.0,
            estimated_duration=section.estimated_time,
            sections=[asdict(section)],
            metadata={
                "puzzle_type": puzzle_type,
                "complexity": complexity
            },
            generation_params={
                "puzzle_type": puzzle_type,
                "complexity": complexity
            }
        )
        
        return level_data
        
    def _generate_switch_puzzle(self, complexity: int) -> List[LevelElement]:
        """Generate switch-based puzzle elements."""
        elements = []
        
        for i in range(complexity):
            switch = LevelElement(
                element_type="switch",
                position=(i * 3, 2),
                size=(1, 1),
                properties={
                    "switch_id": f"switch_{i}",
                    "state": False,
                    "connected_doors": [f"door_{i}"]
                }
            )
            elements.append(switch)
            
            door = LevelElement(
                element_type="door",
                position=(i * 3 + 10, 2),
                size=(1, 2),
                properties={
                    "door_id": f"door_{i}",
                    "open": False,
                    "required_switches": [f"switch_{i}"]
                }
            )
            elements.append(door)
            
        return elements
        
    def _generate_block_puzzle(self, complexity: int) -> List[LevelElement]:
        """Generate block-pushing puzzle elements."""
        elements = []
        
        for i in range(complexity):
            block = LevelElement(
                element_type="pushable_block",
                position=(i * 2, 1),
                size=(1, 1),
                properties={
                    "moveable": True,
                    "weight": 1,
                    "target_position": (i * 2 + 5, 1)
                }
            )
            elements.append(block)
            
            target = LevelElement(
                element_type="target",
                position=(i * 2 + 5, 1),
                size=(1, 0.1),
                properties={
                    "target_id": f"target_{i}",
                    "activated": False
                }
            )
            elements.append(target)
            
        return elements
        
    def _generate_key_puzzle(self, complexity: int) -> List[LevelElement]:
        """Generate key-and-lock puzzle elements."""
        elements = []
        
        for i in range(complexity):
            key = LevelElement(
                element_type="key",
                position=(i * 4, 3),
                size=(0.5, 0.5),
                properties={
                    "key_id": f"key_{i}",
                    "color": ["red", "blue", "green", "yellow"][i % 4]
                }
            )
            elements.append(key)
            
            lock = LevelElement(
                element_type="lock",
                position=(i * 4 + 8, 2),
                size=(1, 1),
                properties={
                    "lock_id": f"lock_{i}",
                    "required_key": f"key_{i}",
                    "color": key.properties["color"]
                }
            )
            elements.append(lock)
            
        return elements
        
    def _estimate_section_time(self, elements: List[LevelElement], difficulty: float) -> float:
        """Estimate time to complete a section."""
        base_time = 30.0  # Base 30 seconds
        
        # Add time based on element count and types
        platform_count = len([e for e in elements if e.element_type == "platform"])
        enemy_count = len([e for e in elements if e.element_type == "enemy"])
        collectible_count = len([e for e in elements if e.element_type == "collectible"])
        
        time_estimate = base_time + (platform_count * 2) + (enemy_count * 5) + (collectible_count * 1)
        
        # Scale by difficulty
        time_estimate *= (1 + difficulty * 0.5)
        
        return time_estimate
        
    async def _save_level_data(self, level_data: LevelData, project_id: str) -> None:
        """Save level data to file."""
        filename = f"{level_data.name}_level.json"
        output_path = settings.data_dir / "assets" / "levels" / project_id / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(level_data.dict(), f, indent=2)
            
    async def _handle_difficulty_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle difficulty optimization requests."""
        level_id = params.get("level_id", "")
        target_time = params.get("target_completion_time", 120.0)
        
        result = await self._optimize_level_difficulty(level_id, target_time)
        
        return {
            "level_id": level_id,
            "optimization_result": result
        }
        
    async def _optimize_level_difficulty(
        self,
        level_id: str,
        target_completion_time: float
    ) -> Dict[str, Any]:
        """Optimize level difficulty to match target completion time."""
        
        # Find the level
        level_data = None
        for key, data in self.generated_levels.items():
            if data.level_id == level_id:
                level_data = data
                break
                
        if not level_data:
            raise ValueError(f"Level {level_id} not found")
            
        current_time = level_data.estimated_duration
        time_ratio = target_completion_time / current_time
        
        # Adjust difficulty based on time ratio
        if time_ratio < 0.8:  # Need to make harder
            difficulty_adjustment = 0.2
        elif time_ratio > 1.2:  # Need to make easier
            difficulty_adjustment = -0.2
        else:
            difficulty_adjustment = 0.0
            
        # Apply adjustments (simplified)
        new_difficulty = max(0.1, min(0.9, level_data.difficulty + difficulty_adjustment))
        
        return {
            "original_difficulty": level_data.difficulty,
            "new_difficulty": new_difficulty,
            "original_time": current_time,
            "target_time": target_completion_time,
            "adjustment": difficulty_adjustment
        }
        
    async def _handle_level_validation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle level validation requests."""
        level_id = params.get("level_id", "")
        
        # Find and validate the level
        level_data = None
        for key, data in self.generated_levels.items():
            if data.level_id == level_id:
                level_data = data
                break
                
        if not level_data:
            return {"valid": False, "error": f"Level {level_id} not found"}
            
        # Convert sections back to LevelSection objects for validation
        sections = []
        for section_data in level_data.sections:
            elements = [
                LevelElement(
                    element_type=elem["element_type"],
                    position=tuple(elem["position"]),
                    size=tuple(elem["size"]),
                    properties=elem["properties"]
                )
                for elem in section_data["elements"]
            ]
            
            section = LevelSection(
                section_id=section_data["section_id"],
                difficulty=section_data["difficulty"],
                elements=elements,
                connections=section_data["connections"],
                theme=section_data["theme"],
                estimated_time=section_data["estimated_time"]
            )
            sections.append(section)
            
        is_valid, issues = self.validator.validate_level(sections)
        
        return {
            "valid": is_valid,
            "issues": issues,
            "level_id": level_id
        }