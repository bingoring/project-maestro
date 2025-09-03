"""Game Design Document Parser and Analyzer.

This module provides comprehensive parsing and analysis of game design documents
to extract structured requirements for agent task generation.
"""

import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from langchain.schema import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..core.logging import get_logger

logger = get_logger("gdd_parser")


class GameGenre(Enum):
    """Supported game genres."""
    PLATFORMER = "platformer"
    PUZZLE = "puzzle" 
    ACTION = "action"
    RPG = "rpg"
    STRATEGY = "strategy"
    RACING = "racing"
    SHOOTER = "shooter"
    ADVENTURE = "adventure"
    SIMULATION = "simulation"
    CASUAL = "casual"


class Platform(Enum):
    """Target platforms."""
    ANDROID = "Android"
    IOS = "iOS"
    WEBGL = "WebGL"
    WINDOWS = "Windows"
    MACOS = "macOS"
    LINUX = "Linux"


@dataclass
class GameMetadata:
    """Basic game metadata."""
    title: str
    genre: GameGenre
    target_platforms: List[Platform]
    description: str
    target_audience: str
    estimated_playtime: Optional[int] = None  # minutes
    complexity_score: float = 0.5  # 0.0 to 1.0


@dataclass
class Character:
    """Character definition."""
    name: str
    role: str  # player, enemy, npc
    description: str
    abilities: List[str]
    visual_style: str
    animations_needed: List[str]


@dataclass
class GameplayMechanic:
    """Core gameplay mechanic."""
    name: str
    description: str
    implementation_complexity: str  # simple, moderate, complex
    required_assets: List[str]
    code_requirements: List[str]


@dataclass
class Level:
    """Level/stage definition."""
    name: str
    theme: str
    objectives: List[str]
    enemies: List[str]
    mechanics: List[str]
    estimated_playtime: int  # minutes
    difficulty: str  # easy, medium, hard


@dataclass
class AudioRequirement:
    """Audio asset requirement."""
    name: str
    type: str  # bgm, sfx
    mood: str
    loop: bool
    triggers: List[str]  # When this audio should play


@dataclass
class VisualAsset:
    """Visual asset requirement."""
    name: str
    type: str  # sprite, background, ui, animation
    style: str
    resolution: str
    description: str
    variations: int = 1


class ParsedGameDesign(BaseModel):
    """Structured game design document output."""
    metadata: Dict[str, Any] = Field(description="Game metadata")
    characters: List[Dict[str, Any]] = Field(description="Character definitions")
    mechanics: List[Dict[str, Any]] = Field(description="Gameplay mechanics")
    levels: List[Dict[str, Any]] = Field(description="Level definitions")
    audio_requirements: List[Dict[str, Any]] = Field(description="Audio requirements")
    visual_assets: List[Dict[str, Any]] = Field(description="Visual asset requirements")
    technical_requirements: Dict[str, Any] = Field(description="Technical specifications")


class GameDesignDocumentParser:
    """Parser for game design documents using LLM analysis."""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=ParsedGameDesign)
        
        self.parsing_prompt = PromptTemplate(
            template="""You are an expert game design analyst. Parse the following game design document 
            and extract structured information for automated game development.

            Game Design Document:
            {document_text}

            Extract the following information in the specified JSON format:

            1. METADATA:
            - Title, genre, target platforms, description
            - Target audience, estimated playtime
            - Complexity score (0.0-1.0 based on implementation difficulty)

            2. CHARACTERS:
            - Name, role (player/enemy/npc), description
            - Special abilities, visual style requirements
            - Required animations (idle, walk, jump, attack, etc.)

            3. GAMEPLAY MECHANICS:
            - Core mechanics with descriptions
            - Implementation complexity (simple/moderate/complex)
            - Required assets and code components

            4. LEVELS/STAGES:
            - Level themes, objectives, enemy types
            - Difficulty progression, estimated playtime per level
            - Special mechanics per level

            5. AUDIO REQUIREMENTS:
            - Background music (mood, style, loop points)
            - Sound effects (trigger conditions, variations)

            6. VISUAL ASSETS:
            - Sprites, backgrounds, UI elements
            - Art style, resolution requirements
            - Animation needs and variations

            7. TECHNICAL REQUIREMENTS:
            - Unity version, required packages
            - Platform-specific optimizations
            - Performance targets

            {format_instructions}

            Parsed Game Design:""",
            input_variables=["document_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def parse_document(self, document_text: str) -> ParsedGameDesign:
        """Parse a game design document into structured data."""
        try:
            logger.info("Starting game design document parsing")
            
            # Clean and preprocess the document
            cleaned_text = self._preprocess_document(document_text)
            
            # Run LLM parsing
            prompt_value = self.parsing_prompt.format_prompt(document_text=cleaned_text)
            llm_output = self.llm(prompt_value.to_string())
            
            # Parse the structured output
            parsed_design = self.output_parser.parse(llm_output)
            
            # Post-process and validate
            validated_design = self._validate_and_enhance(parsed_design)
            
            logger.info(
                "Successfully parsed game design document",
                title=validated_design.metadata.get("title", "Unknown"),
                num_characters=len(validated_design.characters),
                num_levels=len(validated_design.levels),
                complexity_score=validated_design.metadata.get("complexity_score", 0.5)
            )
            
            return validated_design
            
        except Exception as e:
            logger.error("Failed to parse game design document", error=str(e))
            raise

    def _preprocess_document(self, text: str) -> str:
        """Clean and normalize the document text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize headers
        text = re.sub(r'#{4,}', '###', text)  # Limit header depth
        
        # Remove or replace special characters that might confuse LLM
        text = text.replace('\r', '\n')
        text = re.sub(r'[^\w\s\n\r\.,!?;:()\-\'\"/#]', '', text)
        
        return text.strip()

    def _validate_and_enhance(self, parsed_design: ParsedGameDesign) -> ParsedGameDesign:
        """Validate and enhance the parsed design with defaults and corrections."""
        
        # Validate metadata
        metadata = parsed_design.metadata
        if not metadata.get("title"):
            metadata["title"] = "Untitled Game"
            
        if not metadata.get("genre"):
            metadata["genre"] = "casual"
            
        if not metadata.get("complexity_score"):
            # Calculate complexity based on content
            complexity = self._calculate_complexity_score(parsed_design)
            metadata["complexity_score"] = complexity
            
        # Ensure required character animations
        for character in parsed_design.characters:
            if character.get("role") == "player" and not character.get("animations_needed"):
                character["animations_needed"] = ["idle", "walk", "jump"]
                
        # Validate level progression
        if parsed_design.levels:
            for i, level in enumerate(parsed_design.levels):
                if not level.get("difficulty"):
                    if i < len(parsed_design.levels) // 3:
                        level["difficulty"] = "easy"
                    elif i < 2 * len(parsed_design.levels) // 3:
                        level["difficulty"] = "medium"
                    else:
                        level["difficulty"] = "hard"
        
        # Add default audio requirements if missing
        if not parsed_design.audio_requirements:
            parsed_design.audio_requirements = [
                {
                    "name": "main_theme",
                    "type": "bgm",
                    "mood": "upbeat",
                    "loop": True,
                    "triggers": ["game_start", "main_menu"]
                },
                {
                    "name": "jump_sound",
                    "type": "sfx", 
                    "mood": "action",
                    "loop": False,
                    "triggers": ["player_jump"]
                }
            ]
            
        # Ensure minimum visual assets
        required_assets = ["player_sprite", "background", "ui_elements"]
        existing_assets = {asset.get("name") for asset in parsed_design.visual_assets}
        
        for required in required_assets:
            if required not in existing_assets:
                parsed_design.visual_assets.append({
                    "name": required,
                    "type": "sprite" if "sprite" in required else "ui" if "ui" in required else "background",
                    "style": metadata.get("art_style", "pixel"),
                    "resolution": "mobile_optimized",
                    "description": f"Required {required.replace('_', ' ')}",
                    "variations": 1
                })
                
        return parsed_design

    def _calculate_complexity_score(self, parsed_design: ParsedGameDesign) -> float:
        """Calculate implementation complexity score (0.0 to 1.0)."""
        score = 0.0
        
        # Base complexity from mechanics
        complex_mechanics = sum(
            1 for mechanic in parsed_design.mechanics 
            if mechanic.get("implementation_complexity") == "complex"
        )
        moderate_mechanics = sum(
            1 for mechanic in parsed_design.mechanics
            if mechanic.get("implementation_complexity") == "moderate"
        )
        
        score += complex_mechanics * 0.3 + moderate_mechanics * 0.15
        
        # Complexity from number of characters
        score += min(len(parsed_design.characters) * 0.1, 0.3)
        
        # Complexity from number of levels
        score += min(len(parsed_design.levels) * 0.05, 0.2)
        
        # Complexity from audio/visual assets
        score += min(len(parsed_design.audio_requirements) * 0.02, 0.1)
        score += min(len(parsed_design.visual_assets) * 0.01, 0.1)
        
        # Platform complexity
        platforms = parsed_design.metadata.get("target_platforms", [])
        if len(platforms) > 2:
            score += 0.1
            
        return min(score, 1.0)

    def extract_asset_requirements(self, parsed_design: ParsedGameDesign) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and organize asset requirements for each specialist agent."""
        
        requirements = {
            "code": [],
            "visual": [],
            "audio": [],
            "level": []
        }
        
        # Code requirements
        for mechanic in parsed_design.mechanics:
            for code_req in mechanic.get("code_requirements", []):
                requirements["code"].append({
                    "type": "mechanic",
                    "name": mechanic["name"],
                    "requirement": code_req,
                    "complexity": mechanic.get("implementation_complexity", "simple")
                })
                
        for character in parsed_design.characters:
            if character.get("role") == "player":
                requirements["code"].append({
                    "type": "player_controller",
                    "name": f"{character['name']}_controller",
                    "abilities": character.get("abilities", []),
                    "complexity": "moderate"
                })
                
        # Visual requirements
        requirements["visual"] = parsed_design.visual_assets
        
        # Audio requirements  
        requirements["audio"] = parsed_design.audio_requirements
        
        # Level requirements
        requirements["level"] = parsed_design.levels
        
        return requirements


class GameComplexityAnalyzer:
    """Analyzer for assessing game development complexity and effort estimation."""
    
    def __init__(self):
        self.complexity_weights = {
            "mechanics": 0.4,
            "characters": 0.2,
            "levels": 0.15,
            "assets": 0.15,
            "platforms": 0.1
        }
        
    def analyze_development_effort(self, parsed_design: ParsedGameDesign) -> Dict[str, Any]:
        """Analyze development effort and provide time estimates."""
        
        complexity_breakdown = self._analyze_complexity_breakdown(parsed_design)
        time_estimates = self._estimate_development_time(complexity_breakdown, parsed_design)
        agent_workload = self._estimate_agent_workload(parsed_design)
        risks = self._identify_development_risks(parsed_design)
        
        return {
            "complexity_breakdown": complexity_breakdown,
            "time_estimates": time_estimates,
            "agent_workload": agent_workload,
            "development_risks": risks,
            "recommendations": self._generate_recommendations(complexity_breakdown, risks)
        }
    
    def _analyze_complexity_breakdown(self, parsed_design: ParsedGameDesign) -> Dict[str, float]:
        """Break down complexity by category."""
        
        # Mechanics complexity
        mechanics_score = 0.0
        for mechanic in parsed_design.mechanics:
            complexity = mechanic.get("implementation_complexity", "simple")
            if complexity == "complex":
                mechanics_score += 0.8
            elif complexity == "moderate":
                mechanics_score += 0.5
            else:
                mechanics_score += 0.2
                
        mechanics_score = min(mechanics_score / max(len(parsed_design.mechanics), 1), 1.0)
        
        # Character complexity
        character_score = min(len(parsed_design.characters) * 0.3, 1.0)
        
        # Level complexity
        level_score = min(len(parsed_design.levels) * 0.2, 1.0)
        
        # Asset complexity
        asset_count = len(parsed_design.visual_assets) + len(parsed_design.audio_requirements)
        asset_score = min(asset_count * 0.05, 1.0)
        
        # Platform complexity
        platform_count = len(parsed_design.metadata.get("target_platforms", []))
        platform_score = min((platform_count - 1) * 0.3, 1.0)
        
        return {
            "mechanics": mechanics_score,
            "characters": character_score,
            "levels": level_score,
            "assets": asset_score,
            "platforms": platform_score
        }
    
    def _estimate_development_time(self, complexity: Dict[str, float], parsed_design: ParsedGameDesign) -> Dict[str, int]:
        """Estimate development time in minutes for each phase."""
        
        base_times = {
            "planning": 30,
            "code_generation": 120,
            "asset_creation": 180,
            "level_design": 60,
            "integration": 90,
            "testing": 45,
            "build": 30
        }
        
        # Apply complexity multipliers
        overall_complexity = sum(
            complexity[key] * self.complexity_weights[key]
            for key in complexity.keys()
        )
        
        multiplier = 1.0 + overall_complexity
        
        estimated_times = {
            phase: int(base_time * multiplier)
            for phase, base_time in base_times.items()
        }
        
        # Add specific adjustments
        estimated_times["code_generation"] += len(parsed_design.mechanics) * 15
        estimated_times["asset_creation"] += (len(parsed_design.visual_assets) + len(parsed_design.audio_requirements)) * 10
        estimated_times["level_design"] += len(parsed_design.levels) * 20
        
        return estimated_times
    
    def _estimate_agent_workload(self, parsed_design: ParsedGameDesign) -> Dict[str, Dict[str, Any]]:
        """Estimate workload for each specialist agent."""
        
        return {
            "codex": {
                "tasks": len(parsed_design.mechanics) + len(parsed_design.characters),
                "estimated_time": len(parsed_design.mechanics) * 20 + len(parsed_design.characters) * 15,
                "complexity": "moderate"
            },
            "canvas": {
                "tasks": len(parsed_design.visual_assets),
                "estimated_time": len(parsed_design.visual_assets) * 25,
                "complexity": "high" if len(parsed_design.visual_assets) > 20 else "moderate"
            },
            "sonata": {
                "tasks": len(parsed_design.audio_requirements),
                "estimated_time": len(parsed_design.audio_requirements) * 15,
                "complexity": "low"
            },
            "labyrinth": {
                "tasks": len(parsed_design.levels),
                "estimated_time": len(parsed_design.levels) * 30,
                "complexity": "moderate"
            },
            "builder": {
                "tasks": len(parsed_design.metadata.get("target_platforms", [])),
                "estimated_time": len(parsed_design.metadata.get("target_platforms", [])) * 20,
                "complexity": "low"
            }
        }
    
    def _identify_development_risks(self, parsed_design: ParsedGameDesign) -> List[Dict[str, str]]:
        """Identify potential development risks."""
        
        risks = []
        
        # Complexity risks
        if len(parsed_design.mechanics) > 10:
            risks.append({
                "type": "scope",
                "severity": "high",
                "description": "Too many gameplay mechanics may lead to implementation complexity"
            })
            
        if len(parsed_design.visual_assets) > 30:
            risks.append({
                "type": "resources",
                "severity": "medium",
                "description": "High number of visual assets may require extended generation time"
            })
            
        # Platform risks
        platforms = parsed_design.metadata.get("target_platforms", [])
        if len(platforms) > 3:
            risks.append({
                "type": "compatibility",
                "severity": "medium", 
                "description": "Multiple platform targets may require additional testing and optimization"
            })
            
        # Genre-specific risks
        genre = parsed_design.metadata.get("genre")
        if genre in ["rpg", "strategy"]:
            risks.append({
                "type": "complexity",
                "severity": "high",
                "description": f"{genre.upper()} games typically require complex systems and balancing"
            })
            
        return risks
    
    def _generate_recommendations(self, complexity: Dict[str, float], risks: List[Dict[str, str]]) -> List[str]:
        """Generate development recommendations."""
        
        recommendations = []
        
        if complexity["mechanics"] > 0.7:
            recommendations.append("Consider simplifying core mechanics or implementing them in phases")
            
        if complexity["assets"] > 0.8:
            recommendations.append("Plan for extended asset generation time or consider asset reuse")
            
        if any(risk["severity"] == "high" for risk in risks):
            recommendations.append("High-risk elements identified - consider MVP approach for first iteration")
            
        if complexity["platforms"] > 0.5:
            recommendations.append("Start with single platform and expand after core gameplay is validated")
            
        return recommendations