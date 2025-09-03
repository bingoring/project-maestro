"""Unit tests for the game design document parser."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.project_maestro.core.gdd_parser import (
    GameDesignDocumentParser, GameComplexityAnalyzer, ParsedGameDesign,
    GameGenre, Platform, GameMetadata, Character, GameplayMechanic, 
    Level, AudioRequirement, VisualAsset
)
from conftest import MockLLM, sample_game_design_document


class TestGameDesignDocumentParser:
    """Test GDD parser functionality."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        mock_llm = MockLLM()
        parser = GameDesignDocumentParser(mock_llm)
        
        assert parser.llm == mock_llm
        assert parser.output_parser is not None
        assert parser.parsing_prompt is not None
        
    def test_preprocess_document(self):
        """Test document preprocessing."""
        mock_llm = MockLLM()
        parser = GameDesignDocumentParser(mock_llm)
        
        messy_text = """
        
        # Game Title    
        
        
        Some description with    extra   spaces.
        
        
        #### Too many hashes
        
        Special chars: @#$%^&*()
        """
        
        cleaned = parser._preprocess_document(messy_text)
        
        # Should remove excessive whitespace
        assert "\n\n\n" not in cleaned
        # Should normalize headers
        assert "####" not in cleaned
        # Should be stripped
        assert not cleaned.startswith("\n")
        assert not cleaned.endswith("\n")
        
    def test_parse_document_success(self, sample_game_design_document):
        """Test successful document parsing."""
        # Mock LLM response with valid JSON
        mock_response = """{
            "metadata": {
                "title": "Super Jump Adventure",
                "genre": "platformer", 
                "target_platforms": ["Android", "iOS"],
                "description": "A simple platformer game",
                "complexity_score": 0.5
            },
            "characters": [
                {
                    "name": "Hero",
                    "role": "player",
                    "description": "A small, colorful character"
                }
            ],
            "mechanics": [
                {
                    "name": "jumping",
                    "description": "Player can jump to reach platforms",
                    "implementation_complexity": "simple"
                }
            ],
            "levels": [
                {
                    "name": "Level 1",
                    "theme": "Green Hills",
                    "objectives": ["Collect coins"],
                    "difficulty": "easy"
                }
            ],
            "audio_requirements": [
                {
                    "name": "jump_sound",
                    "type": "sfx",
                    "mood": "action"
                }
            ],
            "visual_assets": [
                {
                    "name": "hero_sprite",
                    "type": "sprite", 
                    "style": "pixel"
                }
            ],
            "technical_requirements": {
                "unity_version": "2023.2.0f1"
            }
        }"""
        
        mock_llm = MockLLM([mock_response])
        parser = GameDesignDocumentParser(mock_llm)
        
        result = parser.parse_document(sample_game_design_document)
        
        assert isinstance(result, ParsedGameDesign)
        assert result.metadata["title"] == "Super Jump Adventure"
        assert len(result.characters) == 1
        assert len(result.mechanics) == 1
        assert len(result.levels) == 1
        assert len(result.audio_requirements) == 1
        assert len(result.visual_assets) == 1
        
    def test_validate_and_enhance(self):
        """Test document validation and enhancement."""
        mock_llm = MockLLM()
        parser = GameDesignDocumentParser(mock_llm)
        
        # Create minimal parsed design
        parsed_design = ParsedGameDesign(
            metadata={},  # Empty metadata
            characters=[],
            mechanics=[],
            levels=[],
            audio_requirements=[],
            visual_assets=[]
        )
        
        enhanced = parser._validate_and_enhance(parsed_design)
        
        # Should add default title
        assert enhanced.metadata["title"] == "Untitled Game"
        # Should add default genre
        assert enhanced.metadata["genre"] == "casual"
        # Should have complexity score
        assert "complexity_score" in enhanced.metadata
        # Should add default audio requirements
        assert len(enhanced.audio_requirements) > 0
        # Should add minimum visual assets
        assert len(enhanced.visual_assets) > 0
        
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        mock_llm = MockLLM()
        parser = GameDesignDocumentParser(mock_llm)
        
        # Simple game
        simple_design = ParsedGameDesign(
            metadata={},
            characters=[{"name": "Hero", "role": "player"}],
            mechanics=[{"name": "move", "implementation_complexity": "simple"}],
            levels=[{"name": "Level 1"}],
            audio_requirements=[{"name": "jump_sound"}],
            visual_assets=[{"name": "hero_sprite"}]
        )
        
        simple_score = parser._calculate_complexity_score(simple_design)
        assert 0.0 <= simple_score <= 1.0
        
        # Complex game
        complex_design = ParsedGameDesign(
            metadata={"target_platforms": ["Android", "iOS", "WebGL"]},
            characters=[{"name": f"Char{i}", "role": "player"} for i in range(5)],
            mechanics=[
                {"name": f"mechanic{i}", "implementation_complexity": "complex"} 
                for i in range(3)
            ],
            levels=[{"name": f"Level{i}"} for i in range(10)],
            audio_requirements=[{"name": f"sound{i}"} for i in range(20)],
            visual_assets=[{"name": f"asset{i}"} for i in range(50)]
        )
        
        complex_score = parser._calculate_complexity_score(complex_design)
        assert complex_score > simple_score
        assert complex_score <= 1.0
        
    def test_extract_asset_requirements(self):
        """Test asset requirements extraction."""
        mock_llm = MockLLM()
        parser = GameDesignDocumentParser(mock_llm)
        
        parsed_design = ParsedGameDesign(
            metadata={},
            characters=[
                {
                    "name": "Hero",
                    "role": "player",
                    "abilities": ["jump", "shoot"]
                }
            ],
            mechanics=[
                {
                    "name": "shooting",
                    "code_requirements": ["projectile_system", "ammo_management"]
                }
            ],
            levels=[{"name": "Level 1", "theme": "forest"}],
            audio_requirements=[{"name": "shoot_sound", "type": "sfx"}],
            visual_assets=[{"name": "hero_sprite", "type": "sprite"}]
        )
        
        requirements = parser.extract_asset_requirements(parsed_design)
        
        assert "code" in requirements
        assert "visual" in requirements
        assert "audio" in requirements
        assert "level" in requirements
        
        # Should have code requirements from mechanics
        assert len(requirements["code"]) > 0
        # Should have player controller requirement
        player_controllers = [
            req for req in requirements["code"] 
            if req.get("type") == "player_controller"
        ]
        assert len(player_controllers) > 0
        
        # Should have visual and audio requirements
        assert requirements["visual"] == parsed_design.visual_assets
        assert requirements["audio"] == parsed_design.audio_requirements
        assert requirements["level"] == parsed_design.levels


class TestGameComplexityAnalyzer:
    """Test game complexity analyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = GameComplexityAnalyzer()
        
        assert analyzer.complexity_weights is not None
        assert sum(analyzer.complexity_weights.values()) == 1.0  # Weights should sum to 1
        
    def test_analyze_development_effort(self):
        """Test development effort analysis."""
        analyzer = GameComplexityAnalyzer()
        
        parsed_design = ParsedGameDesign(
            metadata={"title": "Test Game"},
            characters=[{"name": "Hero"}],
            mechanics=[
                {"name": "jump", "implementation_complexity": "simple"},
                {"name": "combat", "implementation_complexity": "complex"}
            ],
            levels=[{"name": "Level 1"}, {"name": "Level 2"}],
            audio_requirements=[{"name": "bgm"}],
            visual_assets=[{"name": "hero_sprite"}]
        )
        
        effort_analysis = analyzer.analyze_development_effort(parsed_design)
        
        assert "complexity_breakdown" in effort_analysis
        assert "time_estimates" in effort_analysis
        assert "agent_workload" in effort_analysis
        assert "development_risks" in effort_analysis
        assert "recommendations" in effort_analysis
        
        # Check complexity breakdown structure
        complexity = effort_analysis["complexity_breakdown"]
        assert "mechanics" in complexity
        assert "characters" in complexity
        assert "levels" in complexity
        assert "assets" in complexity
        assert "platforms" in complexity
        
        # All complexity scores should be between 0 and 1
        for score in complexity.values():
            assert 0.0 <= score <= 1.0
            
    def test_complexity_breakdown(self):
        """Test detailed complexity breakdown analysis."""
        analyzer = GameComplexityAnalyzer()
        
        # Design with varying complexity
        parsed_design = ParsedGameDesign(
            metadata={"target_platforms": ["Android", "iOS"]},
            characters=[{"name": f"Char{i}"} for i in range(3)],
            mechanics=[
                {"name": "simple_move", "implementation_complexity": "simple"},
                {"name": "complex_ai", "implementation_complexity": "complex"},
                {"name": "moderate_combat", "implementation_complexity": "moderate"}
            ],
            levels=[{"name": f"Level{i}"} for i in range(5)],
            audio_requirements=[{"name": f"sound{i}"} for i in range(8)],
            visual_assets=[{"name": f"asset{i}"} for i in range(15)]
        )
        
        complexity = analyzer._analyze_complexity_breakdown(parsed_design)
        
        # Mechanics complexity should account for different complexity levels
        assert complexity["mechanics"] > 0
        # More characters should increase complexity
        assert complexity["characters"] > 0
        # More levels should increase complexity
        assert complexity["levels"] > 0
        # More assets should increase complexity
        assert complexity["assets"] > 0
        # Multiple platforms should increase complexity
        assert complexity["platforms"] > 0
        
    def test_time_estimation(self):
        """Test development time estimation."""
        analyzer = GameComplexityAnalyzer()
        
        # Simple complexity
        simple_complexity = {
            "mechanics": 0.2,
            "characters": 0.1,
            "levels": 0.1,
            "assets": 0.1,
            "platforms": 0.1
        }
        
        simple_design = ParsedGameDesign(
            metadata={},
            characters=[{"name": "Hero"}],
            mechanics=[{"name": "jump"}],
            levels=[{"name": "Level 1"}],
            audio_requirements=[],
            visual_assets=[]
        )
        
        simple_times = analyzer._estimate_development_time(simple_complexity, simple_design)
        
        # Complex complexity
        complex_complexity = {
            "mechanics": 0.8,
            "characters": 0.7,
            "levels": 0.6,
            "assets": 0.8,
            "platforms": 0.5
        }
        
        complex_design = ParsedGameDesign(
            metadata={},
            characters=[{"name": f"Char{i}"} for i in range(5)],
            mechanics=[{"name": f"mechanic{i}"} for i in range(10)],
            levels=[{"name": f"Level{i}"} for i in range(8)],
            audio_requirements=[{"name": f"sound{i}"} for i in range(20)],
            visual_assets=[{"name": f"asset{i}"} for i in range(30)]
        )
        
        complex_times = analyzer._estimate_development_time(complex_complexity, complex_design)
        
        # Complex games should take longer
        assert complex_times["code_generation"] > simple_times["code_generation"]
        assert complex_times["asset_creation"] > simple_times["asset_creation"]
        
        # All time estimates should be positive
        for phase_time in simple_times.values():
            assert phase_time > 0
            
    def test_agent_workload_estimation(self):
        """Test agent workload estimation."""
        analyzer = GameComplexityAnalyzer()
        
        parsed_design = ParsedGameDesign(
            metadata={"target_platforms": ["Android", "iOS"]},
            characters=[{"name": f"Char{i}"} for i in range(2)],
            mechanics=[{"name": f"mechanic{i}"} for i in range(3)],
            levels=[{"name": f"Level{i}"} for i in range(4)],
            audio_requirements=[{"name": f"sound{i}"} for i in range(5)],
            visual_assets=[{"name": f"asset{i}"} for i in range(10)]
        )
        
        workload = analyzer._estimate_agent_workload(parsed_design)
        
        # Should have workload for all agent types
        expected_agents = ["codex", "canvas", "sonata", "labyrinth", "builder"]
        for agent in expected_agents:
            assert agent in workload
            assert "tasks" in workload[agent]
            assert "estimated_time" in workload[agent]
            assert "complexity" in workload[agent]
            
        # Task counts should match design complexity
        assert workload["codex"]["tasks"] == 5  # 3 mechanics + 2 characters
        assert workload["canvas"]["tasks"] == 10  # 10 visual assets
        assert workload["sonata"]["tasks"] == 5  # 5 audio requirements
        assert workload["labyrinth"]["tasks"] == 4  # 4 levels
        assert workload["builder"]["tasks"] == 2  # 2 platforms
        
    def test_risk_identification(self):
        """Test development risk identification."""
        analyzer = GameComplexityAnalyzer()
        
        # High-risk design
        risky_design = ParsedGameDesign(
            metadata={"genre": "rpg", "target_platforms": ["Android", "iOS", "WebGL", "Windows"]},
            characters=[{"name": f"Char{i}"} for i in range(2)],
            mechanics=[{"name": f"mechanic{i}"} for i in range(15)],  # Too many mechanics
            levels=[{"name": f"Level{i}"} for i in range(3)],
            audio_requirements=[{"name": f"sound{i}"} for i in range(5)],
            visual_assets=[{"name": f"asset{i}"} for i in range(35)]  # Too many assets
        )
        
        risks = analyzer._identify_development_risks(risky_design)
        
        # Should identify multiple risks
        assert len(risks) > 0
        
        risk_types = [risk["type"] for risk in risks]
        risk_severities = [risk["severity"] for risk in risks]
        
        # Should identify scope and resource risks
        assert "scope" in risk_types  # Too many mechanics
        assert "resources" in risk_types  # Too many assets
        assert "compatibility" in risk_types  # Too many platforms
        assert "complexity" in risk_types  # RPG genre
        
        # Should have high severity risks
        assert "high" in risk_severities
        
    def test_recommendation_generation(self):
        """Test development recommendation generation."""
        analyzer = GameComplexityAnalyzer()
        
        # High complexity and high risk scenario
        high_complexity = {
            "mechanics": 0.8,
            "characters": 0.5,
            "levels": 0.3,
            "assets": 0.9,
            "platforms": 0.6
        }
        
        high_risks = [
            {"type": "scope", "severity": "high", "description": "Too many features"},
            {"type": "resources", "severity": "medium", "description": "Many assets needed"}
        ]
        
        recommendations = analyzer._generate_recommendations(high_complexity, high_risks)
        
        assert len(recommendations) > 0
        assert isinstance(recommendations, list)
        
        # Should recommend simplification for high complexity
        complexity_recommendations = [
            rec for rec in recommendations 
            if "simplif" in rec.lower() or "mechanic" in rec.lower()
        ]
        assert len(complexity_recommendations) > 0
        
        # Should recommend MVP approach for high risks
        mvp_recommendations = [
            rec for rec in recommendations 
            if "mvp" in rec.lower() or "phase" in rec.lower()
        ]
        assert len(mvp_recommendations) > 0


@pytest.mark.integration
class TestGDDParserIntegration:
    """Integration tests for GDD parser."""
    
    def test_full_parsing_workflow(self, sample_game_design_document):
        """Test complete parsing workflow."""
        # Mock LLM with realistic response
        realistic_response = """{
            "metadata": {
                "title": "Super Jump Adventure",
                "genre": "platformer",
                "target_platforms": ["Android", "iOS"],
                "description": "A simple platformer game with jumping and collecting",
                "target_audience": "Casual gamers",
                "estimated_playtime": 30,
                "complexity_score": 0.4
            },
            "characters": [
                {
                    "name": "Hero",
                    "role": "player",
                    "description": "A small, colorful character",
                    "abilities": ["move", "jump", "collect"],
                    "visual_style": "pixel art",
                    "animations_needed": ["idle", "walk", "jump", "fall"]
                }
            ],
            "mechanics": [
                {
                    "name": "movement",
                    "description": "Left/right movement",
                    "implementation_complexity": "simple",
                    "required_assets": ["movement_animations"],
                    "code_requirements": ["input_system", "physics"]
                },
                {
                    "name": "jumping",
                    "description": "Jump to reach platforms",
                    "implementation_complexity": "simple", 
                    "required_assets": ["jump_animation"],
                    "code_requirements": ["jump_physics"]
                }
            ],
            "levels": [
                {
                    "name": "Green Hills",
                    "theme": "grassland",
                    "objectives": ["Collect 10 coins", "Reach the flag"],
                    "enemies": ["Goomba"],
                    "mechanics": ["movement", "jumping"],
                    "estimated_playtime": 5,
                    "difficulty": "easy"
                }
            ],
            "audio_requirements": [
                {
                    "name": "background_music",
                    "type": "bgm",
                    "mood": "upbeat",
                    "loop": true,
                    "triggers": ["level_start"]
                },
                {
                    "name": "jump_sound",
                    "type": "sfx",
                    "mood": "action",
                    "loop": false,
                    "triggers": ["player_jump"]
                }
            ],
            "visual_assets": [
                {
                    "name": "hero_sprite",
                    "type": "sprite",
                    "style": "pixel art",
                    "resolution": "32x32",
                    "description": "Main character sprite",
                    "variations": 4
                }
            ],
            "technical_requirements": {
                "unity_version": "2023.2.0f1",
                "target_fps": 60,
                "platforms": ["Android", "iOS"],
                "controls": ["touch"]
            }
        }"""
        
        mock_llm = MockLLM([realistic_response])
        parser = GameDesignDocumentParser(mock_llm)
        analyzer = GameComplexityAnalyzer()
        
        # Parse document
        parsed_design = parser.parse_document(sample_game_design_document)
        
        # Analyze complexity
        effort_analysis = analyzer.analyze_development_effort(parsed_design)
        
        # Extract asset requirements
        asset_requirements = parser.extract_asset_requirements(parsed_design)
        
        # Verify complete workflow
        assert parsed_design.metadata["title"] == "Super Jump Adventure"
        assert len(parsed_design.characters) == 1
        assert len(parsed_design.mechanics) == 2
        
        assert "complexity_breakdown" in effort_analysis
        assert "time_estimates" in effort_analysis
        
        assert "code" in asset_requirements
        assert "visual" in asset_requirements
        assert "audio" in asset_requirements
        assert "level" in asset_requirements
        
        # Verify asset requirements are properly structured
        assert len(asset_requirements["code"]) > 0
        assert asset_requirements["visual"] == parsed_design.visual_assets
        assert asset_requirements["audio"] == parsed_design.audio_requirements