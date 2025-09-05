"""
Comprehensive tests for the 3-tier conversation memory system.
Tests performance, efficiency, and integration with LangGraph.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.project_maestro.core.conversation_memory import (
    ConversationMemoryManager,
    ShortTermMemory,
    SummaryMemory,
    EntityExtractor,
    VectorMemoryStore,
    Entity,
    EntityType,
    MemoryContext
)
from src.project_maestro.core.config import settings


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def sample_game_conversation():
    """Sample game development conversation for testing."""
    return [
        HumanMessage(content="Hi, I'm John. I want to create a platformer game like Mario."),
        AIMessage(content="Great! A platformer game is a classic choice. What platform are you targeting?"),
        HumanMessage(content="I'm thinking mobile, specifically Android and iOS. I love pixel art style."),
        AIMessage(content="Perfect! Mobile platformers with pixel art are very popular. What's your experience level?"),
        HumanMessage(content="I'm a beginner programmer, know some Python but new to game development."),
        AIMessage(content="That's a good starting point. Would you like to use a beginner-friendly engine like Unity?"),
        HumanMessage(content="Yes, Unity sounds good. I also really like characters with expressive animations."),
        AIMessage(content="Unity is excellent for mobile games. For expressive characters, we'll focus on smooth animations."),
    ]


class TestShortTermMemory:
    """Test short-term memory sliding window functionality."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_messages(self, sample_game_conversation):
        """Test basic message storage and retrieval."""
        memory = ShortTermMemory(window_size=5)
        user_id = "test_user"
        session_id = "test_session"
        
        # Store messages
        for message in sample_game_conversation[:6]:
            await memory.store_message(message, user_id, session_id)
        
        # Retrieve messages
        retrieved = await memory.retrieve(user_id, session_id)
        
        # Should only have last 5 messages due to sliding window
        assert len(retrieved) == 5
        assert retrieved[0].content == sample_game_conversation[1].content  # First message was dropped
    
    @pytest.mark.asyncio
    async def test_sliding_window_behavior(self):
        """Test that sliding window correctly maintains size."""
        memory = ShortTermMemory(window_size=3)
        user_id = "test_user"
        session_id = "test_session"
        
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(10)
        ]
        
        # Store all messages
        for message in messages:
            await memory.store_message(message, user_id, session_id)
        
        retrieved = await memory.retrieve(user_id, session_id)
        
        # Should only have last 3 messages
        assert len(retrieved) == 3
        assert retrieved[0].content == "Message 7"
        assert retrieved[1].content == "Message 8"
        assert retrieved[2].content == "Message 9"
    
    @pytest.mark.asyncio
    async def test_performance_large_volume(self):
        """Test performance with large message volumes."""
        memory = ShortTermMemory(window_size=100)
        user_id = "test_user"
        session_id = "test_session"
        
        # Generate large number of messages
        messages = [HumanMessage(content=f"Message {i}") for i in range(1000)]
        
        start_time = time.time()
        
        # Store messages
        for message in messages:
            await memory.store_message(message, user_id, session_id)
        
        # Retrieve messages
        retrieved = await memory.retrieve(user_id, session_id)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(retrieved) == 100  # Should maintain window size
        print(f"Short-term memory processed 1000 messages in {execution_time:.2f}s")


class TestSummaryMemory:
    """Test summary memory LLM-based summarization."""
    
    @pytest.mark.asyncio
    async def test_summarization_trigger(self, mock_llm, sample_game_conversation):
        """Test that summarization is triggered when token threshold is exceeded."""
        # Mock LLM response for summarization
        mock_llm.ainvoke.return_value.content = """
        User John wants to create a mobile platformer game similar to Mario.
        Key preferences: Android/iOS platform, pixel art style, beginner programmer with Python experience.
        Chose Unity engine for development. Likes expressive character animations.
        """
        
        # Set low threshold to trigger summarization
        memory = SummaryMemory(mock_llm, max_token_threshold=100)
        user_id = "test_user"
        session_id = "test_session"
        
        # Store messages to exceed threshold
        long_message = HumanMessage(content="This is a very long message " * 50)  # ~200 tokens
        await memory.store_message(long_message, user_id, session_id)
        
        # Give time for async summarization
        await asyncio.sleep(0.1)
        
        # Check if summarization was created
        summary = await memory.retrieve(user_id, session_id)
        assert summary is not None
        assert "John" in summary
        assert "platformer" in summary
        mock_llm.ainvoke.assert_called()
    
    @pytest.mark.asyncio
    async def test_summary_content_quality(self, mock_llm, sample_game_conversation):
        """Test quality of generated summaries."""
        mock_llm.ainvoke.return_value.content = """
        User Profile: John, beginner programmer with Python experience
        Game Vision: Mobile platformer inspired by Mario for Android/iOS
        Art Preference: Pixel art style with expressive character animations
        Engine Choice: Unity for mobile development
        Experience Level: New to game development, eager to learn
        """
        
        memory = SummaryMemory(mock_llm, max_token_threshold=50)
        user_id = "test_user"  
        session_id = "test_session"
        
        # Trigger summarization
        await memory.store_message(
            HumanMessage(content="Long message content " * 20), 
            user_id, 
            session_id
        )
        
        await asyncio.sleep(0.1)
        summary = await memory.retrieve(user_id, session_id)
        
        # Verify summary contains key information
        assert summary is not None
        assert "John" in summary
        assert "platformer" in summary or "Mario" in summary
        assert "pixel art" in summary.lower() or "Unity" in summary
    
    @pytest.mark.asyncio
    async def test_token_efficiency(self, mock_llm):
        """Test that summarization reduces token count effectively."""
        mock_llm.ainvoke.return_value.content = "Concise summary of long conversation."
        
        memory = SummaryMemory(mock_llm, max_token_threshold=200)
        user_id = "test_user"
        session_id = "test_session"
        
        # Store long messages
        total_original_length = 0
        for i in range(5):
            long_content = f"This is message {i} with lots of detailed content " * 20
            total_original_length += len(long_content)
            await memory.store_message(
                HumanMessage(content=long_content), 
                user_id, 
                session_id
            )
        
        await asyncio.sleep(0.1)
        summary = await memory.retrieve(user_id, session_id)
        
        # Summary should be much shorter than original content
        if summary:
            compression_ratio = len(summary) / total_original_length
            assert compression_ratio < 0.1  # At least 90% compression
            print(f"Achieved {(1-compression_ratio)*100:.1f}% compression")


class TestEntityExtractor:
    """Test entity extraction from game development conversations."""
    
    @pytest.mark.asyncio
    async def test_entity_extraction_game_context(self, mock_llm, sample_game_conversation):
        """Test extraction of game development entities."""
        # Mock entity extraction response
        mock_llm.ainvoke.return_value.content = json.dumps([
            {
                "type": "user_profile",
                "content": "John, beginner programmer with Python experience, new to game development",
                "confidence": 0.9,
                "metadata": {"experience_level": "beginner", "languages": ["python"]}
            },
            {
                "type": "game_preference", 
                "content": "Platformer games like Mario",
                "confidence": 0.95,
                "metadata": {"inspiration": "Mario", "genre": "platformer"}
            },
            {
                "type": "platform_preference",
                "content": "Mobile (Android and iOS)",
                "confidence": 0.9,
                "metadata": {"platforms": ["android", "ios"]}
            },
            {
                "type": "art_style",
                "content": "Pixel art with expressive character animations",
                "confidence": 0.85,
                "metadata": {"style": "pixel_art", "focus": "character_animations"}
            }
        ])
        
        extractor = EntityExtractor(mock_llm)
        entities = await extractor.extract_entities(sample_game_conversation, "test_user")
        
        # Verify extracted entities
        assert len(entities) == 4
        
        entity_types = [e.type for e in entities]
        assert EntityType.USER_PROFILE in entity_types
        assert EntityType.GAME_PREFERENCE in entity_types  
        assert EntityType.PLATFORM_PREFERENCE in entity_types
        assert EntityType.ART_STYLE in entity_types
        
        # Check entity content quality
        profile_entity = next(e for e in entities if e.type == EntityType.USER_PROFILE)
        assert "John" in profile_entity.content
        assert "beginner" in profile_entity.content.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, mock_llm):
        """Test that low-confidence entities are filtered out."""
        # Mock response with mixed confidence levels
        mock_llm.ainvoke.return_value.content = json.dumps([
            {
                "type": "user_profile",
                "content": "High confidence entity",
                "confidence": 0.95,
                "metadata": {}
            },
            {
                "type": "game_preference",
                "content": "Low confidence entity", 
                "confidence": 0.3,  # Below threshold
                "metadata": {}
            }
        ])
        
        extractor = EntityExtractor(mock_llm)
        entities = await extractor.extract_entities([HumanMessage(content="test")], "test_user")
        
        # Should only have high-confidence entity
        assert len(entities) == 1
        assert entities[0].confidence >= 0.7
        assert "High confidence" in entities[0].content
    
    @pytest.mark.asyncio
    async def test_entity_extraction_performance(self, mock_llm):
        """Test performance of entity extraction."""
        mock_llm.ainvoke.return_value.content = json.dumps([
            {
                "type": "user_profile",
                "content": "Test entity",
                "confidence": 0.8,
                "metadata": {}
            }
        ])
        
        extractor = EntityExtractor(mock_llm)
        
        # Test with varying conversation lengths
        for msg_count in [5, 20, 50]:
            messages = [HumanMessage(content=f"Message {i}") for i in range(msg_count)]
            
            start_time = time.time()
            entities = await extractor.extract_entities(messages, "test_user")
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert execution_time < 3.0  # Should complete within 3 seconds
            print(f"Extracted entities from {msg_count} messages in {execution_time:.2f}s")


class TestVectorMemoryStore:
    """Test vector-based long-term memory storage."""
    
    @pytest.mark.asyncio
    async def test_entity_storage_and_retrieval(self):
        """Test storing and retrieving entities from vector store."""
        store = VectorMemoryStore()
        
        # Create test entities
        entities = [
            Entity(
                id="entity_1",
                type=EntityType.USER_PROFILE,
                content="John, experienced Unity developer specializing in mobile games",
                metadata={"experience": "advanced", "specialty": "mobile"},
                confidence=0.9,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id="test_user"
            ),
            Entity(
                id="entity_2", 
                type=EntityType.GAME_PREFERENCE,
                content="Prefers RPG and strategy games with complex narratives",
                metadata={"genres": ["rpg", "strategy"]},
                confidence=0.85,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id="test_user"
            )
        ]
        
        # Store entities
        with patch.object(store, 'rag_system') as mock_rag:
            mock_rag.index_document = AsyncMock()
            await store.store_entities(entities)
            
            # Verify indexing was called
            assert mock_rag.index_document.call_count == 2
    
    @pytest.mark.asyncio
    async def test_relevance_based_retrieval(self):
        """Test retrieving relevant entities based on query similarity."""
        store = VectorMemoryStore()
        
        # Mock relevant search results
        mock_results = [
            {
                "content": "user_profile: John, experienced Unity developer",
                "metadata": {
                    "entity_id": "entity_1",
                    "entity_type": "user_profile", 
                    "user_id": "test_user",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
        
        with patch.object(store, 'rag_system') as mock_rag:
            mock_rag.search_similar = AsyncMock(return_value=mock_results)
            
            entities = await store.retrieve_relevant_entities(
                "Unity development experience", 
                "test_user", 
                limit=5
            )
            
            assert len(entities) == 1
            assert entities[0].type == EntityType.USER_PROFILE
            assert "John" in entities[0].content
            
            # Verify search was called with correct parameters
            mock_rag.search_similar.assert_called_with(
                query="Unity development experience",
                namespace="memory:test_user",
                top_k=5,
                similarity_threshold=0.7
            )


class TestConversationMemoryManager:
    """Test the integrated conversation memory manager."""
    
    @pytest.mark.asyncio
    async def test_integrated_memory_flow(self, mock_llm, sample_game_conversation):
        """Test complete memory flow across all layers."""
        # Mock responses
        mock_llm.ainvoke.return_value.content = json.dumps([
            {
                "type": "user_profile",
                "content": "John, beginner game developer",
                "confidence": 0.9,
                "metadata": {}
            }
        ])
        
        manager = ConversationMemoryManager(mock_llm)
        user_id = "test_user"
        session_id = "test_session"
        
        # Store conversation
        for message in sample_game_conversation[:4]:
            await manager.store_message(message, user_id, session_id)
        
        # Get memory context
        context = await manager.get_memory_context(
            user_id, 
            session_id, 
            "I want to add character animations"
        )
        
        # Verify memory context
        assert context.short_term_messages is not None
        assert len(context.short_term_messages) <= 4
        assert context.token_count > 0
        
        # Test prompt context generation
        prompt_context = context.to_prompt_context()
        assert isinstance(prompt_context, str)
        assert len(prompt_context) > 0
    
    @pytest.mark.asyncio 
    async def test_user_profile_summary(self, mock_llm):
        """Test user profile summary generation."""
        manager = ConversationMemoryManager(mock_llm)
        
        # Mock vector store with user entities
        mock_entities = [
            Entity(
                id="1",
                type=EntityType.USER_PROFILE,
                content="John, experienced developer",
                metadata={},
                confidence=0.9,
                created_at=datetime.now(),
                updated_at=datetime.now(), 
                user_id="test_user"
            ),
            Entity(
                id="2",
                type=EntityType.GAME_PREFERENCE,
                content="Loves platformer games",
                metadata={},
                confidence=0.8,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id="test_user"
            )
        ]
        
        with patch.object(manager.vector_store, 'retrieve_relevant_entities') as mock_retrieve:
            mock_retrieve.return_value = mock_entities
            
            profile = await manager.get_user_profile_summary("test_user")
            
            assert "John" in profile
            assert "User Profile:" in profile
            assert "user_profile" in profile.lower() or "game_preference" in profile.lower()
    
    @pytest.mark.asyncio
    async def test_memory_system_performance(self, mock_llm):
        """Test overall memory system performance."""
        mock_llm.ainvoke.return_value.content = "[]"  # Empty entity extraction
        
        manager = ConversationMemoryManager(mock_llm)
        user_id = "test_user"
        session_id = "test_session"
        
        # Test with varying loads
        message_counts = [10, 50, 100]
        
        for count in message_counts:
            messages = [
                HumanMessage(content=f"Message {i} about game development")
                for i in range(count)
            ]
            
            start_time = time.time()
            
            # Store all messages
            for message in messages:
                await manager.store_message(message, user_id, session_id)
            
            # Get memory context
            context = await manager.get_memory_context(user_id, session_id, "query")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertions
            assert execution_time < 10.0  # Should complete within 10 seconds
            assert context is not None
            
            print(f"Processed {count} messages in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_token_efficiency_across_layers(self, mock_llm, sample_game_conversation):
        """Test token efficiency across all memory layers."""
        # Mock summary response
        mock_llm.ainvoke.return_value.content = "Brief summary of conversation"
        
        manager = ConversationMemoryManager(mock_llm)
        user_id = "test_user"
        session_id = "test_session"
        
        # Calculate original token count
        original_content = " ".join([msg.content for msg in sample_game_conversation])
        original_tokens = len(original_content) // 4  # Rough estimation
        
        # Store conversation
        for message in sample_game_conversation:
            await manager.store_message(message, user_id, session_id)
        
        # Allow time for summarization
        await asyncio.sleep(0.1)
        
        # Get memory context
        context = await manager.get_memory_context(user_id, session_id, "test query")
        
        # Compare token efficiency
        if context.token_count > 0:
            efficiency_ratio = context.token_count / original_tokens
            print(f"Memory system token efficiency: {context.token_count}/{original_tokens} = {efficiency_ratio:.2f}")
            
            # Should be more efficient for longer conversations
            if len(sample_game_conversation) > 5:
                assert efficiency_ratio < 0.8  # At least 20% reduction


@pytest.mark.integration
class TestMemorySystemIntegration:
    """Integration tests for memory system with LangGraph."""
    
    @pytest.mark.asyncio
    async def test_langgraph_memory_integration(self, mock_llm):
        """Test memory system integration with LangGraph orchestrator."""
        from src.project_maestro.core.langgraph_orchestrator import LangGraphOrchestrator
        
        # Mock agents
        mock_agents = {}
        
        orchestrator = LangGraphOrchestrator(
            agents=mock_agents,
            llm=mock_llm,
            enable_conversation_memory=True
        )
        
        # Verify conversation memory is initialized
        assert orchestrator.conversation_memory is not None
        assert isinstance(orchestrator.conversation_memory, ConversationMemoryManager)
    
    @pytest.mark.asyncio
    async def test_memory_context_in_state(self, mock_llm):
        """Test memory context enhancement in MaestroState."""
        from src.project_maestro.core.langgraph_orchestrator import LangGraphOrchestrator, MaestroState
        
        orchestrator = LangGraphOrchestrator({}, mock_llm, enable_conversation_memory=True)
        
        # Mock state
        state = MaestroState(
            messages=[HumanMessage(content="Test message")],
            current_agent="supervisor",
            task_context={},
            game_design_doc=None,
            assets_generated={},
            code_artifacts={},
            build_status={},
            workflow_stage="initial",
            handoff_history=[],
            execution_metadata={},
            memory_context=None,
            user_id="test_user",
            session_id="test_session"
        )
        
        # Mock memory context
        mock_context = MemoryContext(
            short_term_messages=[],
            summary="Test summary",
            relevant_entities=[],
            token_count=100
        )
        
        with patch.object(orchestrator.conversation_memory, 'get_memory_context') as mock_get_context:
            mock_get_context.return_value = mock_context
            
            enhanced_state = await orchestrator._enhance_state_with_memory(state)
            
            # Verify memory context was added
            assert enhanced_state["memory_context"] is not None
            assert enhanced_state["memory_context"]["summary"] == "Test summary"
            assert enhanced_state["memory_context"]["token_count"] == 100


if __name__ == "__main__":
    # Run performance benchmarks
    asyncio.run(pytest.main([__file__, "-v", "--tb=short"]))