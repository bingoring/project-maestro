"""
Advanced Conversation Memory System for Project Maestro

Implements a sophisticated 3-tier memory architecture:
1. Short-term Memory: Recent conversation buffer
2. Summary Memory: LLM-based conversation summarization  
3. Long-term Entity Memory: RAG-based persistent user knowledge storage
"""

import asyncio
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import redis.asyncio as redis

from .config import settings, get_enterprise_config
from .logging import get_logger
from .rag_system import EnhancedRAGSystem


logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the conversation system."""
    SHORT_TERM = "short_term"
    SUMMARY = "summary"  
    ENTITY = "entity"


class EntityType(str, Enum):
    """Types of entities extracted from conversations."""
    USER_PROFILE = "user_profile"
    GAME_PREFERENCE = "game_preference"
    PROJECT_HISTORY = "project_history"
    CHARACTER_PREFERENCE = "character_preference"
    ART_STYLE = "art_style"
    PLATFORM_PREFERENCE = "platform_preference"
    DEVELOPMENT_SKILL = "development_skill"
    FEEDBACK_PATTERN = "feedback_pattern"


@dataclass
class Entity:
    """Represents an extracted entity from conversation."""
    id: str
    type: EntityType
    content: str
    metadata: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    user_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_id": self.user_id
        }


@dataclass
class MemoryContext:
    """Complete memory context for a conversation."""
    short_term_messages: List[BaseMessage]
    summary: Optional[str]
    relevant_entities: List[Entity]
    token_count: int
    
    def to_prompt_context(self) -> str:
        """Convert memory context to prompt text."""
        context_parts = []
        
        # Add summary if available
        if self.summary:
            context_parts.append(f"Previous conversation summary: {self.summary}")
        
        # Add relevant entities
        if self.relevant_entities:
            entity_info = []
            for entity in self.relevant_entities:
                entity_info.append(f"- {entity.type.value}: {entity.content}")
            context_parts.append("User information:\n" + "\n".join(entity_info))
        
        # Add recent messages
        if self.short_term_messages:
            recent_msgs = []
            for msg in self.short_term_messages[-5:]:  # Last 5 messages
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                recent_msgs.append(f"{role}: {msg.content}")
            context_parts.append("Recent conversation:\n" + "\n".join(recent_msgs))
        
        return "\n\n".join(context_parts)


class BaseMemoryLayer(ABC):
    """Abstract base class for memory layers."""
    
    @abstractmethod
    async def store_message(self, message: BaseMessage, user_id: str, session_id: str) -> None:
        """Store a message in this memory layer."""
        pass
    
    @abstractmethod
    async def retrieve(self, user_id: str, session_id: str, **kwargs) -> Any:
        """Retrieve data from this memory layer."""
        pass
    
    @abstractmethod
    async def clear(self, user_id: str, session_id: str) -> None:
        """Clear memory for a session."""
        pass


class ShortTermMemory(BaseMemoryLayer):
    """Short-term memory using a sliding window buffer."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.redis_client: Optional[redis.Redis] = None
        self.logger = get_logger("short_term_memory")
        
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=False  # We need binary for pickle
            )
            await self.redis_client.ping()
            
    async def store_message(self, message: BaseMessage, user_id: str, session_id: str) -> None:
        """Store message in short-term buffer."""
        await self.connect()
        
        key = f"short_term:{user_id}:{session_id}"
        
        # Get current messages
        messages_data = await self.redis_client.get(key)
        if messages_data:
            messages = pickle.loads(messages_data)
        else:
            messages = []
        
        # Add new message
        messages.append({
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages (sliding window)
        messages = messages[-self.window_size:]
        
        # Store back to Redis with TTL
        await self.redis_client.setex(
            key, 
            3600 * 24,  # 24 hours TTL
            pickle.dumps(messages)
        )
        
        self.logger.debug(f"Stored message in short-term memory: {user_id}:{session_id}")
    
    async def retrieve(self, user_id: str, session_id: str, **kwargs) -> List[BaseMessage]:
        """Retrieve recent messages."""
        await self.connect()
        
        key = f"short_term:{user_id}:{session_id}"
        messages_data = await self.redis_client.get(key)
        
        if not messages_data:
            return []
        
        messages_info = pickle.loads(messages_data)
        
        # Convert back to BaseMessage objects
        messages = []
        for msg_info in messages_info:
            if msg_info["type"] == "HumanMessage":
                messages.append(HumanMessage(content=msg_info["content"]))
            elif msg_info["type"] == "AIMessage":
                messages.append(AIMessage(content=msg_info["content"]))
            elif msg_info["type"] == "SystemMessage":
                messages.append(SystemMessage(content=msg_info["content"]))
        
        return messages
    
    async def clear(self, user_id: str, session_id: str) -> None:
        """Clear short-term memory for session."""
        await self.connect()
        key = f"short_term:{user_id}:{session_id}"
        await self.redis_client.delete(key)


class SummaryMemory(BaseMemoryLayer):
    """Summary memory using LLM-based conversation summarization."""
    
    def __init__(self, llm: BaseLanguageModel, max_token_threshold: int = 2000):
        self.llm = llm
        self.max_token_threshold = max_token_threshold
        self.redis_client: Optional[redis.Redis] = None
        self.logger = get_logger("summary_memory")
        
        # Summarization prompt template
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that creates concise, useful summaries of conversations about game development.

Your task is to summarize the conversation history, focusing on:
1. User's game development goals and progress
2. Key decisions made about game design, mechanics, art style, etc.
3. User preferences and feedback patterns
4. Important technical discussions or solutions
5. Project status and next steps

Create a summary that captures the essential context needed for future conversations.
Be concise but preserve important details that would help continue the conversation naturally.

Previous summary (if any): {previous_summary}

Recent conversation to summarize:
{conversation_text}

Create a comprehensive summary:"""),
            ("human", "Please summarize this conversation focusing on game development context.")
        ])
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True
            )
            await self.redis_client.ping()
    
    async def store_message(self, message: BaseMessage, user_id: str, session_id: str) -> None:
        """Store message and trigger summarization if needed."""
        await self.connect()
        
        # Check if summarization is needed
        key = f"summary:{user_id}:{session_id}"
        summary_data = await self.redis_client.get(key)
        
        if summary_data:
            summary_info = json.loads(summary_data)
            token_count = summary_info.get("token_count", 0)
        else:
            token_count = 0
            
        # Estimate tokens (rough approximation: 4 chars = 1 token)
        message_tokens = len(message.content) // 4
        token_count += message_tokens
        
        # If we exceed threshold, trigger summarization
        if token_count > self.max_token_threshold:
            await self._create_summary(user_id, session_id)
        else:
            # Update token count
            summary_info = summary_data and json.loads(summary_data) or {}
            summary_info["token_count"] = token_count
            await self.redis_client.setex(
                key,
                3600 * 24 * 7,  # 7 days TTL
                json.dumps(summary_info)
            )
    
    async def _create_summary(self, user_id: str, session_id: str) -> None:
        """Create a summary of the conversation."""
        try:
            # Get recent messages from short-term memory
            short_term = ShortTermMemory()
            recent_messages = await short_term.retrieve(user_id, session_id)
            
            if not recent_messages:
                return
            
            # Get previous summary
            key = f"summary:{user_id}:{session_id}"
            summary_data = await self.redis_client.get(key)
            previous_summary = ""
            if summary_data:
                summary_info = json.loads(summary_data)
                previous_summary = summary_info.get("summary", "")
            
            # Format conversation for summarization
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in recent_messages
            ])
            
            # Generate summary using LLM
            messages = self.summary_prompt.format_messages(
                previous_summary=previous_summary,
                conversation_text=conversation_text
            )
            
            response = await self.llm.ainvoke(messages)
            new_summary = response.content
            
            # Store new summary
            summary_info = {
                "summary": new_summary,
                "token_count": 0,  # Reset token count
                "created_at": datetime.now().isoformat(),
                "message_count": len(recent_messages)
            }
            
            await self.redis_client.setex(
                key,
                3600 * 24 * 7,  # 7 days TTL
                json.dumps(summary_info)
            )
            
            self.logger.info(f"Created summary for {user_id}:{session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create summary: {e}")
    
    async def retrieve(self, user_id: str, session_id: str, **kwargs) -> Optional[str]:
        """Retrieve summary for session."""
        await self.connect()
        
        key = f"summary:{user_id}:{session_id}"
        summary_data = await self.redis_client.get(key)
        
        if summary_data:
            summary_info = json.loads(summary_data)
            return summary_info.get("summary")
        
        return None
    
    async def clear(self, user_id: str, session_id: str) -> None:
        """Clear summary for session."""
        await self.connect()
        key = f"summary:{user_id}:{session_id}"
        await self.redis_client.delete(key)


class EntityExtractor:
    """Extracts entities from conversations using LLM."""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.logger = get_logger("entity_extractor")
        
        # Entity extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured information from game development conversations.

Extract the following types of entities from the conversation:

1. USER_PROFILE: Name, experience level, role (indie dev, student, professional, etc.)
2. GAME_PREFERENCE: Preferred genres (RPG, platformer, puzzle, FPS, etc.)  
3. PROJECT_HISTORY: Previous projects, successes, failures, lessons learned
4. CHARACTER_PREFERENCE: Favorite character types, personality traits, visual styles
5. ART_STYLE: Preferred art styles (pixel art, 3D, cartoon, realistic, minimalist, etc.)
6. PLATFORM_PREFERENCE: Target platforms (mobile, PC, console, web, etc.)
7. DEVELOPMENT_SKILL: Programming languages, engines, tools, artistic skills
8. FEEDBACK_PATTERN: How user responds to suggestions, what they like/dislike

For each entity found, provide:
- type: one of the entity types above
- content: the extracted information (be specific and detailed)
- confidence: confidence score 0.0-1.0
- metadata: additional context or details

Return as JSON array. Only include entities with confidence > 0.7.
If no strong entities found, return empty array.

Conversation to analyze:
{conversation_text}

Extract entities:"""),
            ("human", "Extract relevant entities from this game development conversation.")
        ])
    
    async def extract_entities(
        self, 
        messages: List[BaseMessage], 
        user_id: str
    ) -> List[Entity]:
        """Extract entities from conversation messages."""
        try:
            # Format messages for analysis
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages
            ])
            
            # Generate extraction prompt
            prompt_messages = self.extraction_prompt.format_messages(
                conversation_text=conversation_text
            )
            
            # Extract entities using LLM
            response = await self.llm.ainvoke(prompt_messages)
            
            # Parse JSON response
            try:
                entities_data = json.loads(response.content)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse entity extraction JSON")
                return []
            
            # Convert to Entity objects
            entities = []
            for entity_data in entities_data:
                try:
                    entity = Entity(
                        id=str(uuid.uuid4()),
                        type=EntityType(entity_data["type"].lower()),
                        content=entity_data["content"],
                        metadata=entity_data.get("metadata", {}),
                        confidence=float(entity_data["confidence"]),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        user_id=user_id
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Invalid entity data: {e}")
                    continue
            
            self.logger.info(f"Extracted {len(entities)} entities for user {user_id}")
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []


class VectorMemoryStore:
    """Vector-based long-term memory storage using RAG."""
    
    def __init__(self):
        self.rag_system: Optional[EnhancedRAGSystem] = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.logger = get_logger("vector_memory")
        
    async def initialize(self) -> None:
        """Initialize RAG system for memory storage."""
        if not self.rag_system:
            # Use existing RAG system with memory-specific collection
            self.rag_system = EnhancedRAGSystem()
            await self.rag_system.initialize()
    
    async def store_entities(self, entities: List[Entity]) -> None:
        """Store entities in vector database."""
        await self.initialize()
        
        try:
            for entity in entities:
                # Create document for RAG system
                document_text = f"{entity.type.value}: {entity.content}"
                
                # Add metadata
                metadata = {
                    "entity_id": entity.id,
                    "entity_type": entity.type.value,
                    "user_id": entity.user_id,
                    "confidence": entity.confidence,
                    "created_at": entity.created_at.isoformat(),
                    "source": "conversation_memory",
                    **entity.metadata
                }
                
                # Index in RAG system with user namespace
                await self.rag_system.index_document(
                    content=document_text,
                    metadata=metadata,
                    namespace=f"memory:{entity.user_id}"
                )
            
            self.logger.info(f"Stored {len(entities)} entities in vector memory")
            
        except Exception as e:
            self.logger.error(f"Failed to store entities: {e}")
    
    async def retrieve_relevant_entities(
        self, 
        query: str, 
        user_id: str, 
        limit: int = 10
    ) -> List[Entity]:
        """Retrieve relevant entities based on query."""
        await self.initialize()
        
        try:
            # Search in user's memory namespace
            results = await self.rag_system.search_similar(
                query=query,
                namespace=f"memory:{user_id}",
                top_k=limit,
                similarity_threshold=0.7
            )
            
            # Convert search results back to Entity objects
            entities = []
            for result in results:
                metadata = result.get("metadata", {})
                
                try:
                    entity = Entity(
                        id=metadata["entity_id"],
                        type=EntityType(metadata["entity_type"]),
                        content=result["content"].split(": ", 1)[1],  # Remove type prefix
                        metadata={k: v for k, v in metadata.items() 
                                if k not in ["entity_id", "entity_type", "user_id", "confidence", "created_at", "source"]},
                        confidence=metadata["confidence"],
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                        updated_at=datetime.fromisoformat(metadata["created_at"]),
                        user_id=user_id
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Failed to reconstruct entity: {e}")
                    continue
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve entities: {e}")
            return []


class ConversationMemoryManager:
    """Main memory manager that coordinates all memory layers."""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.short_term = ShortTermMemory()
        self.summary = SummaryMemory(llm)
        self.entity_extractor = EntityExtractor(llm)
        self.vector_store = VectorMemoryStore()
        self.logger = get_logger("conversation_memory")
        
    async def store_message(self, message: BaseMessage, user_id: str, session_id: str) -> None:
        """Store message across all memory layers."""
        try:
            # Store in short-term memory
            await self.short_term.store_message(message, user_id, session_id)
            
            # Store in summary memory (handles summarization automatically)
            await self.summary.store_message(message, user_id, session_id)
            
            # Extract and store entities periodically
            if isinstance(message, HumanMessage):
                await self._extract_and_store_entities(user_id, session_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store message: {e}")
    
    async def _extract_and_store_entities(self, user_id: str, session_id: str) -> None:
        """Extract entities from recent conversation and store in vector memory."""
        try:
            # Get recent messages for entity extraction
            recent_messages = await self.short_term.retrieve(user_id, session_id)
            
            if len(recent_messages) >= 2:  # Need at least some conversation
                # Extract entities
                entities = await self.entity_extractor.extract_entities(
                    recent_messages, user_id
                )
                
                # Store in vector memory
                if entities:
                    await self.vector_store.store_entities(entities)
                    
        except Exception as e:
            self.logger.error(f"Failed to extract and store entities: {e}")
    
    async def get_memory_context(
        self, 
        user_id: str, 
        session_id: str, 
        current_query: str = ""
    ) -> MemoryContext:
        """Get complete memory context for conversation."""
        try:
            # Get short-term messages
            short_term_messages = await self.short_term.retrieve(user_id, session_id)
            
            # Get summary
            summary = await self.summary.retrieve(user_id, session_id)
            
            # Get relevant entities based on current query
            relevant_entities = []
            if current_query:
                relevant_entities = await self.vector_store.retrieve_relevant_entities(
                    current_query, user_id, limit=5
                )
            
            # Calculate total token count
            token_count = 0
            if summary:
                token_count += len(summary) // 4
            for msg in short_term_messages:
                token_count += len(msg.content) // 4
            for entity in relevant_entities:
                token_count += len(entity.content) // 4
            
            return MemoryContext(
                short_term_messages=short_term_messages,
                summary=summary,
                relevant_entities=relevant_entities,
                token_count=token_count
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory context: {e}")
            return MemoryContext([], None, [], 0)
    
    async def clear_session(self, user_id: str, session_id: str) -> None:
        """Clear all memory for a session."""
        try:
            await self.short_term.clear(user_id, session_id)
            await self.summary.clear(user_id, session_id)
            # Note: Vector memory (entities) are not cleared as they are long-term
            
        except Exception as e:
            self.logger.error(f"Failed to clear session memory: {e}")
    
    async def get_user_profile_summary(self, user_id: str) -> str:
        """Get a summary of the user's profile from stored entities."""
        try:
            # Retrieve all user entities
            entities = await self.vector_store.retrieve_relevant_entities(
                "user profile preferences history", user_id, limit=20
            )
            
            if not entities:
                return "New user - no profile information available."
            
            # Group entities by type
            entity_groups = {}
            for entity in entities:
                entity_type = entity.type.value
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                entity_groups[entity_type].append(entity.content)
            
            # Create profile summary
            profile_parts = []
            for entity_type, contents in entity_groups.items():
                profile_parts.append(f"{entity_type.title()}: {'; '.join(contents[:3])}")  # Limit to 3 items per type
            
            return "User Profile:\n" + "\n".join(profile_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {e}")
            return "Error retrieving user profile."