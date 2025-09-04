"""대화 메모리 시스템 - 사용자 대화 저장, 검색, 컨텍스트 관리"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine, func
from pydantic import BaseModel, Field
import asyncio
from sentence_transformers import SentenceTransformer

from .config import settings
from .logging import get_logger

Base = declarative_base()
logger = get_logger("conversation_memory")


class MessageType(str, Enum):
    """메시지 유형"""
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"
    AGENT = "agent"


class SummaryType(str, Enum):
    """요약 유형"""
    AUTO = "auto"
    MANUAL = "manual"
    PERIODIC = "periodic"


# ===== 데이터베이스 모델 =====

class Conversation(Base):
    """대화 테이블"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    metadata = Column(JSONB, default=dict)
    
    # 관계 설정
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "project_id": str(self.project_id) if self.project_id else None,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


class ConversationMessage(Base):
    """대화 메시지 테이블"""
    __tablename__ = "conversation_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False, index=True)
    message_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, default=dict)
    # pgvector를 사용한 임베딩 벡터 저장 (나중에 ALTER TABLE로 추가)
    # embedding_vector = Column(Vector(384))  # sentence-transformers 차원
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    message_order = Column(Integer, nullable=False)
    
    # 관계 설정
    conversation = relationship("Conversation", back_populates="messages")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "message_order": self.message_order
        }


class ConversationSummary(Base):
    """대화 요약 테이블"""
    __tablename__ = "conversation_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    summary_type = Column(String, nullable=False)
    # embedding_vector = Column(Vector(384))  # 나중에 추가
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 관계 설정
    conversation = relationship("Conversation", back_populates="summaries")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "summary_text": self.summary_text,
            "summary_type": self.summary_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# 인덱스 추가
Index('idx_messages_conversation_order', ConversationMessage.conversation_id, ConversationMessage.message_order)
Index('idx_conversations_user_active', Conversation.user_id, Conversation.is_active)


# ===== Pydantic 모델 =====

class ConversationCreate(BaseModel):
    """대화 생성 요청"""
    user_id: str
    project_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageCreate(BaseModel):
    """메시지 생성 요청"""
    message_type: MessageType
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ConversationInfo(BaseModel):
    """대화 정보"""
    id: str
    user_id: str
    project_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    message_count: Optional[int] = None


class MessageInfo(BaseModel):
    """메시지 정보"""
    id: str
    conversation_id: str
    message_type: MessageType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    message_order: int


class SearchResult(BaseModel):
    """검색 결과"""
    conversation_id: str
    title: str
    relevance_score: float
    snippet: str
    created_at: datetime


# ===== 임베딩 매니저 =====

class EmbeddingManager:
    """텍스트 임베딩 생성 및 관리"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = get_logger("embedding_manager")
        
    async def initialize(self):
        """임베딩 모델 초기화"""
        if self.model is None:
            # 모델 로딩은 CPU 집약적이므로 별도 스레드에서 실행
            self.model = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            self.logger.info(f"Embedding model loaded: {self.model_name}")
            
    async def encode(self, texts: List[str]) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        if self.model is None:
            await self.initialize()
            
        # 임베딩 생성은 CPU 집약적이므로 별도 스레드에서 실행
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            self.model.encode,
            texts
        )
        return embeddings
        
    async def encode_single(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 생성"""
        embeddings = await self.encode([text])
        return embeddings[0]


# ===== 대화 메모리 매니저 =====

class ConversationMemoryManager:
    """대화 메모리 시스템의 핵심 매니저"""
    
    def __init__(self):
        self.logger = get_logger("conversation_memory_manager")
        self.embedding_manager = EmbeddingManager()
        
        # 데이터베이스 연결 설정
        self.engine = create_engine(settings.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def get_db_session(self):
        """데이터베이스 세션 생성"""
        return self.SessionLocal()
        
    async def create_conversation(
        self, 
        user_id: str,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConversationInfo]:
        """새 대화 생성"""
        try:
            # 제목이 없으면 기본값 설정
            if not title:
                title = f"대화 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
            conversation = Conversation(
                user_id=user_id,
                project_id=uuid.UUID(project_id) if project_id else None,
                title=title,
                metadata=metadata or {}
            )
            
            with self.get_db_session() as db:
                db.add(conversation)
                db.commit()
                db.refresh(conversation)
                
            self.logger.info(
                "새 대화 생성됨",
                conversation_id=str(conversation.id),
                user_id=user_id,
                title=title
            )
            
            return ConversationInfo(**conversation.to_dict())
            
        except Exception as e:
            self.logger.error("대화 생성 실패", error=str(e))
            return None
            
    async def add_message(
        self,
        conversation_id: str,
        message_type: MessageType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MessageInfo]:
        """대화에 메시지 추가"""
        try:
            with self.get_db_session() as db:
                # 메시지 순서 결정
                max_order = db.query(func.max(ConversationMessage.message_order))\
                             .filter(ConversationMessage.conversation_id == uuid.UUID(conversation_id))\
                             .scalar() or 0
                
                message = ConversationMessage(
                    conversation_id=uuid.UUID(conversation_id),
                    message_type=message_type.value,
                    content=content,
                    metadata=metadata or {},
                    message_order=max_order + 1
                )
                
                db.add(message)
                
                # 대화의 updated_at 갱신
                db.query(Conversation)\
                  .filter(Conversation.id == uuid.UUID(conversation_id))\
                  .update({"updated_at": datetime.utcnow()})
                  
                db.commit()
                db.refresh(message)
                
            # 백그라운드에서 임베딩 생성 (실제로는 Celery 작업으로 처리)
            asyncio.create_task(self._generate_message_embedding(str(message.id), content))
            
            self.logger.info(
                "메시지 추가됨",
                conversation_id=conversation_id,
                message_id=str(message.id),
                message_type=message_type.value
            )
            
            return MessageInfo(**message.to_dict())
            
        except Exception as e:
            self.logger.error("메시지 추가 실패", error=str(e))
            return None
            
    async def _generate_message_embedding(self, message_id: str, content: str):
        """메시지 임베딩 생성 (백그라운드 작업)"""
        try:
            embedding = await self.embedding_manager.encode_single(content)
            # TODO: pgvector 컬럼이 추가되면 임베딩 저장
            self.logger.debug(f"임베딩 생성 완료: message_id={message_id}")
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[ConversationInfo]:
        """대화 정보 조회"""
        try:
            with self.get_db_session() as db:
                conversation = db.query(Conversation)\
                               .filter(
                                   Conversation.id == uuid.UUID(conversation_id),
                                   Conversation.user_id == user_id,
                                   Conversation.is_active == True
                               ).first()
                               
                if not conversation:
                    return None
                    
                # 메시지 수 조회
                message_count = db.query(ConversationMessage)\
                                 .filter(ConversationMessage.conversation_id == conversation.id)\
                                 .count()
                                 
                result = ConversationInfo(**conversation.to_dict())
                result.message_count = message_count
                return result
                
        except Exception as e:
            self.logger.error("대화 조회 실패", error=str(e))
            return None
            
    async def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[MessageInfo]:
        """대화의 메시지 목록 조회"""
        try:
            with self.get_db_session() as db:
                # 사용자 권한 확인
                conversation = db.query(Conversation)\
                               .filter(
                                   Conversation.id == uuid.UUID(conversation_id),
                                   Conversation.user_id == user_id,
                                   Conversation.is_active == True
                               ).first()
                               
                if not conversation:
                    return []
                    
                messages = db.query(ConversationMessage)\
                            .filter(ConversationMessage.conversation_id == uuid.UUID(conversation_id))\
                            .order_by(ConversationMessage.message_order)\
                            .offset(offset)\
                            .limit(limit)\
                            .all()
                            
                return [MessageInfo(**msg.to_dict()) for msg in messages]
                
        except Exception as e:
            self.logger.error("메시지 조회 실패", error=str(e))
            return []
            
    async def list_conversations(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ConversationInfo]:
        """사용자의 대화 목록 조회"""
        try:
            with self.get_db_session() as db:
                query = db.query(Conversation)\
                          .filter(
                              Conversation.user_id == user_id,
                              Conversation.is_active == True
                          )
                          
                if project_id:
                    query = query.filter(Conversation.project_id == uuid.UUID(project_id))
                    
                conversations = query.order_by(Conversation.updated_at.desc())\
                                   .offset(offset)\
                                   .limit(limit)\
                                   .all()
                                   
                result = []
                for conv in conversations:
                    # 메시지 수 조회
                    message_count = db.query(ConversationMessage)\
                                     .filter(ConversationMessage.conversation_id == conv.id)\
                                     .count()
                                     
                    conv_info = ConversationInfo(**conv.to_dict())
                    conv_info.message_count = message_count
                    result.append(conv_info)
                    
                return result
                
        except Exception as e:
            self.logger.error("대화 목록 조회 실패", error=str(e))
            return []
            
    async def search_conversations(
        self,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """대화 내용 검색 (기본 텍스트 검색)"""
        try:
            with self.get_db_session() as db:
                # 기본 텍스트 검색 (나중에 벡터 검색으로 개선)
                search_query = db.query(Conversation, ConversationMessage)\
                                .join(ConversationMessage)\
                                .filter(
                                    Conversation.user_id == user_id,
                                    Conversation.is_active == True,
                                    ConversationMessage.content.ilike(f"%{query}%")
                                )
                                
                if project_id:
                    search_query = search_query.filter(Conversation.project_id == uuid.UUID(project_id))
                    
                results = search_query.order_by(Conversation.updated_at.desc())\
                                    .limit(limit)\
                                    .all()
                                    
                search_results = []
                for conv, msg in results:
                    # 검색어 주변 컨텍스트 추출
                    snippet = self._extract_snippet(msg.content, query)
                    
                    search_results.append(SearchResult(
                        conversation_id=str(conv.id),
                        title=conv.title,
                        relevance_score=0.5,  # 기본값, 나중에 벡터 유사도로 개선
                        snippet=snippet,
                        created_at=conv.created_at
                    ))
                    
                return search_results
                
        except Exception as e:
            self.logger.error("대화 검색 실패", error=str(e))
            return []
            
    def _extract_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """검색어 주변 컨텍스트 추출"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        pos = text_lower.find(query_lower)
        if pos == -1:
            return text[:max_length] + ("..." if len(text) > max_length else "")
            
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + len(query) + max_length // 2)
        
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
            
        return snippet
        
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """대화 삭제 (소프트 삭제)"""
        try:
            with self.get_db_session() as db:
                result = db.query(Conversation)\
                           .filter(
                               Conversation.id == uuid.UUID(conversation_id),
                               Conversation.user_id == user_id
                           )\
                           .update({
                               "is_active": False,
                               "updated_at": datetime.utcnow()
                           })
                           
                db.commit()
                
                if result > 0:
                    self.logger.info("대화 삭제됨", conversation_id=conversation_id, user_id=user_id)
                    return True
                    
                return False
                
        except Exception as e:
            self.logger.error("대화 삭제 실패", error=str(e))
            return False
            
    async def update_conversation_title(
        self,
        conversation_id: str,
        user_id: str,
        new_title: str
    ) -> bool:
        """대화 제목 수정"""
        try:
            with self.get_db_session() as db:
                result = db.query(Conversation)\
                           .filter(
                               Conversation.id == uuid.UUID(conversation_id),
                               Conversation.user_id == user_id,
                               Conversation.is_active == True
                           )\
                           .update({
                               "title": new_title,
                               "updated_at": datetime.utcnow()
                           })
                           
                db.commit()
                return result > 0
                
        except Exception as e:
            self.logger.error("대화 제목 수정 실패", error=str(e))
            return False


# ===== 전역 매니저 인스턴스 =====

_memory_manager: Optional[ConversationMemoryManager] = None


def get_memory_manager() -> ConversationMemoryManager:
    """전역 대화 메모리 매니저 인스턴스 반환"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationMemoryManager()
    return _memory_manager