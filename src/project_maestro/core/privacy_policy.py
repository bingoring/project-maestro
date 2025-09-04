"""개인정보 보호 및 데이터 보존 정책 관리"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine, and_, or_
from pydantic import BaseModel, Field
import uuid

from .config import settings
from .logging import get_logger
from .conversation_memory import get_memory_manager, ConversationMemoryManager

Base = declarative_base()
logger = get_logger("privacy_policy")


class DataCategory(str, Enum):
    """데이터 카테고리"""
    CONVERSATION = "conversation"
    USER_PROFILE = "user_profile"
    PROJECT_DATA = "project_data"
    SYSTEM_LOGS = "system_logs"
    ANALYTICS = "analytics"
    PREFERENCES = "preferences"


class RetentionPeriod(str, Enum):
    """데이터 보존 기간"""
    IMMEDIATE = "immediate"  # 즉시 삭제
    DAYS_30 = "30_days"     # 30일
    DAYS_90 = "90_days"     # 90일
    DAYS_365 = "365_days"   # 1년
    YEARS_7 = "7_years"     # 7년 (법적 요구사항)
    INDEFINITE = "indefinite"  # 무기한 (사용자 동의 하에)


class ConsentType(str, Enum):
    """동의 유형"""
    FUNCTIONAL = "functional"      # 기능 동의
    ANALYTICS = "analytics"        # 분석 동의
    MARKETING = "marketing"        # 마케팅 동의
    PERSONALIZATION = "personalization"  # 개인화 동의


class DataProcessingReason(str, Enum):
    """데이터 처리 목적"""
    SERVICE_PROVISION = "service_provision"
    USER_SUPPORT = "user_support"
    ANALYTICS = "analytics"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"
    MARKETING = "marketing"


# ===== 데이터베이스 모델 =====

class UserConsent(Base):
    """사용자 동의 관리 테이블"""
    __tablename__ = "user_consents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    consent_type = Column(String, nullable=False)
    granted = Column(Boolean, nullable=False, default=False)
    granted_at = Column(DateTime, nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)
    version = Column(String, nullable=False)  # 약관 버전
    metadata = Column(JSONB, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "consent_type": self.consent_type,
            "granted": self.granted,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "version": self.version,
            "metadata": self.metadata
        }


class DataRetentionRule(Base):
    """데이터 보존 규칙 테이블"""
    __tablename__ = "data_retention_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_category = Column(String, nullable=False, index=True)
    retention_period = Column(String, nullable=False)
    processing_reason = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "data_category": self.data_category,
            "retention_period": self.retention_period,
            "processing_reason": self.processing_reason,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata
        }


class DataDeletionRequest(Base):
    """데이터 삭제 요청 테이블"""
    __tablename__ = "data_deletion_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    request_type = Column(String, nullable=False)  # 'forget', 'export', 'delete_all'
    data_categories = Column(JSONB, nullable=False)  # 삭제할 데이터 카테고리 목록
    status = Column(String, default="pending")  # pending, processing, completed, failed
    requested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSONB, default=dict)
    error = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "request_type": self.request_type,
            "data_categories": self.data_categories,
            "status": self.status,
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error
        }


class DataAccessLog(Base):
    """데이터 접근 로그 테이블"""
    __tablename__ = "data_access_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    data_category = Column(String, nullable=False)
    access_type = Column(String, nullable=False)  # 'read', 'write', 'delete', 'export'
    accessed_at = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    metadata = Column(JSONB, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "data_category": self.data_category,
            "access_type": self.access_type,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "metadata": self.metadata
        }


# ===== Pydantic 모델 =====

class ConsentRequest(BaseModel):
    """동의 요청 모델"""
    consent_type: ConsentType
    granted: bool
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None


class ConsentInfo(BaseModel):
    """동의 정보 모델"""
    id: str
    user_id: str
    consent_type: ConsentType
    granted: bool
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    version: str
    metadata: Optional[Dict[str, Any]] = None


class DataDeletionRequestModel(BaseModel):
    """데이터 삭제 요청 모델"""
    request_type: str  # 'forget', 'export', 'delete_all'
    data_categories: List[DataCategory]
    reason: Optional[str] = None


class DataExportResult(BaseModel):
    """데이터 내보내기 결과"""
    conversations: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    consents: List[Dict[str, Any]]
    export_timestamp: datetime
    retention_info: Dict[str, Any]


# ===== 프라이버시 정책 매니저 =====

class PrivacyPolicyManager:
    """개인정보 보호 및 데이터 보존 정책 매니저"""
    
    def __init__(self):
        self.logger = get_logger("privacy_policy_manager")
        
        # 데이터베이스 연결
        self.engine = create_engine(settings.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # 메모리 매니저 참조
        self.memory_manager: Optional[ConversationMemoryManager] = None
        
        # 기본 보존 규칙 초기화
        self._initialize_default_retention_rules()
        
    def get_db_session(self):
        """데이터베이스 세션 생성"""
        return self.SessionLocal()
        
    def _get_memory_manager(self) -> ConversationMemoryManager:
        """지연 로딩으로 메모리 매니저 가져오기"""
        if self.memory_manager is None:
            self.memory_manager = get_memory_manager()
        return self.memory_manager
        
    def _initialize_default_retention_rules(self):
        """기본 데이터 보존 규칙 초기화"""
        try:
            with self.get_db_session() as db:
                # 기본 규칙이 없으면 생성
                existing_rules = db.query(DataRetentionRule).filter(
                    DataRetentionRule.is_active == True
                ).count()
                
                if existing_rules == 0:
                    default_rules = [
                        {
                            "data_category": DataCategory.CONVERSATION.value,
                            "retention_period": RetentionPeriod.DAYS_365.value,
                            "processing_reason": DataProcessingReason.SERVICE_PROVISION.value,
                            "metadata": {"description": "대화 기록 기본 보존"}
                        },
                        {
                            "data_category": DataCategory.USER_PROFILE.value,
                            "retention_period": RetentionPeriod.INDEFINITE.value,
                            "processing_reason": DataProcessingReason.SERVICE_PROVISION.value,
                            "metadata": {"description": "사용자 프로필 정보"}
                        },
                        {
                            "data_category": DataCategory.SYSTEM_LOGS.value,
                            "retention_period": RetentionPeriod.DAYS_90.value,
                            "processing_reason": DataProcessingReason.SECURITY.value,
                            "metadata": {"description": "시스템 로그 보존"}
                        },
                        {
                            "data_category": DataCategory.ANALYTICS.value,
                            "retention_period": RetentionPeriod.DAYS_365.value,
                            "processing_reason": DataProcessingReason.ANALYTICS.value,
                            "metadata": {"description": "분석 데이터 보존"}
                        }
                    ]
                    
                    for rule_data in default_rules:
                        rule = DataRetentionRule(**rule_data)
                        db.add(rule)
                        
                    db.commit()
                    self.logger.info("기본 데이터 보존 규칙 초기화 완료")
                    
        except Exception as e:
            self.logger.error("기본 보존 규칙 초기화 실패", error=str(e))
            
    async def grant_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConsentInfo]:
        """사용자 동의 처리"""
        try:
            with self.get_db_session() as db:
                # 기존 동의 조회
                existing_consent = db.query(UserConsent).filter(
                    and_(
                        UserConsent.user_id == user_id,
                        UserConsent.consent_type == consent_type.value
                    )
                ).order_by(UserConsent.granted_at.desc()).first()
                
                # 새 동의 기록 생성
                new_consent = UserConsent(
                    user_id=user_id,
                    consent_type=consent_type.value,
                    granted=granted,
                    granted_at=datetime.utcnow() if granted else None,
                    withdrawn_at=None if granted else datetime.utcnow(),
                    version=version,
                    metadata=metadata or {}
                )
                
                db.add(new_consent)
                db.commit()
                db.refresh(new_consent)
                
                # 접근 로그 기록
                await self._log_data_access(
                    user_id=user_id,
                    data_category=DataCategory.USER_PROFILE,
                    access_type="write",
                    metadata={
                        "action": "consent_granted" if granted else "consent_withdrawn",
                        "consent_type": consent_type.value
                    }
                )
                
                self.logger.info(
                    "사용자 동의 처리 완료",
                    user_id=user_id,
                    consent_type=consent_type.value,
                    granted=granted
                )
                
                return ConsentInfo(**new_consent.to_dict())
                
        except Exception as e:
            self.logger.error("동의 처리 실패", error=str(e))
            return None
            
    async def get_user_consents(self, user_id: str) -> List[ConsentInfo]:
        """사용자의 현재 동의 상태 조회"""
        try:
            with self.get_db_session() as db:
                # 각 동의 유형별 최신 동의 상태 조회
                consents = []
                for consent_type in ConsentType:
                    latest_consent = db.query(UserConsent).filter(
                        and_(
                            UserConsent.user_id == user_id,
                            UserConsent.consent_type == consent_type.value
                        )
                    ).order_by(UserConsent.granted_at.desc()).first()
                    
                    if latest_consent:
                        consents.append(ConsentInfo(**latest_consent.to_dict()))
                        
                return consents
                
        except Exception as e:
            self.logger.error("동의 상태 조회 실패", error=str(e))
            return []
            
    async def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType
    ) -> bool:
        """특정 동의 유형의 현재 상태 확인"""
        try:
            with self.get_db_session() as db:
                latest_consent = db.query(UserConsent).filter(
                    and_(
                        UserConsent.user_id == user_id,
                        UserConsent.consent_type == consent_type.value
                    )
                ).order_by(UserConsent.granted_at.desc()).first()
                
                return latest_consent.granted if latest_consent else False
                
        except Exception as e:
            self.logger.error("동의 상태 확인 실패", error=str(e))
            return False
            
    async def request_data_deletion(
        self,
        user_id: str,
        request_type: str,
        data_categories: List[DataCategory],
        reason: Optional[str] = None
    ) -> Optional[str]:
        """데이터 삭제 요청"""
        try:
            with self.get_db_session() as db:
                deletion_request = DataDeletionRequest(
                    user_id=user_id,
                    request_type=request_type,
                    data_categories=[cat.value for cat in data_categories],
                    result={"reason": reason} if reason else {}
                )
                
                db.add(deletion_request)
                db.commit()
                db.refresh(deletion_request)
                
                request_id = str(deletion_request.id)
                
                # 백그라운드에서 삭제 처리
                asyncio.create_task(self._process_deletion_request(request_id))
                
                self.logger.info(
                    "데이터 삭제 요청 생성",
                    user_id=user_id,
                    request_id=request_id,
                    request_type=request_type
                )
                
                return request_id
                
        except Exception as e:
            self.logger.error("데이터 삭제 요청 실패", error=str(e))
            return None
            
    async def _process_deletion_request(self, request_id: str):
        """데이터 삭제 요청 처리 (백그라운드 작업)"""
        try:
            with self.get_db_session() as db:
                request = db.query(DataDeletionRequest).filter(
                    DataDeletionRequest.id == uuid.UUID(request_id)
                ).first()
                
                if not request:
                    return
                    
                # 상태 업데이트
                request.status = "processing"
                request.processed_at = datetime.utcnow()
                db.commit()
                
                results = {}
                memory_manager = self._get_memory_manager()
                
                # 데이터 카테고리별 삭제 처리
                for category in request.data_categories:
                    try:
                        if category == DataCategory.CONVERSATION.value:
                            # 대화 데이터 삭제
                            conversations = await memory_manager.list_conversations(
                                user_id=request.user_id,
                                limit=1000
                            )
                            
                            deleted_count = 0
                            for conv in conversations:
                                success = await memory_manager.delete_conversation(
                                    conversation_id=conv.id,
                                    user_id=request.user_id
                                )
                                if success:
                                    deleted_count += 1
                                    
                            results[category] = {
                                "deleted_count": deleted_count,
                                "total_found": len(conversations)
                            }
                            
                        elif category == DataCategory.USER_PROFILE.value:
                            # 사용자 프로필 관련 데이터 삭제
                            # TODO: 실제 사용자 프로필 삭제 로직 구현
                            results[category] = {"status": "not_implemented"}
                            
                        # 다른 카테고리들도 필요시 구현
                            
                    except Exception as e:
                        results[category] = {"error": str(e)}
                        
                # 결과 업데이트
                request.status = "completed"
                request.completed_at = datetime.utcnow()
                request.result = results
                db.commit()
                
                self.logger.info(
                    "데이터 삭제 요청 처리 완료",
                    request_id=request_id,
                    user_id=request.user_id,
                    results=results
                )
                
        except Exception as e:
            # 실패 처리
            with self.get_db_session() as db:
                request = db.query(DataDeletionRequest).filter(
                    DataDeletionRequest.id == uuid.UUID(request_id)
                ).first()
                
                if request:
                    request.status = "failed"
                    request.error = str(e)
                    request.completed_at = datetime.utcnow()
                    db.commit()
                    
            self.logger.error(
                "데이터 삭제 요청 처리 실패",
                request_id=request_id,
                error=str(e)
            )
            
    async def export_user_data(self, user_id: str) -> Optional[DataExportResult]:
        """사용자 데이터 내보내기 (GDPR 데이터 이동권)"""
        try:
            memory_manager = self._get_memory_manager()
            
            # 대화 데이터 수집
            conversations = await memory_manager.list_conversations(
                user_id=user_id,
                limit=1000
            )
            
            conversation_data = []
            for conv in conversations:
                messages = await memory_manager.get_conversation_messages(
                    conversation_id=conv.id,
                    user_id=user_id,
                    limit=1000
                )
                
                conversation_data.append({
                    "conversation": conv.dict(),
                    "messages": [msg.dict() for msg in messages]
                })
                
            # 동의 정보 수집
            consents = await self.get_user_consents(user_id)
            
            # 보존 정책 정보
            with self.get_db_session() as db:
                retention_rules = db.query(DataRetentionRule).filter(
                    DataRetentionRule.is_active == True
                ).all()
                
                retention_info = {
                    rule.data_category: {
                        "retention_period": rule.retention_period,
                        "processing_reason": rule.processing_reason
                    }
                    for rule in retention_rules
                }
                
            # 접근 로그 기록
            await self._log_data_access(
                user_id=user_id,
                data_category=DataCategory.USER_PROFILE,
                access_type="export",
                metadata={"export_timestamp": datetime.utcnow().isoformat()}
            )
            
            export_result = DataExportResult(
                conversations=conversation_data,
                user_preferences={},  # TODO: 실제 사용자 선호사항 수집
                consents=[consent.dict() for consent in consents],
                export_timestamp=datetime.utcnow(),
                retention_info=retention_info
            )
            
            self.logger.info(
                "사용자 데이터 내보내기 완료",
                user_id=user_id,
                conversation_count=len(conversation_data),
                consent_count=len(consents)
            )
            
            return export_result
            
        except Exception as e:
            self.logger.error("사용자 데이터 내보내기 실패", error=str(e))
            return None
            
    async def _log_data_access(
        self,
        user_id: str,
        data_category: DataCategory,
        access_type: str,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """데이터 접근 로그 기록"""
        try:
            with self.get_db_session() as db:
                access_log = DataAccessLog(
                    user_id=user_id,
                    data_category=data_category.value,
                    access_type=access_type,
                    success=success,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata=metadata or {}
                )
                
                db.add(access_log)
                db.commit()
                
        except Exception as e:
            # 로그 기록 실패는 조용히 처리
            self.logger.warning("데이터 접근 로그 기록 실패", error=str(e))
            
    async def cleanup_expired_data(self):
        """만료된 데이터 자동 정리"""
        try:
            with self.get_db_session() as db:
                # 보존 규칙 조회
                retention_rules = db.query(DataRetentionRule).filter(
                    DataRetentionRule.is_active == True
                ).all()
                
                memory_manager = self._get_memory_manager()
                cleanup_results = {}
                
                for rule in retention_rules:
                    try:
                        cutoff_date = self._calculate_cutoff_date(rule.retention_period)
                        if not cutoff_date:
                            continue  # 무기한 보존
                            
                        category = rule.data_category
                        
                        if category == DataCategory.CONVERSATION.value:
                            # 만료된 대화 찾기 및 삭제
                            # TODO: 실제 만료 대화 조회 로직 구현
                            # (현재 대화 모델에 생성일 기반 필터링 추가 필요)
                            cleanup_results[category] = {"status": "규칙 적용됨", "cutoff_date": cutoff_date.isoformat()}
                            
                        elif category == DataCategory.SYSTEM_LOGS.value:
                            # 시스템 로그 정리
                            deleted_logs = db.query(DataAccessLog).filter(
                                DataAccessLog.accessed_at < cutoff_date
                            ).delete()
                            
                            cleanup_results[category] = {
                                "deleted_count": deleted_logs,
                                "cutoff_date": cutoff_date.isoformat()
                            }
                            
                    except Exception as e:
                        cleanup_results[rule.data_category] = {"error": str(e)}
                        
                db.commit()
                
                self.logger.info(
                    "만료된 데이터 정리 완료",
                    results=cleanup_results
                )
                
                return cleanup_results
                
        except Exception as e:
            self.logger.error("만료된 데이터 정리 실패", error=str(e))
            return {}
            
    def _calculate_cutoff_date(self, retention_period: str) -> Optional[datetime]:
        """보존 기간에 따른 만료 날짜 계산"""
        now = datetime.utcnow()
        
        if retention_period == RetentionPeriod.IMMEDIATE.value:
            return now
        elif retention_period == RetentionPeriod.DAYS_30.value:
            return now - timedelta(days=30)
        elif retention_period == RetentionPeriod.DAYS_90.value:
            return now - timedelta(days=90)
        elif retention_period == RetentionPeriod.DAYS_365.value:
            return now - timedelta(days=365)
        elif retention_period == RetentionPeriod.YEARS_7.value:
            return now - timedelta(days=365*7)
        elif retention_period == RetentionPeriod.INDEFINITE.value:
            return None  # 무기한 보존
        else:
            return None
            
    async def get_retention_info(self, user_id: str) -> Dict[str, Any]:
        """사용자의 데이터 보존 정보 조회"""
        try:
            with self.get_db_session() as db:
                retention_rules = db.query(DataRetentionRule).filter(
                    DataRetentionRule.is_active == True
                ).all()
                
                retention_info = {}
                for rule in retention_rules:
                    retention_info[rule.data_category] = {
                        "retention_period": rule.retention_period,
                        "processing_reason": rule.processing_reason,
                        "cutoff_date": self._calculate_cutoff_date(rule.retention_period).isoformat()
                            if self._calculate_cutoff_date(rule.retention_period) else "무기한"
                    }
                    
                return {
                    "user_id": user_id,
                    "retention_rules": retention_info,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("보존 정보 조회 실패", error=str(e))
            return {}


# ===== 전역 매니저 인스턴스 =====

_privacy_manager: Optional[PrivacyPolicyManager] = None


def get_privacy_manager() -> PrivacyPolicyManager:
    """전역 프라이버시 정책 매니저 인스턴스 반환"""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyPolicyManager()
    return _privacy_manager