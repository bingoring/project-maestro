"""대화 메모리 시스템 테스트"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from src.project_maestro.core.conversation_memory import (
    ConversationMemoryManager,
    EmbeddingManager,
    MessageType,
    ConversationInfo,
    MessageInfo,
    SearchResult
)
from src.project_maestro.core.memory_aware_agent import (
    MemoryAwareAgent,
    MemoryContext,
    ConversationSession
)
from src.project_maestro.core.privacy_policy import (
    PrivacyPolicyManager,
    ConsentType,
    DataCategory,
    RetentionPeriod
)


class TestEmbeddingManager:
    """임베딩 매니저 테스트"""
    
    @pytest.fixture
    def embedding_manager(self):
        return EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")
    
    @pytest.mark.asyncio
    async def test_embedding_initialization(self, embedding_manager):
        """임베딩 모델 초기화 테스트"""
        # 모델이 아직 로드되지 않았는지 확인
        assert embedding_manager.model is None
        
        # 초기화 실행
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_model.return_value = Mock()
            await embedding_manager.initialize()
            
            # 모델이 로드되었는지 확인
            assert embedding_manager.model is not None
            mock_model.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    
    @pytest.mark.asyncio
    async def test_encode_single_text(self, embedding_manager):
        """단일 텍스트 임베딩 생성 테스트"""
        test_text = "안녕하세요, 테스트 메시지입니다."
        
        with patch.object(embedding_manager, 'model') as mock_model:
            # Mock 임베딩 결과
            mock_embedding = [[0.1, 0.2, 0.3]]
            mock_model.encode.return_value = mock_embedding
            
            result = await embedding_manager.encode_single(test_text)
            
            # 결과 검증
            assert result is not None
            assert len(result) == 3
            mock_model.encode.assert_called_once_with([test_text])
    
    @pytest.mark.asyncio
    async def test_encode_multiple_texts(self, embedding_manager):
        """다중 텍스트 임베딩 생성 테스트"""
        test_texts = [
            "첫 번째 메시지",
            "두 번째 메시지",
            "세 번째 메시지"
        ]
        
        with patch.object(embedding_manager, 'model') as mock_model:
            # Mock 임베딩 결과
            mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            mock_model.encode.return_value = mock_embeddings
            
            results = await embedding_manager.encode(test_texts)
            
            # 결과 검증
            assert results is not None
            assert len(results) == 3
            mock_model.encode.assert_called_once_with(test_texts)


class TestConversationMemoryManager:
    """대화 메모리 매니저 테스트"""
    
    @pytest.fixture
    def memory_manager(self):
        with patch('src.project_maestro.core.conversation_memory.create_engine'):
            with patch('src.project_maestro.core.conversation_memory.sessionmaker'):
                manager = ConversationMemoryManager()
                # Mock 데이터베이스 세션
                manager.get_db_session = Mock()
                return manager
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, memory_manager):
        """대화 생성 테스트"""
        user_id = "test_user_123"
        project_id = str(uuid.uuid4())
        title = "테스트 대화"
        
        # Mock 데이터베이스 세션과 객체
        mock_session = Mock()
        mock_conversation = Mock()
        mock_conversation.id = uuid.uuid4()
        mock_conversation.to_dict.return_value = {
            "id": str(mock_conversation.id),
            "user_id": user_id,
            "project_id": project_id,
            "title": title,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "is_active": True,
            "metadata": {}
        }
        
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        memory_manager.get_db_session.return_value = mock_session
        
        # 대화 생성 실행
        result = await memory_manager.create_conversation(
            user_id=user_id,
            project_id=project_id,
            title=title
        )
        
        # 결과 검증
        assert result is not None
        assert isinstance(result, ConversationInfo)
        assert result.user_id == user_id
        assert result.title == title
        
        # 데이터베이스 호출 검증
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_message(self, memory_manager):
        """메시지 추가 테스트"""
        conversation_id = str(uuid.uuid4())
        message_type = MessageType.USER
        content = "테스트 메시지입니다."
        
        # Mock 데이터베이스 세션
        mock_session = Mock()
        mock_message = Mock()
        mock_message.id = uuid.uuid4()
        mock_message.to_dict.return_value = {
            "id": str(mock_message.id),
            "conversation_id": conversation_id,
            "message_type": message_type.value,
            "content": content,
            "metadata": {},
            "created_at": datetime.now(),
            "message_order": 1
        }
        
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.scalar.return_value = 0  # max_order
        mock_session.query.return_value.filter.return_value.update = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        memory_manager.get_db_session.return_value = mock_session
        
        # 임베딩 생성 Mock
        with patch.object(memory_manager, '_generate_message_embedding') as mock_embedding:
            mock_embedding.return_value = None
            
            # 메시지 추가 실행
            result = await memory_manager.add_message(
                conversation_id=conversation_id,
                message_type=message_type,
                content=content
            )
            
            # 결과 검증
            assert result is not None
            assert isinstance(result, MessageInfo)
            assert result.content == content
            assert result.message_type == message_type
            
            # 임베딩 생성 호출 검증
            mock_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_conversations(self, memory_manager):
        """대화 검색 테스트"""
        user_id = "test_user_123"
        query = "테스트 검색어"
        
        # Mock 검색 결과
        mock_conversation = Mock()
        mock_conversation.id = uuid.uuid4()
        mock_conversation.title = "테스트 대화"
        mock_conversation.created_at = datetime.now()
        mock_conversation.updated_at = datetime.now()
        
        mock_message = Mock()
        mock_message.content = "테스트 검색어가 포함된 메시지입니다."
        
        # Mock 데이터베이스 쿼리
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_query_result = [(mock_conversation, mock_message)]
        
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_query_result
        mock_session.query.return_value = mock_query
        
        memory_manager.get_db_session.return_value = mock_session
        
        # 검색 실행
        results = await memory_manager.search_conversations(
            user_id=user_id,
            query=query,
            limit=10
        )
        
        # 결과 검증
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert query in results[0].snippet
    
    @pytest.mark.asyncio
    async def test_delete_conversation(self, memory_manager):
        """대화 삭제 테스트"""
        conversation_id = str(uuid.uuid4())
        user_id = "test_user_123"
        
        # Mock 데이터베이스 세션
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.update.return_value = 1
        mock_session.commit = Mock()
        
        memory_manager.get_db_session.return_value = mock_session
        
        # 삭제 실행
        result = await memory_manager.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        # 결과 검증
        assert result is True
        mock_session.commit.assert_called_once()


class TestMemoryAwareAgent:
    """메모리 인식 에이전트 테스트"""
    
    @pytest.fixture
    def memory_agent(self):
        with patch('src.project_maestro.core.memory_aware_agent.get_memory_manager'):
            from src.project_maestro.core.agent_framework import AgentType
            
            agent = MemoryAwareAgent(
                name="test_agent",
                agent_type=AgentType.CODEX,
                memory_enabled=True
            )
            
            # Mock 메모리 매니저
            agent.memory_manager = Mock()
            return agent
    
    @pytest.mark.asyncio
    async def test_remember_functionality(self, memory_agent):
        """기억 기능 테스트"""
        user_id = "test_user_123"
        content = "에이전트가 기억해야 할 내용"
        conversation_id = str(uuid.uuid4())
        
        # Mock 메모리 매니저 응답
        mock_message = MessageInfo(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.AGENT,
            content=content,
            created_at=datetime.now(),
            message_order=1
        )
        
        memory_agent.memory_manager.add_message = AsyncMock(return_value=mock_message)
        
        # 기억 실행
        result = await memory_agent.remember(
            user_id=user_id,
            content=content,
            conversation_id=conversation_id
        )
        
        # 결과 검증
        assert result == conversation_id
        memory_agent.memory_manager.add_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recall_functionality(self, memory_agent):
        """회상 기능 테스트"""
        user_id = "test_user_123"
        query = "이전에 논의한 내용"
        
        # Mock 검색 결과
        mock_results = [
            SearchResult(
                conversation_id=str(uuid.uuid4()),
                title="이전 대화",
                relevance_score=0.8,
                snippet="이전에 논의한 내용과 관련된 메시지",
                created_at=datetime.now()
            )
        ]
        
        memory_agent.memory_manager.search_conversations = AsyncMock(return_value=mock_results)
        
        # 회상 실행
        results = await memory_agent.recall(
            user_id=user_id,
            query=query
        )
        
        # 결과 검증
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        memory_agent.memory_manager.search_conversations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_memory_context(self, memory_agent):
        """메모리 컨텍스트 생성 테스트"""
        user_id = "test_user_123"
        
        # Mock 관련 대화 목록
        mock_conversations = [
            ConversationInfo(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title="관련 대화 1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"agent_type": "codex"}
            )
        ]
        
        memory_agent.memory_manager.list_conversations = AsyncMock(return_value=mock_conversations)
        memory_agent.memory_manager.search_conversations = AsyncMock(return_value=[])
        
        # 메모리 컨텍스트 생성
        context = await memory_agent.get_memory_context(user_id=user_id)
        
        # 결과 검증
        assert isinstance(context, MemoryContext)
        assert len(context.relevant_conversations) == 1
        
        # 컨텍스트 문자열 변환 테스트
        context_string = context.to_context_string()
        assert "관련 대화 히스토리" in context_string
    
    @pytest.mark.asyncio
    async def test_forget_functionality(self, memory_agent):
        """망각 기능 테스트"""
        user_id = "test_user_123"
        conversation_id = str(uuid.uuid4())
        
        memory_agent.memory_manager.delete_conversation = AsyncMock(return_value=True)
        
        # 망각 실행
        result = await memory_agent.forget(
            user_id=user_id,
            conversation_id=conversation_id,
            reason="사용자 요청"
        )
        
        # 결과 검증
        assert result is True
        memory_agent.memory_manager.delete_conversation.assert_called_once_with(
            conversation_id=conversation_id,
            user_id=user_id
        )


class TestConversationSession:
    """대화 세션 테스트"""
    
    @pytest.fixture
    def conversation_session(self):
        user_id = "test_user_123"
        project_id = str(uuid.uuid4())
        
        with patch('src.project_maestro.core.memory_aware_agent.get_memory_manager'):
            session = ConversationSession(
                user_id=user_id,
                project_id=project_id
            )
            session.memory_manager = Mock()
            return session
    
    @pytest.mark.asyncio
    async def test_start_session(self, conversation_session):
        """세션 시작 테스트"""
        # Mock 대화 생성 결과
        mock_conversation = ConversationInfo(
            id=str(uuid.uuid4()),
            user_id=conversation_session.user_id,
            title="테스트 세션",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        conversation_session.memory_manager.create_conversation = AsyncMock(return_value=mock_conversation)
        
        # 세션 시작
        conversation_id = await conversation_session.start_session("테스트 세션")
        
        # 결과 검증
        assert conversation_id == mock_conversation.id
        assert conversation_session.conversation_id == mock_conversation.id
    
    @pytest.mark.asyncio
    async def test_add_user_message(self, conversation_session):
        """사용자 메시지 추가 테스트"""
        conversation_session.conversation_id = str(uuid.uuid4())
        content = "사용자 메시지 테스트"
        
        # Mock 메시지 추가 결과
        mock_message = MessageInfo(
            id=str(uuid.uuid4()),
            conversation_id=conversation_session.conversation_id,
            message_type=MessageType.USER,
            content=content,
            created_at=datetime.now(),
            message_order=1
        )
        
        conversation_session.memory_manager.add_message = AsyncMock(return_value=mock_message)
        
        # 메시지 추가
        message_id = await conversation_session.add_user_message(content)
        
        # 결과 검증
        assert message_id == mock_message.id
        conversation_session.memory_manager.add_message.assert_called_once_with(
            conversation_id=conversation_session.conversation_id,
            message_type=MessageType.USER,
            content=content
        )
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, conversation_session):
        """대화 히스토리 조회 테스트"""
        conversation_session.conversation_id = str(uuid.uuid4())
        
        # Mock 메시지 목록
        mock_messages = [
            MessageInfo(
                id=str(uuid.uuid4()),
                conversation_id=conversation_session.conversation_id,
                message_type=MessageType.USER,
                content="사용자 메시지",
                created_at=datetime.now(),
                message_order=1
            ),
            MessageInfo(
                id=str(uuid.uuid4()),
                conversation_id=conversation_session.conversation_id,
                message_type=MessageType.ASSISTANT,
                content="어시스턴트 응답",
                created_at=datetime.now(),
                message_order=2
            )
        ]
        
        conversation_session.memory_manager.get_conversation_messages = AsyncMock(return_value=mock_messages)
        
        # 히스토리 조회
        history = await conversation_session.get_conversation_history()
        
        # 결과 검증
        assert len(history) == 2
        assert all(isinstance(msg, MessageInfo) for msg in history)


class TestPrivacyPolicyManager:
    """프라이버시 정책 매니저 테스트"""
    
    @pytest.fixture
    def privacy_manager(self):
        with patch('src.project_maestro.core.privacy_policy.create_engine'):
            with patch('src.project_maestro.core.privacy_policy.sessionmaker'):
                manager = PrivacyPolicyManager()
                manager.get_db_session = Mock()
                return manager
    
    @pytest.mark.asyncio
    async def test_grant_consent(self, privacy_manager):
        """동의 처리 테스트"""
        user_id = "test_user_123"
        consent_type = ConsentType.FUNCTIONAL
        
        # Mock 데이터베이스 세션
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        privacy_manager.get_db_session.return_value = mock_session
        
        # Mock 접근 로그
        with patch.object(privacy_manager, '_log_data_access') as mock_log:
            mock_log.return_value = None
            
            # 동의 처리
            result = await privacy_manager.grant_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=True
            )
            
            # 결과 검증
            assert result is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_consent(self, privacy_manager):
        """동의 상태 확인 테스트"""
        user_id = "test_user_123"
        consent_type = ConsentType.ANALYTICS
        
        # Mock 동의 기록
        mock_consent = Mock()
        mock_consent.granted = True
        
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_consent
        
        privacy_manager.get_db_session.return_value = mock_session
        
        # 동의 상태 확인
        result = await privacy_manager.check_consent(
            user_id=user_id,
            consent_type=consent_type
        )
        
        # 결과 검증
        assert result is True
    
    @pytest.mark.asyncio
    async def test_request_data_deletion(self, privacy_manager):
        """데이터 삭제 요청 테스트"""
        user_id = "test_user_123"
        data_categories = [DataCategory.CONVERSATION, DataCategory.USER_PROFILE]
        
        # Mock 데이터베이스 세션
        mock_session = Mock()
        mock_deletion_request = Mock()
        mock_deletion_request.id = uuid.uuid4()
        
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        privacy_manager.get_db_session.return_value = mock_session
        
        # Mock 백그라운드 처리
        with patch('asyncio.create_task') as mock_task:
            mock_task.return_value = None
            
            # 삭제 요청
            request_id = await privacy_manager.request_data_deletion(
                user_id=user_id,
                request_type="delete_all",
                data_categories=data_categories
            )
            
            # 결과 검증
            assert request_id is not None
            mock_session.add.assert_called_once()
            mock_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_export_user_data(self, privacy_manager):
        """데이터 내보내기 테스트"""
        user_id = "test_user_123"
        
        # Mock 메모리 매니저
        mock_memory_manager = Mock()
        mock_conversations = [
            ConversationInfo(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title="테스트 대화",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        mock_memory_manager.list_conversations = AsyncMock(return_value=mock_conversations)
        mock_memory_manager.get_conversation_messages = AsyncMock(return_value=[])
        
        privacy_manager._get_memory_manager = Mock(return_value=mock_memory_manager)
        
        # Mock 동의 정보
        with patch.object(privacy_manager, 'get_user_consents') as mock_consents:
            mock_consents.return_value = []
            
            # Mock 데이터베이스 세션
            mock_session = Mock()
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=None)
            mock_session.query.return_value.filter.return_value.all.return_value = []
            
            privacy_manager.get_db_session.return_value = mock_session
            
            # Mock 접근 로그
            with patch.object(privacy_manager, '_log_data_access') as mock_log:
                mock_log.return_value = None
                
                # 데이터 내보내기
                result = await privacy_manager.export_user_data(user_id)
                
                # 결과 검증
                assert result is not None
                assert len(result.conversations) == 1
                mock_log.assert_called_once()


# 통합 테스트
class TestConversationMemoryIntegration:
    """대화 메모리 시스템 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """전체 대화 흐름 통합 테스트"""
        user_id = "integration_test_user"
        
        # Mock 모든 의존성
        with patch('src.project_maestro.core.conversation_memory.create_engine'):
            with patch('src.project_maestro.core.conversation_memory.sessionmaker'):
                with patch('src.project_maestro.core.memory_aware_agent.get_memory_manager'):
                    
                    # 1. 대화 세션 시작
                    session = ConversationSession(user_id=user_id)
                    session.memory_manager = Mock()
                    
                    mock_conversation = ConversationInfo(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        title="통합 테스트 대화",
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    session.memory_manager.create_conversation = AsyncMock(return_value=mock_conversation)
                    session.memory_manager.add_message = AsyncMock(return_value=Mock())
                    
                    # 세션 시작
                    conversation_id = await session.start_session("통합 테스트 대화")
                    assert conversation_id is not None
                    
                    # 2. 사용자 메시지 추가
                    await session.add_user_message("안녕하세요, 테스트입니다.")
                    
                    # 3. 에이전트 응답 추가
                    await session.add_agent_response(
                        agent_name="test_agent",
                        content="안녕하세요! 도움이 필요하시면 말씀해주세요.",
                        metadata={"response_type": "greeting"}
                    )
                    
                    # 4. 세션 종료
                    result = await session.end_session()
                    assert result is True
                    
                    # Mock 호출 검증
                    assert session.memory_manager.create_conversation.called
                    assert session.memory_manager.add_message.call_count == 3  # user + agent + system


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])