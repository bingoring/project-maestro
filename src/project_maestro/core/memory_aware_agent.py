"""메모리 인식 에이전트 - 대화 기억 기능을 갖춘 확장된 베이스 에이전트"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .agent_framework import BaseAgent, AgentTask, AgentType
from .conversation_memory import (
    get_memory_manager, 
    ConversationMemoryManager,
    MessageType,
    ConversationInfo,
    MessageInfo,
    SearchResult
)
from .logging import get_logger


class MemoryContext:
    """에이전트를 위한 메모리 컨텍스트"""
    
    def __init__(
        self,
        relevant_conversations: List[ConversationInfo] = None,
        similar_messages: List[MessageInfo] = None,
        project_history: List[str] = None,
        user_preferences: Dict[str, Any] = None
    ):
        self.relevant_conversations = relevant_conversations or []
        self.similar_messages = similar_messages or []
        self.project_history = project_history or []
        self.user_preferences = user_preferences or {}
        
    def to_context_string(self) -> str:
        """메모리 컨텍스트를 문자열로 변환 (프롬프트 주입용)"""
        context_parts = []
        
        if self.relevant_conversations:
            context_parts.append("## 관련 대화 히스토리")
            for conv in self.relevant_conversations[:3]:  # 최대 3개의 관련 대화
                context_parts.append(f"- {conv.title} ({conv.created_at.strftime('%Y-%m-%d')})")
                
        if self.similar_messages:
            context_parts.append("\n## 유사한 이전 상호작용")
            for msg in self.similar_messages[:5]:  # 최대 5개의 유사 메시지
                snippet = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                context_parts.append(f"- {snippet}")
                
        if self.project_history:
            context_parts.append(f"\n## 프로젝트 기록")
            for item in self.project_history[:5]:  # 최대 5개 항목
                context_parts.append(f"- {item}")
                
        if self.user_preferences:
            context_parts.append("\n## 사용자 선호사항")
            for key, value in self.user_preferences.items():
                context_parts.append(f"- {key}: {value}")
                
        return "\n".join(context_parts) if context_parts else ""


class MemoryAwareAgent(BaseAgent):
    """메모리 기능을 갖춘 확장된 베이스 에이전트"""
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        memory_enabled: bool = True,
        context_window_size: int = 10,
        **kwargs
    ):
        super().__init__(name, agent_type, **kwargs)
        self.memory_enabled = memory_enabled
        self.context_window_size = context_window_size
        self.memory_manager: Optional[ConversationMemoryManager] = None
        self.logger = get_logger(f"memory_aware_agent.{name}")
        
        if self.memory_enabled:
            self.memory_manager = get_memory_manager()
            
    async def remember(
        self,
        user_id: str,
        content: str,
        message_type: MessageType = MessageType.AGENT,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """에이전트 동작이나 결과를 메모리에 저장"""
        if not self.memory_enabled or not self.memory_manager:
            return None
            
        try:
            # 대화가 없으면 새로 생성
            if not conversation_id:
                conversation = await self.memory_manager.create_conversation(
                    user_id=user_id,
                    title=f"{self.name} 에이전트 세션",
                    metadata={
                        "agent_type": self.agent_type.value,
                        "agent_name": self.name,
                        "created_by_agent": True
                    }
                )
                if conversation:
                    conversation_id = conversation.id
                else:
                    self.logger.error("대화 생성 실패")
                    return None
                    
            # 메시지 추가
            message_metadata = metadata or {}
            message_metadata.update({
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "timestamp": datetime.now().isoformat()
            })
            
            message = await self.memory_manager.add_message(
                conversation_id=conversation_id,
                message_type=message_type,
                content=content,
                metadata=message_metadata
            )
            
            if message:
                self.logger.info(
                    "메모리에 저장됨",
                    conversation_id=conversation_id,
                    message_id=message.id,
                    content_length=len(content)
                )
                return conversation_id
            else:
                self.logger.error("메시지 저장 실패")
                return None
                
        except Exception as e:
            self.logger.error("메모리 저장 실패", error=str(e))
            return None
            
    async def recall(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = None
    ) -> List[SearchResult]:
        """관련 메모리 검색 및 회상"""
        if not self.memory_enabled or not self.memory_manager:
            return []
            
        try:
            search_limit = limit or self.context_window_size
            
            # 특정 대화 내에서 검색하거나 전체 검색
            if conversation_id:
                # 특정 대화의 메시지 검색 (현재는 기본 구현)
                messages = await self.memory_manager.get_conversation_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    limit=search_limit
                )
                
                # 검색 결과 형태로 변환
                results = []
                for msg in messages:
                    if query.lower() in msg.content.lower():
                        results.append(SearchResult(
                            conversation_id=conversation_id,
                            title=f"Message from {msg.message_type}",
                            relevance_score=0.8,  # 기본값
                            snippet=msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                            created_at=msg.created_at
                        ))
                return results
            else:
                # 전체 대화에서 검색
                return await self.memory_manager.search_conversations(
                    user_id=user_id,
                    query=query,
                    limit=search_limit
                )
                
        except Exception as e:
            self.logger.error("메모리 회상 실패", error=str(e))
            return []
            
    async def get_memory_context(
        self,
        user_id: str,
        current_task: Optional[AgentTask] = None,
        conversation_id: Optional[str] = None
    ) -> MemoryContext:
        """현재 상황에 맞는 메모리 컨텍스트 생성"""
        if not self.memory_enabled or not self.memory_manager:
            return MemoryContext()
            
        try:
            context = MemoryContext()
            
            # 관련 대화 조회
            recent_conversations = await self.memory_manager.list_conversations(
                user_id=user_id,
                limit=10
            )
            
            # 에이전트 타입별로 필터링
            agent_conversations = [
                conv for conv in recent_conversations
                if conv.metadata and conv.metadata.get("agent_type") == self.agent_type.value
            ]
            context.relevant_conversations = agent_conversations[:5]
            
            # 현재 작업과 관련된 컨텍스트 검색
            if current_task:
                search_query = f"{current_task.action} {' '.join(current_task.parameters.keys())}"
                similar_results = await self.memory_manager.search_conversations(
                    user_id=user_id,
                    query=search_query,
                    limit=5
                )
                
                # 검색 결과를 메시지 형태로 변환 (실제로는 메시지 검색 API 필요)
                context.similar_messages = []
                
            # 프로젝트 기록 (메타데이터에서 추출)
            if current_task and current_task.parameters.get("project_id"):
                project_conversations = await self.memory_manager.list_conversations(
                    user_id=user_id,
                    project_id=current_task.parameters["project_id"],
                    limit=5
                )
                context.project_history = [conv.title for conv in project_conversations]
                
            # 사용자 선호사항 (메타데이터에서 추출)
            if recent_conversations:
                preferences = {}
                for conv in recent_conversations[:10]:
                    if conv.metadata:
                        for key, value in conv.metadata.items():
                            if key.startswith("preference_"):
                                preferences[key.replace("preference_", "")] = value
                context.user_preferences = preferences
                
            return context
            
        except Exception as e:
            self.logger.error("메모리 컨텍스트 생성 실패", error=str(e))
            return MemoryContext()
            
    async def forget(
        self,
        user_id: str,
        conversation_id: str,
        reason: str = "User request"
    ) -> bool:
        """특정 대화나 정보를 메모리에서 삭제"""
        if not self.memory_enabled or not self.memory_manager:
            return False
            
        try:
            success = await self.memory_manager.delete_conversation(
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            if success:
                self.logger.info(
                    "메모리에서 삭제됨",
                    conversation_id=conversation_id,
                    reason=reason
                )
                
            return success
            
        except Exception as e:
            self.logger.error("메모리 삭제 실패", error=str(e))
            return False
            
    async def summarize_memory(
        self,
        user_id: str,
        conversation_id: str
    ) -> Optional[str]:
        """대화 내용 요약"""
        if not self.memory_enabled or not self.memory_manager:
            return None
            
        try:
            # 대화 메시지 조회
            messages = await self.memory_manager.get_conversation_messages(
                conversation_id=conversation_id,
                user_id=user_id,
                limit=1000  # 전체 메시지
            )
            
            if not messages:
                return None
                
            # 간단한 통계 기반 요약 (실제로는 LLM을 사용해야 함)
            total_messages = len(messages)
            user_messages = len([m for m in messages if m.message_type == MessageType.USER])
            agent_messages = len([m for m in messages if m.message_type == MessageType.AGENT])
            
            # 주요 키워드 추출 (간단한 구현)
            all_text = " ".join([msg.content for msg in messages])
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # 3글자 이상만
                    word_freq[word] = word_freq.get(word, 0) + 1
                    
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            summary = f"""
대화 요약:
- 총 메시지: {total_messages}개
- 사용자 메시지: {user_messages}개  
- 에이전트 메시지: {agent_messages}개
- 주요 키워드: {', '.join([word for word, _ in top_words[:5]])}
- 기간: {messages[0].created_at.strftime('%Y-%m-%d')} ~ {messages[-1].created_at.strftime('%Y-%m-%d')}
"""
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error("메모리 요약 실패", error=str(e))
            return None
            
    async def execute_task_with_memory(
        self,
        task: AgentTask,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """메모리 컨텍스트를 활용한 작업 실행"""
        if not self.memory_enabled:
            return await self.execute_task(task)
            
        try:
            # 메모리 컨텍스트 준비
            memory_context = await self.get_memory_context(
                user_id=user_id,
                current_task=task,
                conversation_id=conversation_id
            )
            
            # 작업 실행 전 메모리 저장
            await self.remember(
                user_id=user_id,
                content=f"작업 시작: {task.action}",
                message_type=MessageType.SYSTEM,
                conversation_id=conversation_id,
                metadata={
                    "task_id": task.id,
                    "task_action": task.action,
                    "task_parameters": task.parameters
                }
            )
            
            # 메모리 컨텍스트를 task.parameters에 추가
            enhanced_task = task.copy()
            enhanced_task.parameters["memory_context"] = memory_context.to_context_string()
            enhanced_task.parameters["conversation_id"] = conversation_id
            
            # 실제 작업 실행
            result = await self.execute_task(enhanced_task)
            
            # 작업 결과 메모리에 저장
            await self.remember(
                user_id=user_id,
                content=f"작업 완료: {task.action}\n결과: {str(result)[:500]}...",
                message_type=MessageType.AGENT,
                conversation_id=conversation_id,
                metadata={
                    "task_id": task.id,
                    "task_result": result,
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            # 오류도 메모리에 저장
            if conversation_id:
                await self.remember(
                    user_id=user_id,
                    content=f"작업 실패: {task.action}\n오류: {str(e)}",
                    message_type=MessageType.SYSTEM,
                    conversation_id=conversation_id,
                    metadata={
                        "task_id": task.id,
                        "error": str(e),
                        "success": False
                    }
                )
            raise
            
    def _inject_memory_into_prompt(self, base_prompt: str, memory_context: MemoryContext) -> str:
        """기본 프롬프트에 메모리 컨텍스트 주입"""
        context_string = memory_context.to_context_string()
        
        if not context_string:
            return base_prompt
            
        enhanced_prompt = f"""
{base_prompt}

## 메모리 컨텍스트
이전 대화와 상호작용에서 다음 정보를 참고하세요:

{context_string}

위 정보를 참고하여 사용자의 요청에 더욱 맞춤화된 응답을 제공하세요.
"""
        return enhanced_prompt
        
    def get_memory_enhanced_prompt(self, base_prompt: str, memory_context: MemoryContext) -> str:
        """메모리 컨텍스트가 포함된 향상된 프롬프트 반환"""
        return self._inject_memory_into_prompt(base_prompt, memory_context)


class ConversationSession:
    """대화 세션 관리 클래스"""
    
    def __init__(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        memory_manager: Optional[ConversationMemoryManager] = None
    ):
        self.user_id = user_id
        self.project_id = project_id
        self.conversation_id: Optional[str] = None
        self.memory_manager = memory_manager or get_memory_manager()
        self.logger = get_logger("conversation_session")
        
    async def start_session(self, title: Optional[str] = None) -> Optional[str]:
        """새 대화 세션 시작"""
        try:
            conversation = await self.memory_manager.create_conversation(
                user_id=self.user_id,
                project_id=self.project_id,
                title=title or f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                metadata={
                    "session_start": datetime.now().isoformat(),
                    "project_id": self.project_id
                }
            )
            
            if conversation:
                self.conversation_id = conversation.id
                self.logger.info(
                    "대화 세션 시작",
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                return self.conversation_id
                
            return None
            
        except Exception as e:
            self.logger.error("대화 세션 시작 실패", error=str(e))
            return None
            
    async def add_user_message(self, content: str) -> Optional[str]:
        """사용자 메시지 추가"""
        if not self.conversation_id:
            # 세션이 없으면 자동 시작
            await self.start_session()
            
        if self.conversation_id:
            message = await self.memory_manager.add_message(
                conversation_id=self.conversation_id,
                message_type=MessageType.USER,
                content=content
            )
            return message.id if message else None
        return None
        
    async def add_agent_response(self, agent_name: str, content: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """에이전트 응답 추가"""
        if not self.conversation_id:
            return None
            
        response_metadata = metadata or {}
        response_metadata.update({
            "agent_name": agent_name,
            "response_timestamp": datetime.now().isoformat()
        })
        
        message = await self.memory_manager.add_message(
            conversation_id=self.conversation_id,
            message_type=MessageType.ASSISTANT,
            content=content,
            metadata=response_metadata
        )
        return message.id if message else None
        
    async def get_conversation_history(self, limit: int = 50) -> List[MessageInfo]:
        """대화 히스토리 조회"""
        if not self.conversation_id:
            return []
            
        return await self.memory_manager.get_conversation_messages(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            limit=limit
        )
        
    async def end_session(self) -> bool:
        """대화 세션 종료"""
        if not self.conversation_id:
            return True
            
        try:
            # 세션 종료 메시지 추가
            await self.memory_manager.add_message(
                conversation_id=self.conversation_id,
                message_type=MessageType.SYSTEM,
                content="대화 세션 종료",
                metadata={
                    "session_end": datetime.now().isoformat(),
                    "session_duration": "계산 필요"  # TODO: 실제 계산
                }
            )
            
            self.logger.info(
                "대화 세션 종료",
                conversation_id=self.conversation_id,
                user_id=self.user_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error("대화 세션 종료 실패", error=str(e))
            return False
        finally:
            self.conversation_id = None