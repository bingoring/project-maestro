"""대화 메모리 API 엔드포인트"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from ...core.conversation_memory import (
    get_memory_manager,
    ConversationMemoryManager,
    ConversationCreate,
    MessageCreate,
    ConversationInfo,
    MessageInfo,
    SearchResult,
    MessageType
)
from ...core.logging import get_logger

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])
logger = get_logger("conversation_api")


# ===== 요청/응답 모델 =====

class ConversationCreateRequest(BaseModel):
    """대화 생성 요청"""
    title: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Optional[dict] = None


class MessageCreateRequest(BaseModel):
    """메시지 생성 요청"""
    message_type: MessageType
    content: str
    metadata: Optional[dict] = None


class ConversationUpdateRequest(BaseModel):
    """대화 수정 요청"""
    title: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str
    project_id: Optional[str] = None
    limit: Optional[int] = 20


class ConversationResponse(BaseModel):
    """대화 응답"""
    success: bool
    data: Optional[ConversationInfo] = None
    message: str = ""


class ConversationListResponse(BaseModel):
    """대화 목록 응답"""
    success: bool
    data: List[ConversationInfo]
    total: Optional[int] = None
    message: str = ""


class MessageListResponse(BaseModel):
    """메시지 목록 응답"""
    success: bool
    data: List[MessageInfo]
    message: str = ""


class SearchResponse(BaseModel):
    """검색 응답"""
    success: bool
    data: List[SearchResult]
    message: str = ""


# ===== 의존성 =====

async def get_current_user_id() -> str:
    """현재 사용자 ID 반환 (실제로는 JWT 토큰에서 추출)"""
    # TODO: JWT 인증 구현
    return "test_user"


def get_memory_service() -> ConversationMemoryManager:
    """대화 메모리 매니저 의존성"""
    return get_memory_manager()


# ===== API 엔드포인트 =====

@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """새 대화 생성"""
    try:
        conversation = await memory.create_conversation(
            user_id=user_id,
            project_id=request.project_id,
            title=request.title,
            metadata=request.metadata
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="대화 생성에 실패했습니다."
            )
            
        return ConversationResponse(
            success=True,
            data=conversation,
            message="대화가 성공적으로 생성되었습니다."
        )
        
    except Exception as e:
        logger.error("대화 생성 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/", response_model=ConversationListResponse)
async def list_conversations(
    project_id: Optional[str] = Query(None, description="프로젝트 ID로 필터링"),
    limit: int = Query(50, ge=1, le=100, description="조회할 대화 수"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """사용자의 대화 목록 조회"""
    try:
        conversations = await memory.list_conversations(
            user_id=user_id,
            project_id=project_id,
            limit=limit,
            offset=offset
        )
        
        return ConversationListResponse(
            success=True,
            data=conversations,
            message=f"{len(conversations)}개의 대화를 찾았습니다."
        )
        
    except Exception as e:
        logger.error("대화 목록 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """특정 대화 정보 조회"""
    try:
        conversation = await memory.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다."
            )
            
        return ConversationResponse(
            success=True,
            data=conversation,
            message="대화 정보를 조회했습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("대화 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화 정보 수정"""
    try:
        if request.title:
            success = await memory.update_conversation_title(
                conversation_id=conversation_id,
                user_id=user_id,
                new_title=request.title
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="대화를 찾을 수 없거나 수정 권한이 없습니다."
                )
                
        # 수정된 대화 정보 반환
        conversation = await memory.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        return ConversationResponse(
            success=True,
            data=conversation,
            message="대화 정보가 수정되었습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("대화 수정 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 수정 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화 삭제"""
    try:
        success = await memory.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없거나 삭제 권한이 없습니다."
            )
            
        return {"success": True, "message": "대화가 삭제되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("대화 삭제 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/{conversation_id}/messages", response_model=MessageListResponse)
async def add_message(
    conversation_id: str,
    request: MessageCreateRequest,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화에 메시지 추가"""
    try:
        # 대화 존재 및 권한 확인
        conversation = await memory.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다."
            )
            
        message = await memory.add_message(
            conversation_id=conversation_id,
            message_type=request.message_type,
            content=request.content,
            metadata=request.metadata
        )
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="메시지 추가에 실패했습니다."
            )
            
        return MessageListResponse(
            success=True,
            data=[message],
            message="메시지가 추가되었습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("메시지 추가 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"메시지 추가 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500, description="조회할 메시지 수"),
    offset: int = Query(0, ge=0, description="시작 위치"),
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화의 메시지 목록 조회"""
    try:
        messages = await memory.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return MessageListResponse(
            success=True,
            data=messages,
            message=f"{len(messages)}개의 메시지를 조회했습니다."
        )
        
    except Exception as e:
        logger.error("메시지 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"메시지 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_conversations(
    request: SearchRequest,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화 내용 검색"""
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="검색어를 입력해주세요."
            )
            
        results = await memory.search_conversations(
            user_id=user_id,
            query=request.query,
            project_id=request.project_id,
            limit=request.limit or 20
        )
        
        return SearchResponse(
            success=True,
            data=results,
            message=f"'{request.query}'에 대해 {len(results)}개의 결과를 찾았습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("대화 검색 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 검색 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/{conversation_id}/context")
async def get_conversation_context(
    conversation_id: str,
    limit: int = Query(10, ge=1, le=50, description="관련 대화 수"),
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화의 관련 컨텍스트 조회 (관련 대화들)"""
    try:
        # 현재는 기본 구현으로 최근 대화들 반환
        # TODO: 벡터 유사도 기반 관련 대화 검색 구현
        conversation = await memory.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다."
            )
            
        related_conversations = await memory.list_conversations(
            user_id=user_id,
            project_id=conversation.project_id,
            limit=limit,
            offset=0
        )
        
        # 현재 대화 제외
        related_conversations = [
            conv for conv in related_conversations 
            if conv.id != conversation_id
        ]
        
        return ConversationListResponse(
            success=True,
            data=related_conversations[:limit-1],
            message=f"{len(related_conversations)}개의 관련 대화를 찾았습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("컨텍스트 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"컨텍스트 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/{conversation_id}/summarize")
async def summarize_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    memory: ConversationMemoryManager = Depends(get_memory_service)
):
    """대화 요약 생성"""
    try:
        # 대화 존재 확인
        conversation = await memory.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다."
            )
            
        # 메시지 조회
        messages = await memory.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=1000  # 전체 메시지 조회
        )
        
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="요약할 메시지가 없습니다."
            )
            
        # TODO: AI 기반 요약 생성 (OpenAI/Claude API 사용)
        # 현재는 간단한 통계 정보 반환
        user_messages = len([m for m in messages if m.message_type == MessageType.USER])
        assistant_messages = len([m for m in messages if m.message_type == MessageType.ASSISTANT])
        
        summary = {
            "conversation_id": conversation_id,
            "title": conversation.title,
            "message_count": len(messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "duration": "대화 기간 분석 필요",  # TODO: 시작-종료 시간 계산
            "topics": ["주제 분석 기능 구현 필요"],  # TODO: 주제 추출
            "summary": "AI 기반 요약 기능 구현 필요"  # TODO: 실제 요약 생성
        }
        
        return {
            "success": True,
            "data": summary,
            "message": "대화 요약이 생성되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("대화 요약 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 요약 중 오류가 발생했습니다: {str(e)}"
        )