"""프라이버시 및 데이터 관리 API 엔드포인트"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...core.privacy_policy import (
    get_privacy_manager,
    PrivacyPolicyManager,
    ConsentRequest,
    ConsentInfo,
    DataDeletionRequestModel,
    DataExportResult,
    ConsentType,
    DataCategory
)
from ...core.logging import get_logger

router = APIRouter(prefix="/api/v1/privacy", tags=["privacy"])
logger = get_logger("privacy_api")


# ===== 요청/응답 모델 =====

class ConsentUpdateRequest(BaseModel):
    """동의 상태 업데이트 요청"""
    consent_type: ConsentType
    granted: bool
    version: str = "1.0"


class ConsentResponse(BaseModel):
    """동의 응답"""
    success: bool
    data: Optional[ConsentInfo] = None
    message: str = ""


class ConsentListResponse(BaseModel):
    """동의 목록 응답"""
    success: bool
    data: List[ConsentInfo]
    message: str = ""


class DeletionRequestResponse(BaseModel):
    """삭제 요청 응답"""
    success: bool
    request_id: Optional[str] = None
    message: str = ""


class DataExportResponse(BaseModel):
    """데이터 내보내기 응답"""
    success: bool
    data: Optional[DataExportResult] = None
    message: str = ""


class RetentionInfoResponse(BaseModel):
    """데이터 보존 정보 응답"""
    success: bool
    data: Optional[dict] = None
    message: str = ""


# ===== 의존성 =====

async def get_current_user_id() -> str:
    """현재 사용자 ID 반환 (실제로는 JWT 토큰에서 추출)"""
    # TODO: JWT 인증 구현
    return "test_user"


def get_privacy_service() -> PrivacyPolicyManager:
    """프라이버시 정책 매니저 의존성"""
    return get_privacy_manager()


# ===== API 엔드포인트 =====

@router.post("/consent", response_model=ConsentResponse)
async def update_consent(
    request: ConsentUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """사용자 동의 상태 업데이트"""
    try:
        consent = await privacy_manager.grant_consent(
            user_id=user_id,
            consent_type=request.consent_type,
            granted=request.granted,
            version=request.version
        )
        
        if not consent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="동의 상태 업데이트에 실패했습니다."
            )
            
        action = "동의" if request.granted else "철회"
        return ConsentResponse(
            success=True,
            data=consent,
            message=f"{request.consent_type.value} 동의가 {action}되었습니다."
        )
        
    except Exception as e:
        logger.error("동의 상태 업데이트 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"동의 상태 업데이트 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/consent", response_model=ConsentListResponse)
async def get_consents(
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """사용자의 현재 동의 상태 조회"""
    try:
        consents = await privacy_manager.get_user_consents(user_id)
        
        return ConsentListResponse(
            success=True,
            data=consents,
            message=f"{len(consents)}개의 동의 상태를 조회했습니다."
        )
        
    except Exception as e:
        logger.error("동의 상태 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"동의 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/consent/{consent_type}")
async def check_consent(
    consent_type: ConsentType,
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """특정 동의 유형의 현재 상태 확인"""
    try:
        granted = await privacy_manager.check_consent(
            user_id=user_id,
            consent_type=consent_type
        )
        
        return {
            "success": True,
            "consent_type": consent_type.value,
            "granted": granted,
            "message": f"{consent_type.value} 동의 상태: {'동의함' if granted else '동의하지 않음'}"
        }
        
    except Exception as e:
        logger.error("동의 상태 확인 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"동의 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/delete-request", response_model=DeletionRequestResponse)
async def request_data_deletion(
    request: DataDeletionRequestModel,
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """데이터 삭제 요청 (잊혀질 권리)"""
    try:
        request_id = await privacy_manager.request_data_deletion(
            user_id=user_id,
            request_type=request.request_type,
            data_categories=request.data_categories,
            reason=request.reason
        )
        
        if not request_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="데이터 삭제 요청 생성에 실패했습니다."
            )
            
        return DeletionRequestResponse(
            success=True,
            request_id=request_id,
            message="데이터 삭제 요청이 접수되었습니다. 처리 완료까지 시간이 소요될 수 있습니다."
        )
        
    except Exception as e:
        logger.error("데이터 삭제 요청 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터 삭제 요청 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/export", response_model=DataExportResponse)
async def export_user_data(
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """사용자 데이터 내보내기 (데이터 이동권)"""
    try:
        export_result = await privacy_manager.export_user_data(user_id)
        
        if not export_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="데이터 내보내기에 실패했습니다."
            )
            
        return DataExportResponse(
            success=True,
            data=export_result,
            message="사용자 데이터가 성공적으로 내보내졌습니다."
        )
        
    except Exception as e:
        logger.error("데이터 내보내기 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터 내보내기 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/retention-info", response_model=RetentionInfoResponse)
async def get_retention_info(
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """데이터 보존 정책 정보 조회"""
    try:
        retention_info = await privacy_manager.get_retention_info(user_id)
        
        return RetentionInfoResponse(
            success=True,
            data=retention_info,
            message="데이터 보존 정책 정보를 조회했습니다."
        )
        
    except Exception as e:
        logger.error("보존 정책 조회 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"보존 정책 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/data/{category}")
async def forget_data_category(
    category: DataCategory,
    user_id: str = Depends(get_current_user_id),
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """특정 데이터 카테고리 즉시 삭제"""
    try:
        request_id = await privacy_manager.request_data_deletion(
            user_id=user_id,
            request_type="forget",
            data_categories=[category],
            reason=f"Immediate deletion of {category.value}"
        )
        
        if not request_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="데이터 삭제 요청 생성에 실패했습니다."
            )
            
        return {
            "success": True,
            "request_id": request_id,
            "message": f"{category.value} 데이터 삭제 요청이 접수되었습니다."
        }
        
    except Exception as e:
        logger.error("데이터 카테고리 삭제 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/cleanup-expired")
async def cleanup_expired_data(
    privacy_manager: PrivacyPolicyManager = Depends(get_privacy_service)
):
    """만료된 데이터 자동 정리 (관리자용)"""
    try:
        # TODO: 관리자 권한 확인 필요
        
        results = await privacy_manager.cleanup_expired_data()
        
        return {
            "success": True,
            "data": results,
            "message": "만료된 데이터 정리가 완료되었습니다."
        }
        
    except Exception as e:
        logger.error("만료된 데이터 정리 API 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"만료된 데이터 정리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/policy")
async def get_privacy_policy():
    """개인정보 처리방침 조회"""
    try:
        # 정적 개인정보 처리방침 반환
        policy = {
            "version": "1.0",
            "last_updated": "2024-01-01",
            "policy": {
                "data_collection": {
                    "description": "Project Maestro는 서비스 제공을 위해 최소한의 개인정보를 수집합니다.",
                    "categories": [
                        {
                            "name": "대화 데이터",
                            "purpose": "AI 에이전트와의 대화 서비스 제공",
                            "retention": "1년",
                            "legal_basis": "서비스 제공 동의"
                        },
                        {
                            "name": "사용자 식별 정보",
                            "purpose": "계정 관리 및 서비스 제공",
                            "retention": "무기한 (사용자 탈퇴시 삭제)",
                            "legal_basis": "서비스 제공 동의"
                        },
                        {
                            "name": "시스템 로그",
                            "purpose": "보안 및 서비스 개선",
                            "retention": "90일",
                            "legal_basis": "정당한 이익"
                        }
                    ]
                },
                "user_rights": [
                    "개인정보 처리 현황 통지 요구권",
                    "개인정보 처리 정지 요구권",
                    "개인정보 수정·삭제 요구권",
                    "손해배상 청구권",
                    "데이터 이동권"
                ],
                "contact": {
                    "email": "privacy@project-maestro.com",
                    "phone": "02-1234-5678",
                    "address": "서울특별시 강남구 테헤란로 123"
                }
            }
        }
        
        return {
            "success": True,
            "data": policy,
            "message": "개인정보 처리방침을 조회했습니다."
        }
        
    except Exception as e:
        logger.error("개인정보 처리방침 조회 오류", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="개인정보 처리방침 조회 중 오류가 발생했습니다."
        )