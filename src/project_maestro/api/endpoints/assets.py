"""Asset management API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends

from ...core.storage import get_asset_manager
from ...core.logging import get_logger
from ..models import BaseResponse, AssetResponse, PaginationParams, PaginatedResponse

router = APIRouter()
logger = get_logger("api.assets")


@router.post("/upload", response_model=AssetResponse)
async def upload_asset(
    project_id: str = Form(...),
    asset_type: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Upload a new asset file."""
    
    try:
        asset_manager = get_asset_manager()
        
        # Save uploaded file temporarily
        import tempfile
        import json
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid metadata JSON format"
                )
        
        # Upload to asset manager
        asset_info = await asset_manager.upload_asset(
            project_id=project_id,
            asset_type=asset_type,
            file_path=tmp_path,
            filename=file.filename,
            metadata=parsed_metadata
        )
        
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
        
        if not asset_info:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload asset"
            )
        
        # Get download URL
        download_url = await asset_manager.get_asset_url(asset_info.id)
        
        return AssetResponse(
            id=asset_info.id,
            project_id=asset_info.project_id,
            filename=asset_info.filename,
            asset_type=asset_info.asset_type,
            mime_type=asset_info.mime_type,
            file_size=asset_info.file_size,
            created_at=asset_info.created_at,
            metadata=asset_info.metadata,
            download_url=download_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload asset", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload asset: {str(e)}"
        )


@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(asset_id: str):
    """Get asset information."""
    
    try:
        asset_manager = get_asset_manager()
        asset_info = asset_manager.get_asset_info(asset_id)
        
        if not asset_info:
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{asset_id}' not found"
            )
        
        # Get download URL
        download_url = await asset_manager.get_asset_url(asset_id)
        
        return AssetResponse(
            id=asset_info.id,
            project_id=asset_info.project_id,
            filename=asset_info.filename,
            asset_type=asset_info.asset_type,
            mime_type=asset_info.mime_type,
            file_size=asset_info.file_size,
            created_at=asset_info.created_at,
            metadata=asset_info.metadata,
            download_url=download_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get asset", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get asset: {str(e)}"
        )


@router.get("/", response_model=PaginatedResponse)
async def list_assets(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    pagination: PaginationParams = Depends()
):
    """List assets with pagination and filtering."""
    
    try:
        asset_manager = get_asset_manager()
        
        # Get assets
        if project_id:
            assets = asset_manager.list_project_assets(
                project_id=project_id,
                asset_type=asset_type,
                limit=pagination.size,
                offset=(pagination.page - 1) * pagination.size
            )
        else:
            # For demo purposes, return empty list if no project_id
            assets = []
        
        # Convert to response models
        asset_responses = []
        for asset in assets:
            download_url = await asset_manager.get_asset_url(asset.id)
            asset_responses.append(AssetResponse(
                id=asset.id,
                project_id=asset.project_id,
                filename=asset.filename,
                asset_type=asset.asset_type,
                mime_type=asset.mime_type,
                file_size=asset.file_size,
                created_at=asset.created_at,
                metadata=asset.metadata,
                download_url=download_url
            ))
        
        # Mock pagination data
        total = len(asset_responses)
        pages = (total + pagination.size - 1) // pagination.size if total > 0 else 1
        
        return PaginatedResponse(
            items=asset_responses,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages,
            has_next=pagination.page < pages,
            has_prev=pagination.page > 1
        )
        
    except Exception as e:
        logger.error("Failed to list assets", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list assets: {str(e)}"
        )


@router.delete("/{asset_id}", response_model=BaseResponse)
async def delete_asset(asset_id: str):
    """Delete an asset."""
    
    try:
        asset_manager = get_asset_manager()
        success = await asset_manager.delete_asset(asset_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{asset_id}' not found"
            )
        
        return BaseResponse(
            success=True,
            message=f"Asset '{asset_id}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete asset", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete asset: {str(e)}"
        )


@router.get("/{asset_id}/download")
async def download_asset(asset_id: str):
    """Get download URL for an asset."""
    
    try:
        asset_manager = get_asset_manager()
        asset_info = asset_manager.get_asset_info(asset_id)
        
        if not asset_info:
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{asset_id}' not found"
            )
        
        download_url = await asset_manager.get_asset_url(asset_id, expiry_seconds=3600)
        
        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate download URL"
            )
        
        return {"download_url": download_url, "expires_in": 3600}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get download URL", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get download URL: {str(e)}"
        )


@router.get("/search/", response_model=List[AssetResponse])
async def search_assets(
    query: str = Query(..., description="Search query"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    limit: int = Query(50, description="Maximum results to return")
):
    """Search assets by filename or metadata."""
    
    try:
        asset_manager = get_asset_manager()
        assets = asset_manager.search_assets(
            query=query,
            project_id=project_id,
            asset_type=asset_type,
            limit=limit
        )
        
        # Convert to response models
        asset_responses = []
        for asset in assets:
            download_url = await asset_manager.get_asset_url(asset.id)
            asset_responses.append(AssetResponse(
                id=asset.id,
                project_id=asset.project_id,
                filename=asset.filename,
                asset_type=asset.asset_type,
                mime_type=asset.mime_type,
                file_size=asset.file_size,
                created_at=asset.created_at,
                metadata=asset.metadata,
                download_url=download_url
            ))
        
        return asset_responses
        
    except Exception as e:
        logger.error("Failed to search assets", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search assets: {str(e)}"
        )


@router.get("/types/", response_model=List[str])
async def list_asset_types():
    """List all available asset types."""
    return [
        "code",
        "sprites", 
        "character_sprites",
        "background",
        "ui_element",
        "audio",
        "bgm",
        "sfx",
        "levels",
        "prefabs",
        "materials",
        "animations"
    ]