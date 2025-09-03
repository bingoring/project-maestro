"""Asset storage and management system for Project Maestro."""

import asyncio
import hashlib
import mimetypes
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO, Union

import boto3
from botocore.exceptions import ClientError
from minio import Minio
from minio.error import S3Error
import aiofiles
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field

from .config import settings
from .logging import get_logger


Base = declarative_base()


class AssetMetadata(Base):
    """Database model for asset metadata."""
    __tablename__ = "assets"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    asset_type = Column(String, nullable=False, index=True)
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String, nullable=False, unique=True, index=True)
    storage_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(Text)  # JSON string for additional metadata
    is_active = Column(Boolean, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "filename": self.filename,
            "asset_type": self.asset_type,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "storage_path": self.storage_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata_json,
            "is_active": self.is_active
        }


class AssetInfo(BaseModel):
    """Asset information model."""
    id: str
    project_id: str
    filename: str
    asset_type: str
    mime_type: str
    file_size: int
    file_hash: str
    storage_path: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def upload_file(
        self,
        file_path: Union[str, Path],
        storage_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload a file to storage."""
        pass
        
    @abstractmethod
    async def download_file(
        self,
        storage_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Download a file from storage."""
        pass
        
    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage."""
        pass
        
    @abstractmethod
    async def file_exists(self, storage_path: str) -> bool:
        """Check if a file exists in storage."""
        pass
        
    @abstractmethod
    async def get_file_url(
        self,
        storage_path: str,
        expiry_seconds: int = 3600
    ) -> Optional[str]:
        """Get a temporary URL for file access."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("local_storage")
        
    async def upload_file(
        self,
        file_path: Union[str, Path],
        storage_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload file to local storage."""
        try:
            source = Path(file_path)
            destination = self.base_path / storage_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(source, 'rb') as src:
                async with aiofiles.open(destination, 'wb') as dst:
                    await dst.write(await src.read())
                    
            self.logger.info(
                "File uploaded to local storage",
                storage_path=storage_path,
                file_size=destination.stat().st_size
            )
            return True
            
        except Exception as e:
            self.logger.error("Failed to upload to local storage", error=str(e))
            return False
            
    async def download_file(
        self,
        storage_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Download file from local storage."""
        try:
            source = self.base_path / storage_path
            destination = Path(local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            if not source.exists():
                return False
                
            async with aiofiles.open(source, 'rb') as src:
                async with aiofiles.open(destination, 'wb') as dst:
                    await dst.write(await src.read())
                    
            return True
            
        except Exception as e:
            self.logger.error("Failed to download from local storage", error=str(e))
            return False
            
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from local storage."""
        try:
            file_path = self.base_path / storage_path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            self.logger.error("Failed to delete from local storage", error=str(e))
            return False
            
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in local storage."""
        return (self.base_path / storage_path).exists()
        
    async def get_file_url(
        self,
        storage_path: str,
        expiry_seconds: int = 3600
    ) -> Optional[str]:
        """Get file URL (local file path)."""
        file_path = self.base_path / storage_path
        return f"file://{file_path.absolute()}" if file_path.exists() else None


class MinIOBackend(StorageBackend):
    """MinIO/S3-compatible storage backend."""
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = True
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.logger = get_logger("minio_storage")
        
        # Ensure bucket exists
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Created bucket: {bucket_name}")
        except S3Error as e:
            self.logger.error(f"Failed to create bucket: {e}")
            
    async def upload_file(
        self,
        file_path: Union[str, Path],
        storage_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload file to MinIO."""
        try:
            file_path = Path(file_path)
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            
            # Run in thread pool since minio client is synchronous
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.fput_object(
                    self.bucket_name,
                    storage_path,
                    str(file_path),
                    content_type=content_type,
                    metadata=metadata or {}
                )
            )
            
            self.logger.info(
                "File uploaded to MinIO",
                storage_path=storage_path,
                bucket=self.bucket_name
            )
            return True
            
        except S3Error as e:
            self.logger.error("Failed to upload to MinIO", error=str(e))
            return False
            
    async def download_file(
        self,
        storage_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Download file from MinIO."""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.fget_object(
                    self.bucket_name,
                    storage_path,
                    str(local_path)
                )
            )
            
            return True
            
        except S3Error as e:
            self.logger.error("Failed to download from MinIO", error=str(e))
            return False
            
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from MinIO."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.remove_object(self.bucket_name, storage_path)
            )
            return True
            
        except S3Error as e:
            self.logger.error("Failed to delete from MinIO", error=str(e))
            return False
            
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in MinIO."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.stat_object(self.bucket_name, storage_path)
            )
            return True
        except S3Error:
            return False
            
    async def get_file_url(
        self,
        storage_path: str,
        expiry_seconds: int = 3600
    ) -> Optional[str]:
        """Get presigned URL for file access."""
        try:
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.presigned_get_object(
                    self.bucket_name,
                    storage_path,
                    expires=timedelta(seconds=expiry_seconds)
                )
            )
            return url
        except S3Error:
            return None


class S3Backend(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(
        self,
        access_key: str,
        secret_key: str,
        region: str,
        bucket_name: str
    ):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        self.bucket_name = bucket_name
        self.logger = get_logger("s3_storage")
        
        # Ensure bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                    if region != 'us-east-1' else {}
                )
                self.logger.info(f"Created S3 bucket: {bucket_name}")
                
    async def upload_file(
        self,
        file_path: Union[str, Path],
        storage_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload file to S3."""
        try:
            file_path = Path(file_path)
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
                
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    storage_path,
                    ExtraArgs=extra_args
                )
            )
            
            self.logger.info(
                "File uploaded to S3",
                storage_path=storage_path,
                bucket=self.bucket_name
            )
            return True
            
        except ClientError as e:
            self.logger.error("Failed to upload to S3", error=str(e))
            return False
            
    async def download_file(
        self,
        storage_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Download file from S3."""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.download_file(
                    self.bucket_name,
                    storage_path,
                    str(local_path)
                )
            )
            
            return True
            
        except ClientError as e:
            self.logger.error("Failed to download from S3", error=str(e))
            return False
            
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=storage_path
                )
            )
            return True
            
        except ClientError as e:
            self.logger.error("Failed to delete from S3", error=str(e))
            return False
            
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=storage_path
                )
            )
            return True
        except ClientError:
            return False
            
    async def get_file_url(
        self,
        storage_path: str,
        expiry_seconds: int = 3600
    ) -> Optional[str]:
        """Generate presigned URL for file access."""
        try:
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': storage_path},
                    ExpiresIn=expiry_seconds
                )
            )
            return url
        except ClientError:
            return None


class AssetManager:
    """Central asset management system."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
        self.logger = get_logger("asset_manager")
        
        # Database setup
        engine = create_engine(settings.database_url)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)
        
    def _get_db_session(self):
        """Get database session."""
        return self.SessionLocal()
        
    def _calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
        
    def _generate_storage_path(
        self,
        project_id: str,
        asset_type: str,
        filename: str
    ) -> str:
        """Generate storage path for an asset."""
        return f"projects/{project_id}/{asset_type}/{filename}"
        
    async def upload_asset(
        self,
        project_id: str,
        asset_type: str,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AssetInfo]:
        """Upload an asset and store its metadata."""
        
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error("File not found", file_path=str(file_path))
            return None
            
        if not filename:
            filename = file_path.name
            
        try:
            # Calculate file properties
            file_size = file_path.stat().st_size
            file_hash = self._calculate_file_hash(file_path)
            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            
            # Check for existing asset with same hash
            with self._get_db_session() as db:
                existing = db.query(AssetMetadata).filter(
                    AssetMetadata.file_hash == file_hash,
                    AssetMetadata.project_id == project_id,
                    AssetMetadata.is_active == True
                ).first()
                
                if existing:
                    self.logger.info(
                        "Asset with same hash already exists",
                        asset_id=existing.id,
                        file_hash=file_hash
                    )
                    return AssetInfo(**existing.to_dict())
                    
            # Generate storage path and upload
            storage_path = self._generate_storage_path(project_id, asset_type, filename)
            
            success = await self.storage_backend.upload_file(
                file_path,
                storage_path,
                metadata
            )
            
            if not success:
                self.logger.error("Failed to upload asset to storage")
                return None
                
            # Create database record
            import uuid
            asset_id = str(uuid.uuid4())
            
            asset_metadata = AssetMetadata(
                id=asset_id,
                project_id=project_id,
                filename=filename,
                asset_type=asset_type,
                mime_type=mime_type,
                file_size=file_size,
                file_hash=file_hash,
                storage_path=storage_path,
                metadata_json=json.dumps(metadata) if metadata else None
            )
            
            with self._get_db_session() as db:
                db.add(asset_metadata)
                db.commit()
                db.refresh(asset_metadata)
                
            self.logger.info(
                "Asset uploaded successfully",
                asset_id=asset_id,
                project_id=project_id,
                filename=filename,
                file_size=file_size
            )
            
            return AssetInfo(**asset_metadata.to_dict())
            
        except Exception as e:
            self.logger.error("Failed to upload asset", error=str(e))
            return None
            
    async def download_asset(
        self,
        asset_id: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Download an asset to local path."""
        
        with self._get_db_session() as db:
            asset = db.query(AssetMetadata).filter(
                AssetMetadata.id == asset_id,
                AssetMetadata.is_active == True
            ).first()
            
            if not asset:
                self.logger.error("Asset not found", asset_id=asset_id)
                return False
                
            success = await self.storage_backend.download_file(
                asset.storage_path,
                local_path
            )
            
            if success:
                self.logger.info(
                    "Asset downloaded successfully",
                    asset_id=asset_id,
                    local_path=str(local_path)
                )
                
            return success
            
    async def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset (soft delete)."""
        
        with self._get_db_session() as db:
            asset = db.query(AssetMetadata).filter(
                AssetMetadata.id == asset_id
            ).first()
            
            if not asset:
                return False
                
            # Soft delete in database
            asset.is_active = False
            asset.updated_at = datetime.utcnow()
            db.commit()
            
            # Delete from storage (optional)
            await self.storage_backend.delete_file(asset.storage_path)
            
            self.logger.info("Asset deleted", asset_id=asset_id)
            return True
            
    def get_asset_info(self, asset_id: str) -> Optional[AssetInfo]:
        """Get asset information."""
        
        with self._get_db_session() as db:
            asset = db.query(AssetMetadata).filter(
                AssetMetadata.id == asset_id,
                AssetMetadata.is_active == True
            ).first()
            
            if asset:
                return AssetInfo(**asset.to_dict())
            return None
            
    def list_project_assets(
        self,
        project_id: str,
        asset_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AssetInfo]:
        """List assets for a project."""
        
        with self._get_db_session() as db:
            query = db.query(AssetMetadata).filter(
                AssetMetadata.project_id == project_id,
                AssetMetadata.is_active == True
            )
            
            if asset_type:
                query = query.filter(AssetMetadata.asset_type == asset_type)
                
            assets = query.order_by(AssetMetadata.created_at.desc())\
                         .offset(offset)\
                         .limit(limit)\
                         .all()
                         
            return [AssetInfo(**asset.to_dict()) for asset in assets]
            
    async def get_asset_url(
        self,
        asset_id: str,
        expiry_seconds: int = 3600
    ) -> Optional[str]:
        """Get temporary URL for asset access."""
        
        with self._get_db_session() as db:
            asset = db.query(AssetMetadata).filter(
                AssetMetadata.id == asset_id,
                AssetMetadata.is_active == True
            ).first()
            
            if not asset:
                return None
                
            return await self.storage_backend.get_file_url(
                asset.storage_path,
                expiry_seconds
            )
            
    def search_assets(
        self,
        query: str,
        project_id: Optional[str] = None,
        asset_type: Optional[str] = None,
        limit: int = 50
    ) -> List[AssetInfo]:
        """Search assets by filename or metadata."""
        
        with self._get_db_session() as db:
            db_query = db.query(AssetMetadata).filter(
                AssetMetadata.is_active == True
            )
            
            if project_id:
                db_query = db_query.filter(AssetMetadata.project_id == project_id)
                
            if asset_type:
                db_query = db_query.filter(AssetMetadata.asset_type == asset_type)
                
            # Simple text search on filename
            db_query = db_query.filter(
                AssetMetadata.filename.ilike(f"%{query}%")
            )
            
            assets = db_query.order_by(AssetMetadata.created_at.desc())\
                            .limit(limit)\
                            .all()
                            
            return [AssetInfo(**asset.to_dict()) for asset in assets]


# Global asset manager instance
_asset_manager: Optional[AssetManager] = None


def get_asset_manager() -> AssetManager:
    """Get the global asset manager instance."""
    global _asset_manager
    if _asset_manager is None:
        # Create storage backend based on configuration
        if settings.storage_type == "minio":
            backend = MinIOBackend(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name=settings.minio_bucket_name,
                secure=settings.minio_secure
            )
        elif settings.storage_type == "s3":
            backend = S3Backend(
                access_key=settings.aws_access_key_id,
                secret_key=settings.aws_secret_access_key,
                region=settings.aws_region,
                bucket_name=settings.s3_bucket_name
            )
        else:
            # Default to local storage
            backend = LocalStorageBackend(settings.data_dir / "storage")
            
        _asset_manager = AssetManager(backend)
        
    return _asset_manager