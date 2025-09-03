"""Configuration management for Project Maestro."""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://maestro:password@localhost:5432/project_maestro"
    )
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=10)
    
    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")
    
    # AI Services
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    stable_diffusion_api_key: Optional[str] = Field(default=None)
    
    # LangChain
    langchain_tracing_v2: bool = Field(default=False)
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")
    langchain_api_key: Optional[str] = Field(default=None)
    langchain_project: str = Field(default="project-maestro")
    
    # File Storage
    storage_type: str = Field(default="minio")  # minio, s3, local
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_secure: bool = Field(default=False)
    minio_bucket_name: str = Field(default="maestro-assets")
    
    # AWS S3
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_region: str = Field(default="us-west-2")
    s3_bucket_name: str = Field(default="maestro-assets")
    
    # Unity Build Settings
    unity_path: str = Field(
        default="/Applications/Unity/Hub/Editor/2023.2.0f1/Unity.app/Contents/MacOS/Unity"
    )
    unity_project_path: str = Field(default="/tmp/unity_projects")
    unity_build_path: str = Field(default="/tmp/maestro_builds")
    
    # External AI Services
    suno_api_key: Optional[str] = Field(default=None)
    udio_api_key: Optional[str] = Field(default=None)
    replicate_api_token: Optional[str] = Field(default=None)
    
    # Monitoring
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3000)
    
    # Security
    secret_key: str = Field(default="your-super-secret-key-change-in-production")
    access_token_expire_minutes: int = Field(default=30)
    algorithm: str = Field(default="HS256")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)
    
    # Task Timeouts (in seconds)
    task_timeout_default: int = Field(default=300)
    task_timeout_code_generation: int = Field(default=600)
    task_timeout_art_generation: int = Field(default=1200)
    task_timeout_sound_generation: int = Field(default=900)
    task_timeout_level_generation: int = Field(default=400)
    task_timeout_build: int = Field(default=1800)
    
    # Agent Configuration
    max_concurrent_agents: int = Field(default=10)
    agent_retry_attempts: int = Field(default=3)
    agent_retry_delay: float = Field(default=1.0)
    
    # Project Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("/tmp/maestro_data"))
    logs_dir: Path = Field(default_factory=lambda: Path("/tmp/maestro_logs"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("data_dir", "logs_dir", pre=True, always=True)
    def ensure_directories_exist(cls, v):
        """Ensure required directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
        
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
        
    @validator("storage_type")
    def validate_storage_type(cls, v):
        """Validate storage type."""
        allowed = ["minio", "s3", "local"]
        if v not in allowed:
            raise ValueError(f"Storage type must be one of: {allowed}")
        return v
        
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
        
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


# Global settings instance
settings = Settings()

# Ensure required API keys are set for AI services
def validate_ai_services():
    """Validate that required AI service API keys are configured."""
    missing_keys = []
    
    if not settings.openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    if not settings.anthropic_api_key:
        missing_keys.append("ANTHROPIC_API_KEY")
        
    if missing_keys and settings.is_production():
        raise ValueError(
            f"Missing required API keys in production: {', '.join(missing_keys)}"
        )
        
    return len(missing_keys) == 0

# Validate on import if in production
if settings.is_production():
    validate_ai_services()