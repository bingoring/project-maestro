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
    
    # ====== Enterprise Knowledge Management ======
    
    # Enterprise Systems Integration
    jira_enabled: bool = Field(default=False)
    jira_base_url: Optional[str] = Field(default=None)
    jira_username: Optional[str] = Field(default=None)
    jira_api_token: Optional[str] = Field(default=None)
    jira_project_keys: List[str] = Field(default_factory=list)
    
    slack_enabled: bool = Field(default=False)
    slack_bot_token: Optional[str] = Field(default=None)
    slack_app_token: Optional[str] = Field(default=None)
    slack_channels: List[str] = Field(default_factory=list)
    slack_max_history_days: int = Field(default=30)
    
    confluence_enabled: bool = Field(default=False)
    confluence_base_url: Optional[str] = Field(default=None)
    confluence_username: Optional[str] = Field(default=None)
    confluence_api_token: Optional[str] = Field(default=None)
    confluence_space_keys: List[str] = Field(default_factory=list)
    
    # RAG System Configuration
    rag_enabled: bool = Field(default=True)
    rag_vector_store_type: str = Field(default="chroma")  # chroma, pinecone, weaviate
    rag_embedding_model: str = Field(default="text-embedding-ada-002")
    rag_chunk_size: int = Field(default=1000)
    rag_chunk_overlap: int = Field(default=200)
    rag_max_results: int = Field(default=10)
    rag_similarity_threshold: float = Field(default=0.7)
    
    # Vector Database Settings
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection_name: str = Field(default="maestro_enterprise")
    
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_environment: str = Field(default="us-west1-gcp")
    pinecone_index_name: str = Field(default="maestro-enterprise")
    
    # Query Agent Configuration
    query_agent_enabled: bool = Field(default=True)
    query_agent_model: str = Field(default="gpt-4-turbo-preview")
    query_agent_temperature: float = Field(default=0.1)
    query_agent_max_tokens: int = Field(default=1000)
    query_cascading_enabled: bool = Field(default=True)
    query_complexity_threshold: float = Field(default=0.6)
    
    # Intent Analysis Configuration
    intent_analysis_enabled: bool = Field(default=True)
    intent_analysis_model: str = Field(default="gpt-3.5-turbo")
    intent_confidence_threshold: float = Field(default=0.8)
    intent_fallback_to_general: bool = Field(default=True)
    
    # Enterprise Data Sync Settings
    enterprise_sync_interval_hours: int = Field(default=24)
    enterprise_sync_batch_size: int = Field(default=100)
    enterprise_data_retention_days: int = Field(default=90)
    enterprise_sync_enabled: bool = Field(default=True)
    
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
        
    @validator("rag_vector_store_type")
    def validate_vector_store_type(cls, v):
        """Validate vector store type."""
        allowed = ["chroma", "pinecone", "weaviate"]
        if v not in allowed:
            raise ValueError(f"RAG vector store type must be one of: {allowed}")
        return v
        
    @validator("jira_base_url", "confluence_base_url", pre=True, always=True)
    def validate_enterprise_urls(cls, v):
        """Validate enterprise system URLs."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Enterprise system URLs must start with http:// or https://")
        return v
        
    @validator("rag_similarity_threshold", "query_complexity_threshold", "intent_confidence_threshold")
    def validate_threshold_values(cls, v):
        """Validate threshold values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold values must be between 0 and 1")
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

def validate_enterprise_services():
    """Validate enterprise system configurations."""
    issues = []
    
    # Jira validation
    if settings.jira_enabled:
        if not settings.jira_base_url:
            issues.append("JIRA_BASE_URL is required when Jira is enabled")
        if not settings.jira_username:
            issues.append("JIRA_USERNAME is required when Jira is enabled")
        if not settings.jira_api_token:
            issues.append("JIRA_API_TOKEN is required when Jira is enabled")
    
    # Slack validation
    if settings.slack_enabled:
        if not settings.slack_bot_token:
            issues.append("SLACK_BOT_TOKEN is required when Slack is enabled")
    
    # Confluence validation
    if settings.confluence_enabled:
        if not settings.confluence_base_url:
            issues.append("CONFLUENCE_BASE_URL is required when Confluence is enabled")
        if not settings.confluence_username:
            issues.append("CONFLUENCE_USERNAME is required when Confluence is enabled")
        if not settings.confluence_api_token:
            issues.append("CONFLUENCE_API_TOKEN is required when Confluence is enabled")
    
    # Vector database validation
    if settings.rag_enabled:
        if settings.rag_vector_store_type == "pinecone" and not settings.pinecone_api_key:
            issues.append("PINECONE_API_KEY is required when using Pinecone vector store")
    
    if issues and settings.is_production():
        raise ValueError(f"Enterprise configuration issues: {'; '.join(issues)}")
        
    return len(issues) == 0


def get_enterprise_config():
    """Get enterprise system configuration for easy access."""
    return {
        "jira": {
            "enabled": settings.jira_enabled,
            "base_url": settings.jira_base_url,
            "username": settings.jira_username,
            "api_token": settings.jira_api_token,
            "project_keys": settings.jira_project_keys,
        },
        "slack": {
            "enabled": settings.slack_enabled,
            "bot_token": settings.slack_bot_token,
            "app_token": settings.slack_app_token,
            "channels": settings.slack_channels,
            "max_history_days": settings.slack_max_history_days,
        },
        "confluence": {
            "enabled": settings.confluence_enabled,
            "base_url": settings.confluence_base_url,
            "username": settings.confluence_username,
            "api_token": settings.confluence_api_token,
            "space_keys": settings.confluence_space_keys,
        },
        "rag": {
            "enabled": settings.rag_enabled,
            "vector_store_type": settings.rag_vector_store_type,
            "embedding_model": settings.rag_embedding_model,
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
            "max_results": settings.rag_max_results,
            "similarity_threshold": settings.rag_similarity_threshold,
        },
        "query_agent": {
            "enabled": settings.query_agent_enabled,
            "model": settings.query_agent_model,
            "temperature": settings.query_agent_temperature,
            "max_tokens": settings.query_agent_max_tokens,
            "cascading_enabled": settings.query_cascading_enabled,
            "complexity_threshold": settings.query_complexity_threshold,
        },
    }


# Validate on import if in production
if settings.is_production():
    validate_ai_services()
    validate_enterprise_services()