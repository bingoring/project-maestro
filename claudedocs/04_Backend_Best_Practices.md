# Python 기초부터 마스터까지 - 4단계: 백엔드 개발 베스트 프랙티스

## 목차
1. [테스트 주도 개발 (TDD)](#테스트-주도-개발-tdd)
2. [코드 품질과 정적 분석](#코드-품질과-정적-분석)
3. [보안 베스트 프랙티스](#보안-베스트-프랙티스)
4. [성능 최적화 기법](#성능-최적화-기법)
5. [CI/CD 파이프라인](#cicd-파이프라인)
6. [프로덕션 운영 전략](#프로덕션-운영-전략)

---

## 테스트 주도 개발 (TDD)

### Pytest 기반 테스트 구조

```python
# tests/conftest.py - 테스트 설정 및 픽스처
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# 테스트 환경 설정
os.environ["MAESTRO_ENVIRONMENT"] = "test"
os.environ["MAESTRO_DEBUG"] = "true"

class MockLLM(BaseLanguageModel):
    """테스트용 모의 LLM"""
    def __init__(self, responses: list = None):
        self.responses = responses or ["Test response"]
        self.call_count = 0

    async def _agenerate(self, messages, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)

@pytest.fixture
async def test_db():
    """테스트 데이터베이스 픽스처"""
    # 테스트 DB 생성
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    # 정리
    await test_engine.dispose()

@pytest.fixture
def mock_agent_registry():
    """모의 에이전트 레지스트리"""
    registry = MagicMock()
    registry.get_agents_by_type.return_value = [MagicMock()]
    return registry
```

### 단위 테스트 예시

```python
# tests/test_agent_framework.py
import pytest
from unittest.mock import AsyncMock, patch

class TestOrchestratorAgent:
    """오케스트레이터 에이전트 테스트"""

    @pytest.mark.asyncio
    async def test_task_execution(self, mock_llm):
        """태스크 실행 테스트"""
        # Given
        agent = OrchestratorAgent()
        agent.llm = mock_llm

        task = AgentTask(
            id="test-task-001",
            type="project_analysis",
            description="Analyze game design document",
            parameters={"gdd_content": "Test game design"}
        )

        # When
        result = await agent.execute_task(task)

        # Then
        assert result is not None
        assert "analysis" in result
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_workflow_creation(self, mock_llm):
        """워크플로우 생성 테스트"""
        # Given
        agent = OrchestratorAgent()
        agent.llm = mock_llm

        project_spec = ProjectSpec(
            id="test-project",
            title="Test Game",
            description="Test description",
            genre="action",
            art_style="pixel",
            gameplay_mechanics=["jump", "shoot"]
        )

        # When
        workflow = await agent.create_workflow(project_spec)

        # Then
        assert len(workflow.steps) > 0
        assert workflow.project_id == "test-project"
        assert any(step.agent_type == AgentType.CANVAS for step in workflow.steps)

    def test_capabilities_definition(self):
        """에이전트 능력 정의 테스트"""
        # Given
        agent = OrchestratorAgent()

        # When
        capabilities = agent.get_capabilities()

        # Then
        expected_capabilities = [
            "project_analysis",
            "workflow_creation",
            "task_assignment",
            "progress_tracking"
        ]
        assert all(cap in capabilities for cap in expected_capabilities)
```

### 통합 테스트

```python
# tests/integration/test_project_workflow.py
import pytest
from httpx import AsyncClient

class TestProjectWorkflowIntegration:
    """프로젝트 워크플로우 통합 테스트"""

    @pytest.mark.asyncio
    async def test_complete_project_creation_flow(self, test_client, test_db):
        """전체 프로젝트 생성 플로우 테스트"""
        # Given
        project_data = {
            "title": "Integration Test Game",
            "description": "Test game for integration testing",
            "game_design_document": "A simple platformer game with jumping mechanics..."
        }

        # When - 프로젝트 생성 요청
        response = await test_client.post("/api/v1/projects/", json=project_data)

        # Then - 프로젝트 생성 확인
        assert response.status_code == 201
        project = response.json()
        assert project["title"] == project_data["title"]
        assert project["status"] == "processing"

        project_id = project["id"]

        # When - 프로젝트 상태 확인 (비동기 처리 완료 대기)
        import asyncio
        await asyncio.sleep(2)  # 실제로는 polling이나 webhook 사용

        response = await test_client.get(f"/api/v1/projects/{project_id}")

        # Then - 처리 완료 확인
        assert response.status_code == 200
        updated_project = response.json()
        assert updated_project["status"] in ["completed", "processing"]

    @pytest.mark.asyncio
    async def test_agent_collaboration(self, test_client, mock_agent_registry):
        """에이전트 협업 테스트"""
        with patch('project_maestro.core.agent_framework.agent_registry', mock_agent_registry):
            # Given
            project_data = {"title": "Agent Test", "game_design_document": "Test GDD"}

            # When
            response = await test_client.post("/api/v1/projects/", json=project_data)

            # Then
            assert response.status_code == 201
            # 에이전트 호출 확인
            mock_agent_registry.get_agents_by_type.assert_called()
```

### 테스트 커버리지와 리포팅

```python
# pytest.ini 설정
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = [
    "--strict-markers",
    "--cov=src/project_maestro",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80"  # 80% 이상 커버리지 요구
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow running tests",
]

# 테스트 실행 명령어
# pytest                          # 모든 테스트
# pytest -m unit                  # 단위 테스트만
# pytest -m "not slow"           # 느린 테스트 제외
# pytest --cov-report=html       # HTML 커버리지 리포트
```

---

## 코드 품질과 정적 분석

### Black + isort + mypy 설정

```toml
# pyproject.toml 코드 품질 도구 설정

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.mypy_cache
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["project_maestro"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
```

### 타입 힌트 활용

```python
# 완전한 타입 힌트 예시
from typing import TypeVar, Generic, Protocol, Union, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

T = TypeVar('T')
U = TypeVar('U', bound='BaseAgent')

class Repository(Protocol, Generic[T]):
    """리포지토리 프로토콜"""
    async def create(self, entity: T) -> T: ...
    async def get_by_id(self, id: str) -> T | None: ...
    async def update(self, entity: T) -> T: ...
    async def delete(self, id: str) -> bool: ...

@dataclass
class Result(Generic[T]):
    """결과 래퍼 클래스"""
    success: bool
    data: T | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        return cls(success=True, data=data)

    @classmethod
    def error(cls, message: str) -> 'Result[T]':
        return cls(success=False, error=message)

# 리터럴 타입 활용
ProjectStatus = Literal["draft", "processing", "completed", "failed"]

class ProjectService:
    """타입 안전한 프로젝트 서비스"""

    def __init__(self, repository: Repository[Project]):
        self.repository = repository

    async def create_project(
        self,
        title: str,
        gdd_content: str,
        target_platform: Literal["mobile", "web", "desktop"] = "mobile"
    ) -> Result[Project]:
        """프로젝트 생성"""
        try:
            project = Project(
                id=str(uuid.uuid4()),
                title=title,
                gdd_content=gdd_content,
                target_platform=target_platform,
                status="draft"
            )

            created_project = await self.repository.create(project)
            return Result.ok(created_project)

        except Exception as e:
            return Result.error(f"프로젝트 생성 실패: {e}")

    async def update_status(
        self,
        project_id: str,
        status: ProjectStatus
    ) -> Result[Project]:
        """프로젝트 상태 업데이트"""
        project = await self.repository.get_by_id(project_id)
        if not project:
            return Result.error("프로젝트를 찾을 수 없습니다")

        project.status = status
        updated_project = await self.repository.update(project)
        return Result.ok(updated_project)
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
```

---

## 보안 베스트 프랙티스

### 인증 및 권한 관리

```python
# 보안 설정
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    """보안 관리자"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        """액세스 토큰 생성"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> dict | None:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

# FastAPI 의존성
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> User:
    """현재 사용자 조회"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = security_manager.verify_token(token.credentials)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = await user_repository.get_by_id(user_id)
    if user is None:
        raise credentials_exception

    return user

# 보호된 엔드포인트
@router.get("/protected")
async def protected_endpoint(current_user: User = Depends(get_current_user)):
    """보호된 엔드포인트"""
    return {"message": f"Hello {current_user.username}"}
```

### 입력 검증 및 SQL 인젝션 방지

```python
from pydantic import BaseModel, validator, Field
from sqlalchemy import text
import re

class ProjectCreateRequest(BaseModel):
    """안전한 프로젝트 생성 요청"""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    gdd_content: str = Field(..., min_length=100)

    @validator('title')
    def validate_title(cls, v):
        # XSS 방지
        if re.search(r'[<>"\']', v):
            raise ValueError('제목에 특수문자가 포함될 수 없습니다')
        return v.strip()

    @validator('gdd_content')
    def validate_gdd_content(cls, v):
        # 악성 스크립트 탐지
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'onclick=',
            r'onerror='
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('허용되지 않는 내용이 포함되었습니다')
        return v

# 파라미터화된 쿼리 (SQL 인젝션 방지)
class ProjectRepository:
    """안전한 리포지토리"""

    async def search_projects(self, search_term: str) -> List[Project]:
        """안전한 프로젝트 검색"""
        # 파라미터 바인딩 사용 (SQL 인젝션 방지)
        query = text("""
            SELECT * FROM projects
            WHERE title ILIKE :search_term
            OR description ILIKE :search_term
        """)

        result = await self.session.execute(
            query,
            {"search_term": f"%{search_term}%"}
        )
        return result.fetchall()

    async def get_user_projects(self, user_id: str) -> List[Project]:
        """사용자별 프로젝트 조회 (권한 확인)"""
        # SQLAlchemy ORM 사용 (안전함)
        query = select(Project).where(
            Project.user_id == user_id
        )
        result = await self.session.execute(query)
        return result.scalars().all()
```

### 시크릿 관리

```python
# 환경변수 및 시크릿 관리
from pydantic_settings import BaseSettings
import secrets

class Settings(BaseSettings):
    """보안이 고려된 설정"""

    # 시크릿 키 (환경변수에서만 읽기)
    secret_key: str = Field(..., env="SECRET_KEY")
    database_url: str = Field(..., env="DATABASE_URL")

    # API 키들 (옵셔널, 환경변수 우선)
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, env="ANTHROPIC_API_KEY")

    # 개발환경 기본값, 프로덕션에서는 환경변수 필수
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('SECRET_KEY는 최소 32자 이상이어야 합니다')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 시크릿 키 생성 유틸리티
def generate_secret_key():
    """안전한 시크릿 키 생성"""
    return secrets.token_urlsafe(32)

# .env 파일 (절대 Git에 커밋하지 않음)
# SECRET_KEY=your-super-secret-key-here-32-chars-min
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
# OPENAI_API_KEY=sk-...
```

---

## 성능 최적화 기법

### 데이터베이스 최적화

```python
# 인덱스 및 쿼리 최적화
from sqlalchemy import Index, func
from sqlalchemy.orm import selectinload, joinedload

class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID, primary_key=True)
    title = Column(String(200), nullable=False)
    status = Column(String(50), default="draft")
    user_id = Column(UUID, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 복합 인덱스 정의
    __table_args__ = (
        Index('idx_user_status', 'user_id', 'status'),
        Index('idx_created_at', 'created_at'),
    )

class OptimizedProjectRepository:
    """최적화된 프로젝트 리포지토리"""

    async def get_user_projects_with_assets(self, user_id: str) -> List[Project]:
        """관련 자산과 함께 프로젝트 조회 (N+1 문제 해결)"""
        query = select(Project).options(
            selectinload(Project.assets),  # 별도 쿼리로 로드
            joinedload(Project.user)       # JOIN으로 로드
        ).where(Project.user_id == user_id)

        result = await self.session.execute(query)
        return result.unique().scalars().all()

    async def get_project_statistics(self) -> Dict[str, int]:
        """집계 쿼리 최적화"""
        query = select(
            func.count(Project.id).label('total'),
            func.count(Project.id).filter(Project.status == 'completed').label('completed'),
            func.count(Project.id).filter(Project.status == 'processing').label('processing')
        )

        result = await self.session.execute(query)
        row = result.first()

        return {
            'total': row.total,
            'completed': row.completed,
            'processing': row.processing
        }

# 연결 풀 최적화
engine = create_async_engine(
    settings.database_url,
    pool_size=20,           # 기본 연결 수
    max_overflow=30,        # 추가 연결 수
    pool_pre_ping=True,     # 연결 상태 확인
    pool_recycle=3600,      # 1시간마다 연결 재생성
    echo=False              # 프로덕션에서는 False
)
```

### 캐싱 전략

```python
# 다층 캐싱 시스템
from functools import wraps
import asyncio

def cache_with_ttl(ttl: int = 300, cache_layer: str = "redis"):
    """TTL 기반 캐싱 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 캐시에서 조회
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 캐시 미스 시 실행
            result = await func(*args, **kwargs)

            # 캐시에 저장
            await cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator

class SmartCache:
    """지능형 캐싱 시스템"""

    def __init__(self):
        self.memory_cache = {}  # 메모리 캐시
        self.redis_cache = RedisCache()  # Redis 캐시

    @cache_with_ttl(ttl=600)
    async def get_project_analysis(self, project_id: str) -> Dict:
        """프로젝트 분석 결과 캐싱"""
        # 실제 분석 로직 (무거운 작업)
        project = await project_repository.get_by_id(project_id)
        analysis = await complex_analysis_service.analyze(project)
        return analysis

    async def invalidate_project_cache(self, project_id: str):
        """프로젝트 관련 캐시 무효화"""
        patterns = [
            f"get_project_analysis:*{project_id}*",
            f"project_details:*{project_id}*",
            f"project_assets:*{project_id}*"
        ]

        for pattern in patterns:
            await self.redis_cache.delete_pattern(pattern)
```

### 비동기 처리 최적화

```python
# 배치 처리 및 병렬 실행
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """배치 처리 최적화"""

    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_projects_batch(self, project_ids: List[str]) -> List[Dict]:
        """프로젝트 배치 처리"""
        # 배치 단위로 분할
        batches = [
            project_ids[i:i + self.batch_size]
            for i in range(0, len(project_ids), self.batch_size)
        ]

        # 병렬 처리
        tasks = [
            self._process_single_batch(batch)
            for batch in batches
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 병합
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"배치 처리 실패: {result}")
                continue
            all_results.extend(result)

        return all_results

    async def _process_single_batch(self, project_ids: List[str]) -> List[Dict]:
        """단일 배치 처리"""
        # CPU 집약적 작업은 스레드 풀에서 실행
        loop = asyncio.get_event_loop()

        cpu_tasks = [
            loop.run_in_executor(
                self.executor,
                self._cpu_intensive_analysis,
                project_id
            )
            for project_id in project_ids
        ]

        # I/O 집약적 작업은 비동기로 실행
        io_tasks = [
            self._io_intensive_fetch(project_id)
            for project_id in project_ids
        ]

        # 모든 작업 완료 대기
        cpu_results = await asyncio.gather(*cpu_tasks)
        io_results = await asyncio.gather(*io_tasks)

        # 결과 조합
        return [
            {"cpu": cpu, "io": io}
            for cpu, io in zip(cpu_results, io_results)
        ]

# 커넥션 풀 관리
class ConnectionManager:
    """연결 관리자"""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(100)  # 동시 연결 제한

    async def bounded_request(self, url: str) -> Dict:
        """제한된 동시 요청"""
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
```

---

## CI/CD 파이프라인

### GitHub Actions 워크플로우

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.7.1"

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_maestro
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}

    - name: Install dependencies
      run: |
        poetry install --with dev

    - name: Run linting
      run: |
        poetry run black --check .
        poetry run isort --check-only .
        poetry run flake8 .
        poetry run mypy src/

    - name: Run tests
      env:
        DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost/test_maestro
        REDIS_URL: redis://localhost:6379/0
        SECRET_KEY: test-secret-key-for-ci-only
      run: |
        poetry run pytest --cov=src/ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8

    - name: Run Bandit security lint
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: Deploy to production
      run: |
        echo "배포 스크립트 실행"
        # 실제 배포 로직
```

### Docker 최적화

```dockerfile
# Dockerfile - 멀티스테이지 빌드
FROM python:3.11-slim as builder

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN pip install poetry

# 의존성 파일 복사
COPY pyproject.toml poetry.lock ./

# 의존성 설치 (가상환경 비활성화)
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root

# 프로덕션 이미지
FROM python:3.11-slim as production

# 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 빌더에서 Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 사용자 생성
RUN useradd --create-home --shell /bin/bash maestro
USER maestro
WORKDIR /home/maestro

# 소스 코드 복사
COPY --chown=maestro:maestro src/ ./src/
COPY --chown=maestro:maestro pyproject.toml ./

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 애플리케이션 실행
EXPOSE 8000
CMD ["uvicorn", "src.project_maestro.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 프로덕션 운영 전략

### 모니터링과 알림

```python
# 프로덕션 모니터링 설정
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import asyncio
import logging

class ProductionMonitoring:
    """프로덕션 모니터링"""

    def __init__(self):
        # 메트릭 정의
        self.http_requests = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )

        self.response_time = Histogram(
            'http_response_time_seconds',
            'HTTP response time'
        )

        self.active_connections = Gauge(
            'active_database_connections',
            'Number of active database connections'
        )

        # 알림 설정
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% 오류율
            'response_time': 2.0,  # 2초 응답 시간
            'memory_usage': 0.85   # 85% 메모리 사용률
        }

    async def check_system_health(self):
        """시스템 상태 확인"""
        while True:
            try:
                # CPU, 메모리 사용률 확인
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent / 100

                # 데이터베이스 연결 확인
                db_connections = await self._check_db_connections()

                # 임계값 초과 시 알림
                if memory_usage > self.alert_thresholds['memory_usage']:
                    await self._send_alert(
                        "HIGH_MEMORY_USAGE",
                        f"메모리 사용률: {memory_usage:.1%}"
                    )

                # 메트릭 업데이트
                self.active_connections.set(db_connections)

            except Exception as e:
                logger.error(f"모니터링 오류: {e}")

            await asyncio.sleep(30)  # 30초마다 확인

    async def _send_alert(self, alert_type: str, message: str):
        """알림 전송"""
        # Slack, 이메일, PagerDuty 등으로 알림
        logger.critical(f"[ALERT] {alert_type}: {message}")

        # 외부 알림 서비스 호출
        await self._notify_slack(alert_type, message)

# 로그 집계 설정 (ELK Stack)
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/maestro/app.log',
            'maxBytes': 100_000_000,  # 100MB
            'backupCount': 5,
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}
```

### 배포 전략

```python
# 블루-그린 배포 스크립트
import subprocess
import time
import requests

class BlueGreenDeployment:
    """블루-그린 배포 관리"""

    def __init__(self):
        self.blue_port = 8000
        self.green_port = 8001
        self.load_balancer_url = "http://localhost:80"

    async def deploy_new_version(self, image_tag: str):
        """새 버전 배포"""
        try:
            # 1. 현재 활성 환경 확인
            active_env = await self._get_active_environment()
            inactive_env = "green" if active_env == "blue" else "blue"
            inactive_port = self.green_port if inactive_env == "green" else self.blue_port

            logger.info(f"현재 활성 환경: {active_env}")
            logger.info(f"배포 대상 환경: {inactive_env}")

            # 2. 비활성 환경에 새 버전 배포
            await self._deploy_to_environment(inactive_env, image_tag, inactive_port)

            # 3. 헬스 체크
            if not await self._health_check(inactive_port):
                raise Exception("새 버전 헬스 체크 실패")

            # 4. 트래픽 점진적 전환 (카나리 배포)
            await self._gradual_traffic_switch(active_env, inactive_env)

            # 5. 이전 환경 정리
            await self._cleanup_old_environment(active_env)

            logger.info("배포 완료")

        except Exception as e:
            # 롤백
            logger.error(f"배포 실패: {e}")
            await self._rollback(active_env)
            raise

    async def _deploy_to_environment(self, env: str, image_tag: str, port: int):
        """특정 환경에 배포"""
        # Docker 컨테이너 시작
        cmd = [
            "docker", "run", "-d",
            "--name", f"maestro-{env}",
            "-p", f"{port}:8000",
            "-e", f"ENVIRONMENT={env}",
            f"ghcr.io/project-maestro/project-maestro:{image_tag}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"컨테이너 시작 실패: {result.stderr}")

        # 컨테이너 시작 대기
        await asyncio.sleep(10)

    async def _health_check(self, port: int, timeout: int = 60) -> bool:
        """헬스 체크"""
        url = f"http://localhost:{port}/health"

        for _ in range(timeout):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        return True
            except:
                pass

            await asyncio.sleep(1)

        return False

    async def _gradual_traffic_switch(self, old_env: str, new_env: str):
        """점진적 트래픽 전환"""
        # 10% → 50% → 100% 단계적 전환
        traffic_ratios = [10, 50, 100]

        for ratio in traffic_ratios:
            await self._update_load_balancer(new_env, ratio)

            # 모니터링 기간
            await asyncio.sleep(300)  # 5분 대기

            # 오류율 확인
            error_rate = await self._check_error_rate()
            if error_rate > 0.05:  # 5% 초과 시 롤백
                raise Exception(f"오류율 초과: {error_rate:.2%}")

        logger.info("트래픽 전환 완료")

# Kubernetes 배포 (Helm 차트)
# values.yaml
replicaCount: 3

image:
  repository: ghcr.io/project-maestro/project-maestro
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.projectmaestro.dev
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

---

## 마무리: 백엔드 마스터로의 여정

### 학습 여정 요약

지금까지 4단계에 걸쳐 Python 백엔드 개발의 전체 스펙트럼을 다뤘습니다:

1. **기초 단계**: Python 문법, 함수, 클래스, 모듈 시스템
2. **고급 단계**: 비동기, 병렬처리, 예외처리, 의존성 주입
3. **아키텍처 단계**: 마이크로서비스, 이벤트 드리븐, API 설계
4. **실무 단계**: 테스트, 보안, 성능, CI/CD, 운영

### 실무 적용 체크리스트

**개발 준비**
- [ ] 개발환경 설정 (Poetry, Pre-commit, IDE)
- [ ] 프로젝트 구조 설계 (레이어드 아키텍처)
- [ ] 의존성 관리 (pyproject.toml 설정)

**코드 품질**
- [ ] 타입 힌트 100% 적용
- [ ] 단위 테스트 80% 이상 커버리지
- [ ] 코드 스타일 자동화 (Black, isort)
- [ ] 정적 분석 도구 적용 (mypy, flake8)

**아키텍처**
- [ ] 비동기 처리 설계
- [ ] 데이터베이스 최적화
- [ ] 캐싱 전략 수립
- [ ] 모니터링 시스템 구축

**운영**
- [ ] CI/CD 파이프라인 구축
- [ ] 컨테이너화 (Docker)
- [ ] 보안 검사 자동화
- [ ] 로그 및 메트릭 수집

### 계속 학습할 주제들

**심화 주제**
- 분산 시스템 설계 패턴
- 마이크로서비스 오케스트레이션
- 이벤트 소싱 및 CQRS
- 서버리스 아키텍처

**기술 트렌드**
- FastAPI 고급 기능
- PostgreSQL 성능 튜닝
- Kubernetes 운영
- 관찰가능성 (Observability)

### 토론 준비 완료!

이제 다음과 같은 백엔드 토론에 자신 있게 참여할 수 있습니다:

- "마이크로서비스 vs 모놀리스 아키텍처 선택 기준"
- "Python GIL의 한계와 해결책"
- "데이터베이스 스케일링 전략"
- "API 설계 베스트 프랙티스"
- "클라우드 네이티브 애플리케이션 설계"
- "DevOps 문화와 CI/CD 파이프라인"

**토론 시 활용할 수 있는 키워드들:**
- 이벤트 드리븐 아키텍처, CQRS, 사가 패턴
- 비동기 처리, 연결 풀링, 캐싱 전략
- 컨테이너 오케스트레이션, 서비스 메시
- 분산 추적, 메트릭, 로그 집계
- 보안 by 설계, 제로 트러스트

축하합니다! 이제 Python 백엔드 개발자로서 실무에서 바로 활용할 수 있는 종합적인 지식을 갖추셨습니다. 🎉