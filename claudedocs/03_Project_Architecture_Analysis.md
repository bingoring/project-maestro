# Python 기초부터 마스터까지 - 3단계: 프로젝트 아키텍처 분석

## 목차
1. [프로젝트 전체 구조 개요](#프로젝트-전체-구조-개요)
2. [마이크로서비스 아키텍처 패턴](#마이크로서비스-아키텍처-패턴)
3. [의존성 관리와 모듈 구조](#의존성-관리와-모듈-구조)
4. [FastAPI 기반 REST API 설계](#fastapi-기반-rest-api-설계)
5. [에이전트 시스템 아키텍처](#에이전트-시스템-아키텍처)
6. [데이터 계층과 스토리지](#데이터-계층과-스토리지)
7. [이벤트 드리븐 아키텍처](#이벤트-드리븐-아키텍처)
8. [모니터링과 관찰가능성](#모니터링과-관찰가능성)

---

## 프로젝트 전체 구조 개요

### Project Maestro 시스템 개요

Project Maestro는 **AI 에이전트 기반 게임 프로토타이핑 자동화 시스템**입니다:

```
프로젝트 마에스트로 아키텍처
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (웹 인터페이스)                  │
└─────────────────────────┬───────────────────────────────────┘
                         │ HTTP/WebSocket
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Gateway                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Project API │ │ Agent API   │ │ Asset API              │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                         │ Event Bus (Redis)
┌─────────────────────────▼───────────────────────────────────┐
│                   Agent Orchestrator                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Builder     │ │ Canvas      │ │ Codex                  │ │
│  │ Agent       │ │ Agent       │ │ Agent                  │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                         │
┌─────────────────────────▼───────────────────────────────────┐
│                    데이터 계층                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ PostgreSQL  │ │ Redis Cache │ │ MinIO (파일 스토리지)    │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 디렉토리 구조 분석

```python
src/project_maestro/
├── __init__.py              # 패키지 초기화
├── cli.py                   # CLI 인터페이스 (Typer)
│
├── api/                     # REST API 계층
│   ├── main.py             # FastAPI 애플리케이션
│   ├── models.py           # Pydantic 모델들
│   └── endpoints/          # API 엔드포인트들
│       ├── projects.py     # 프로젝트 관리
│       ├── agents.py       # 에이전트 관리
│       ├── assets.py       # 자산 관리
│       ├── builds.py       # 빌드 관리
│       └── analytics.py    # 분석 데이터
│
├── agents/                  # AI 에이전트들
│   ├── orchestrator.py     # 마스터 오케스트레이터
│   ├── builder_agent.py    # 빌드 에이전트
│   ├── canvas_agent.py     # 캔버스(UI) 에이전트
│   ├── codex_agent.py      # 코드 생성 에이전트
│   └── query_agent.py      # 쿼리 처리 에이전트
│
├── core/                    # 핵심 시스템 컴포넌트
│   ├── config.py           # 설정 관리
│   ├── logging.py          # 로깅 시스템
│   ├── agent_framework.py  # 에이전트 프레임워크
│   ├── gdd_parser.py       # 게임 디자인 문서 파서
│   ├── rag_system.py       # RAG (검색 증강 생성)
│   ├── intelligent_cache.py # 지능형 캐싱
│   ├── distributed_workflow.py # 분산 워크플로우
│   ├── monitoring.py       # 모니터링 시스템
│   └── message_queue.py    # 메시지 큐
│
├── integrations/           # 외부 서비스 통합
└── scripts/               # 유틸리티 스크립트
```

**아키텍처 원칙:**
- **레이어드 아키텍처**: API → 비즈니스 로직 → 데이터 계층
- **마이크로서비스 지향**: 각 에이전트는 독립적 서비스
- **이벤트 드리븐**: Redis 기반 메시지 큐로 느슨한 결합
- **도메인 주도 설계**: 게임 개발 도메인 중심 구조

---

## 마이크로서비스 아키텍처 패턴

### 서비스 분리 전략

```python
# src/project_maestro/core/agent_framework.py에서 발췌
from enum import Enum
from abc import ABC, abstractmethod

class AgentType(Enum):
    """에이전트 타입 정의 - 각각이 마이크로서비스"""
    ORCHESTRATOR = "orchestrator"     # 워크플로우 조정
    BUILDER = "builder"               # 프로젝트 빌드
    CANVAS = "canvas"                 # UI/그래픽 생성
    CODEX = "codex"                   # 코드 생성
    LABYRINTH = "labyrinth"          # 레벨 디자인
    SONATA = "sonata"                # 사운드/음악
    QUERY = "query"                  # 데이터 쿼리

class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""

    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.capabilities = self._define_capabilities()

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Any:
        """태스크 실행 - 각 에이전트마다 다른 구현"""
        pass

    @abstractmethod
    def _define_capabilities(self) -> List[str]:
        """에이전트 능력 정의"""
        pass
```

### 서비스 간 통신 패턴

```python
# src/project_maestro/core/message_queue.py 분석
import asyncio
import json
from typing import Dict, Any, Callable, List
from enum import Enum
import redis.asyncio as redis

class EventType(Enum):
    """이벤트 타입 정의"""
    PROJECT_CREATED = "project.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_COMPLETED = "task.completed"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    BUILD_STARTED = "build.started"
    BUILD_COMPLETED = "build.completed"
    ASSET_GENERATED = "asset.generated"

class EventBus:
    """Redis 기반 이벤트 버스"""

    def __init__(self, redis_url: str):
        self.redis_pool = redis.ConnectionPool.from_url(redis_url)
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self._listening = False

    async def publish(self, event_type: EventType, data: Dict[str, Any]):
        """이벤트 발행"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            message = {
                'event_type': event_type.value,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'id': str(uuid.uuid4())
            }

            await r.publish(
                f"maestro.{event_type.value}",
                json.dumps(message)
            )

    async def subscribe(self, event_type: EventType, handler: Callable):
        """이벤트 구독"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def start_listening(self):
        """이벤트 리스닝 시작"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            pubsub = r.pubsub()

            # 모든 마에스트로 이벤트 구독
            await pubsub.psubscribe("maestro.*")

            self._listening = True
            async for message in pubsub.listen():
                if message['type'] == 'pmessage':
                    await self._handle_message(message)

# 서비스 간 통신 예시
async def project_creation_workflow():
    """프로젝트 생성 워크플로우"""
    event_bus = get_event_bus()

    # 1. 프로젝트 생성 이벤트 발행
    await event_bus.publish(EventType.PROJECT_CREATED, {
        'project_id': project_id,
        'title': 'New Game Project',
        'complexity': 7
    })

    # 2. 오케스트레이터가 이벤트 수신하여 작업 분배
    # 3. 각 에이전트가 할당된 작업 수행
    # 4. 완료 이벤트 발행
```

### API Gateway 패턴

```python
# src/project_maestro/api/main.py 분석
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("Starting Project Maestro API server")

    # 이벤트 버스 초기화
    event_bus = await get_event_bus()
    await event_bus.start_listening()

    # 에이전트 초기화
    await _initialize_agents()

    # 모니터링 시작
    await start_monitoring()

    yield  # 서버 실행

    # 종료 시
    await stop_monitoring()
    await event_bus.stop_listening()

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Project Maestro API",
    description="AI Agent-based Game Prototyping Automation System",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 미들웨어 (마이크로서비스 간 통신용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (서비스별 엔드포인트)
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(builds.router, prefix="/api/v1/builds", tags=["builds"])
```

**마이크로서비스 이점:**
- **독립적 배포**: 각 에이전트를 별도로 스케일링 가능
- **기술 스택 다양성**: 에이전트별로 최적 기술 선택 가능
- **장애 격리**: 한 에이전트 실패가 전체 시스템에 영향 안 줌
- **팀 독립성**: 에이전트별로 팀이 독립적으로 개발 가능

---

## 의존성 관리와 모듈 구조

### 현대적 Python 의존성 관리

프로젝트는 `pyproject.toml`을 사용한 현대적 Python 패키징을 채택합니다:

```toml
# pyproject.toml 주요 의존성 분석

[project]
name = "project-maestro"
requires-python = ">=3.11"  # 최신 Python 버전 요구

dependencies = [
    # 🤖 AI/ML 프레임워크
    "langchain>=0.3.0",           # LLM 애플리케이션 프레임워크
    "langchain-openai>=0.2.0",    # OpenAI 통합
    "langchain-anthropic>=0.2.0", # Anthropic 통합
    "langgraph>=0.2.0",           # 그래프 기반 워크플로우

    # 🌐 웹 프레임워크
    "fastapi>=0.104.0",           # 모던 Python 웹 프레임워크
    "uvicorn[standard]>=0.24.0",  # ASGI 서버
    "pydantic>=2.5.0",            # 데이터 검증

    # 📊 데이터베이스
    "sqlalchemy>=2.0.0",          # ORM (최신 2.0 스타일)
    "asyncpg>=0.29.0",            # PostgreSQL 비동기 드라이버
    "redis>=5.0.0",               # 캐싱 및 메시지 큐

    # ⚡ 태스크 큐
    "celery[redis]>=5.3.0",       # 분산 태스크 처리

    # 📁 파일 스토리지
    "minio>=7.2.0",               # S3 호환 스토리지
    "boto3>=1.34.0",              # AWS 통합

    # 📊 모니터링
    "prometheus-client>=0.19.0",   # 메트릭 수집
    "opentelemetry-api>=1.21.0",   # 분산 추적
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",             # 테스팅
    "black>=23.12.0",             # 코드 포맷팅
    "mypy>=1.8.0",                # 타입 체킹
]
```

### 계층화된 의존성 구조

```python
# 의존성 계층 구조
"""
Application Layer (FastAPI)
    ↓ depends on
Business Logic Layer (Agents, Core)
    ↓ depends on
Infrastructure Layer (Database, Cache, Storage)
    ↓ depends on
External Services (OpenAI, MinIO, Redis)
"""

# src/project_maestro/core/config.py - 의존성 주입 컨테이너
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경별 설정 관리"""

    # 환경 구분
    environment: str = Field(default="development")

    # AI 서비스 설정
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # 데이터베이스 설정
    database_url: str = Field(
        default="postgresql+asyncpg://maestro:password@localhost:5432/project_maestro"
    )

    # Redis 설정
    redis_url: str = Field(default="redis://localhost:6379/0")

    # 스토리지 설정
    storage_type: str = Field(default="minio")
    minio_endpoint: str = Field(default="localhost:9000")

    class Config:
        env_file = ".env"  # 환경변수 파일 자동 로드

# 전역 설정 인스턴스
settings = Settings()
```

### 모듈 간 의존성 해결

```python
# src/project_maestro/__init__.py
"""패키지 레벨에서 주요 컴포넌트 노출"""

from .core.config import settings
from .core.logging import logger

# 외부에서 쉽게 접근 가능한 API
__all__ = ["settings", "logger"]

# 각 모듈에서 사용법
# from project_maestro import settings, logger
```

```python
# 의존성 주입 패턴 예시
from typing import Protocol

class StorageInterface(Protocol):
    """스토리지 인터페이스"""
    async def upload_file(self, key: str, data: bytes) -> str: ...
    async def download_file(self, key: str) -> bytes: ...

class AssetManager:
    """자산 관리자 - 의존성 주입 받음"""

    def __init__(self, storage: StorageInterface):
        self.storage = storage  # 인터페이스에 의존

    async def save_asset(self, asset_data: bytes) -> str:
        asset_id = generate_uuid()
        url = await self.storage.upload_file(asset_id, asset_data)
        return url

# 런타임에 구현체 주입
minio_storage = MinIOStorage(settings.minio_endpoint)
asset_manager = AssetManager(minio_storage)
```

---

## FastAPI 기반 REST API 설계

### RESTful API 설계 원칙

```python
# src/project_maestro/api/endpoints/projects.py 분석
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

router = APIRouter()

# RESTful 엔드포인트 설계
@router.post("/", response_model=ProjectResponse)
async def create_project(
    request: ProjectCreateRequest,      # 요청 본문
    background_tasks: BackgroundTasks,  # 비동기 작업
    db: Session = Depends(get_db)       # 의존성 주입
):
    """새 게임 프로젝트 생성

    HTTP POST /api/v1/projects/
    """
    try:
        project_id = str(uuid.uuid4())

        # 오케스트레이터 에이전트 조회
        orchestrator = agent_registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        if not orchestrator:
            raise HTTPException(
                status_code=503,
                detail="Orchestrator agent not available"
            )

        # 백그라운드에서 프로젝트 처리 시작
        background_tasks.add_task(
            _process_project_creation,
            project_id,
            request.game_design_document
        )

        # 즉시 응답 반환 (비동기 처리)
        return ProjectResponse(
            id=project_id,
            title=request.title,
            status="processing",
            created_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"프로젝트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 오류")

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: Session = Depends(get_db)
):
    """프로젝트 조회

    HTTP GET /api/v1/projects/{project_id}
    """
    # 프로젝트 조회 로직...

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    db: Session = Depends(get_db)
):
    """프로젝트 업데이트

    HTTP PUT /api/v1/projects/{project_id}
    """
    # 프로젝트 업데이트 로직...

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db)
):
    """프로젝트 삭제

    HTTP DELETE /api/v1/projects/{project_id}
    """
    # 프로젝트 삭제 로직...

@router.get("/", response_model=PaginatedResponse[ProjectResponse])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """프로젝트 목록 조회 (페이지네이션)

    HTTP GET /api/v1/projects/?skip=0&limit=50&status=active
    """
    # 페이지네이션된 프로젝트 목록 반환...
```

### Pydantic 모델을 통한 데이터 검증

```python
# src/project_maestro/api/models.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class ProjectStatus(str, Enum):
    """프로젝트 상태"""
    DRAFT = "draft"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProjectCreateRequest(BaseModel):
    """프로젝트 생성 요청"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    game_design_document: str = Field(..., min_length=100)
    target_platform: str = Field(default="mobile")

    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('제목은 비어있을 수 없습니다')
        return v.strip()

    @validator('game_design_document')
    def validate_gdd(cls, v):
        if len(v.split()) < 50:
            raise ValueError('게임 디자인 문서가 너무 짧습니다')
        return v

class ProjectResponse(BaseModel):
    """프로젝트 응답"""
    id: str
    title: str
    description: Optional[str]
    status: ProjectStatus
    progress: float = Field(ge=0.0, le=100.0)
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True  # SQLAlchemy 모델과 호환

class BaseResponse(BaseModel):
    """기본 응답 구조"""
    success: bool = True
    message: Optional[str] = None

class ErrorResponse(BaseResponse):
    """오류 응답"""
    success: bool = False
    error_code: str
    details: Optional[Dict[str, Any]] = None

# 제네릭 페이지네이션 응답
from typing import TypeVar, Generic
T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """페이지네이션된 응답"""
    items: List[T]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool
```

### 미들웨어와 예외 처리

```python
# API 레벨 예외 처리
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Pydantic 검증 오류 처리"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="입력 데이터가 올바르지 않습니다",
            details={"validation_errors": exc.errors()}
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code="HTTP_ERROR",
            message=exc.detail
        ).dict()
    )

# 요청/응답 로깅 미들웨어
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    # 요청 로깅
    logger.info(
        "요청 시작",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host
    )

    response = await call_next(request)

    # 응답 로깅
    process_time = time.time() - start_time
    logger.info(
        "요청 완료",
        status_code=response.status_code,
        process_time=process_time
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## 에이전트 시스템 아키텍처

### 에이전트 프레임워크 설계

```python
# src/project_maestro/agents/orchestrator.py 분석
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class OrchestratorAgent(BaseAgent):
    """마스터 오케스트레이터 에이전트"""

    def __init__(self):
        super().__init__("orchestrator-001", AgentType.ORCHESTRATOR)

        # LangChain 체인 구성
        self.llm = self._setup_llm()
        self.memory = MemorySaver()
        self.tools = self._setup_tools()

        # ReAct 에이전트 생성 (최신 LangGraph)
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.memory
        )

    def _setup_llm(self):
        """언어 모델 설정"""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )

    def _setup_tools(self):
        """에이전트 도구 설정"""
        return [
            self._create_project_analysis_tool(),
            self._create_task_assignment_tool(),
            self._create_progress_tracking_tool(),
            self._create_quality_check_tool()
        ]

    async def execute_task(self, task: AgentTask) -> Any:
        """태스크 실행"""
        config = {"configurable": {"thread_id": task.id}}

        # 대화형 실행
        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=task.description)]},
            config=config
        )

        return response["messages"][-1].content

class ProjectSpec(BaseModel):
    """프로젝트 명세서"""
    id: str
    title: str
    description: str
    genre: str
    platform: str = "mobile"
    art_style: str
    gameplay_mechanics: List[str]
    characters: List[Dict[str, Any]]
    environments: List[Dict[str, Any]]
    technical_requirements: Dict[str, Any]
    estimated_complexity: int = Field(ge=1, le=10)

class WorkflowStep(BaseModel):
    """워크플로우 단계"""
    id: str
    name: str
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = []
    estimated_duration: timedelta
```

### 에이전트 간 협업 패턴

```python
class AgentCollaboration:
    """에이전트 협업 관리"""

    async def execute_collaborative_workflow(self, project_spec: ProjectSpec):
        """협업 워크플로우 실행"""

        # 1단계: 분석 및 계획 (Orchestrator)
        analysis_result = await self._analyze_project(project_spec)

        # 2단계: 병렬 자산 생성
        asset_tasks = await asyncio.gather(
            self._generate_code(analysis_result),      # Codex Agent
            self._generate_graphics(analysis_result),  # Canvas Agent
            self._generate_sounds(analysis_result),    # Sonata Agent
            self._design_levels(analysis_result),      # Labyrinth Agent
            return_exceptions=True
        )

        # 3단계: 통합 및 빌드 (Builder Agent)
        build_result = await self._integrate_assets(asset_tasks)

        # 4단계: 품질 검증 (Query Agent)
        quality_report = await self._verify_quality(build_result)

        return {
            'project_id': project_spec.id,
            'assets': asset_tasks,
            'build': build_result,
            'quality': quality_report
        }

    async def _generate_code(self, analysis: Dict) -> Dict:
        """코드 생성 (Codex Agent)"""
        codex_agent = agent_registry.get_agent("codex-001")

        task = AgentTask(
            id=f"code-gen-{uuid.uuid4()}",
            type="code_generation",
            description=f"Generate {analysis['platform']} game code",
            parameters={
                'genre': analysis['genre'],
                'mechanics': analysis['mechanics'],
                'platform': analysis['platform']
            }
        )

        return await codex_agent.execute_task(task)
```

### LangChain/LangGraph 통합

```python
# 최신 LangChain 패턴 활용
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langgraph.graph import StateGraph, END

class GameDevelopmentWorkflow:
    """게임 개발 워크플로우 그래프"""

    def __init__(self):
        self.workflow = StateGraph(WorkflowState)
        self._build_graph()

    def _build_graph(self):
        """워크플로우 그래프 구성"""

        # 노드 추가
        self.workflow.add_node("analyze", self._analyze_requirements)
        self.workflow.add_node("plan", self._create_development_plan)
        self.workflow.add_node("generate_assets", self._generate_assets)
        self.workflow.add_node("integrate", self._integrate_components)
        self.workflow.add_node("test", self._run_quality_tests)

        # 엣지 정의 (흐름 제어)
        self.workflow.set_entry_point("analyze")
        self.workflow.add_edge("analyze", "plan")
        self.workflow.add_edge("plan", "generate_assets")
        self.workflow.add_edge("generate_assets", "integrate")
        self.workflow.add_edge("integrate", "test")

        # 조건부 분기
        self.workflow.add_conditional_edges(
            "test",
            self._should_iterate,
            {
                "iterate": "plan",  # 품질 불만족 시 재시도
                "complete": END     # 완료
            }
        )

    async def _analyze_requirements(self, state: WorkflowState):
        """요구사항 분석"""
        gdd_parser = GameDesignDocumentParser()
        analysis = await gdd_parser.parse(state["gdd_content"])

        state["analysis"] = analysis
        return state

    def _should_iterate(self, state: WorkflowState) -> str:
        """반복 여부 결정"""
        quality_score = state.get("quality_score", 0)
        return "complete" if quality_score > 8.0 else "iterate"
```

---

## 데이터 계층과 스토리지

### SQLAlchemy 2.0 스타일 ORM

```python
# 모던 SQLAlchemy 사용법
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Project(Base):
    """프로젝트 모델"""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="draft")
    gdd_content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # 관계 정의
    assets = relationship("Asset", back_populates="project")
    builds = relationship("Build", back_populates="project")

class Asset(Base):
    """자산 모델"""
    __tablename__ = "assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"))
    asset_type = Column(String(50))  # code, image, sound, model
    file_url = Column(String(500))
    metadata = Column(Text)  # JSON 형태의 메타데이터

    # 관계
    project = relationship("Project", back_populates="assets")

# 비동기 데이터베이스 엔진
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# 의존성 주입용 데이터베이스 세션
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

### 비동기 데이터베이스 작업

```python
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

class ProjectRepository:
    """프로젝트 리포지토리"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_project(self, project_data: dict) -> Project:
        """프로젝트 생성"""
        project = Project(**project_data)
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        """프로젝트 조회 (관련 자산 포함)"""
        query = select(Project).options(
            selectinload(Project.assets),
            selectinload(Project.builds)
        ).where(Project.id == project_id)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_project_status(self, project_id: str, status: str):
        """프로젝트 상태 업데이트"""
        query = update(Project).where(
            Project.id == project_id
        ).values(
            status=status,
            updated_at=datetime.utcnow()
        )

        await self.session.execute(query)
        await self.session.commit()

    async def list_projects(self, skip: int = 0, limit: int = 50) -> List[Project]:
        """프로젝트 목록 조회"""
        query = select(Project).offset(skip).limit(limit).order_by(
            Project.created_at.desc()
        )

        result = await self.session.execute(query)
        return result.scalars().all()
```

### Redis 캐싱 및 세션 관리

```python
# src/project_maestro/core/intelligent_cache.py에서 발췌
import redis.asyncio as redis
import json
import pickle
from typing import Optional, Any

class RedisCache:
    """Redis 기반 캐싱 시스템"""

    def __init__(self, redis_url: str):
        self.redis_pool = redis.ConnectionPool.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            data = await r.get(key)
            if data:
                return pickle.loads(data)
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """캐시에 데이터 저장"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            serialized = pickle.dumps(value)
            await r.setex(key, ttl, serialized)

    async def delete(self, key: str):
        """캐시에서 데이터 삭제"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            await r.delete(key)

    async def get_or_compute(self, key: str, compute_func, ttl: int = 3600):
        """캐시 우선 조회, 없으면 계산 후 저장"""
        # 캐시에서 먼저 확인
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # 캐시 미스 시 계산
        computed_value = await compute_func()
        await self.set(key, computed_value, ttl)

        return computed_value

# 사용 예시
cache = RedisCache(settings.redis_url)

async def get_project_analysis(project_id: str):
    """프로젝트 분석 결과 조회 (캐시 활용)"""

    async def compute_analysis():
        # 실제 분석 로직 (시간이 오래 걸림)
        project = await project_repo.get_project(project_id)
        return await complex_analysis(project)

    return await cache.get_or_compute(
        f"project_analysis:{project_id}",
        compute_analysis,
        ttl=1800  # 30분 캐시
    )
```

### MinIO 기반 파일 스토리지

```python
from minio import Minio
from minio.error import S3Error
import io

class AssetStorage:
    """MinIO 기반 자산 스토리지"""

    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False  # 개발환경
        )
        self.bucket_name = "maestro-assets"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """버킷 존재 확인 및 생성"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            logger.error(f"버킷 생성 실패: {e}")

    async def upload_asset(self, asset_id: str, data: bytes, content_type: str) -> str:
        """자산 업로드"""
        try:
            # 파일 스트림 생성
            data_stream = io.BytesIO(data)

            # MinIO에 업로드
            self.client.put_object(
                self.bucket_name,
                asset_id,
                data_stream,
                length=len(data),
                content_type=content_type
            )

            # 공개 URL 생성
            url = f"http://{settings.minio_endpoint}/{self.bucket_name}/{asset_id}"
            return url

        except S3Error as e:
            logger.error(f"자산 업로드 실패: {e}")
            raise

    async def download_asset(self, asset_id: str) -> bytes:
        """자산 다운로드"""
        try:
            response = self.client.get_object(self.bucket_name, asset_id)
            return response.data
        except S3Error as e:
            logger.error(f"자산 다운로드 실패: {e}")
            raise
```

---

## 이벤트 드리븐 아키텍처

### 이벤트 기반 통신 패턴

```python
# src/project_maestro/core/message_queue.py 심화 분석
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio

@dataclass
class Event:
    """이벤트 모델"""
    id: str
    type: EventType
    source: str  # 이벤트 발생 소스
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # 관련 이벤트 추적용

class EventHandler:
    """이벤트 핸들러 기본 클래스"""

    async def handle(self, event: Event) -> None:
        """이벤트 처리"""
        pass

    def can_handle(self, event_type: EventType) -> bool:
        """처리 가능한 이벤트 타입인지 확인"""
        return False

class ProjectCreatedHandler(EventHandler):
    """프로젝트 생성 이벤트 핸들러"""

    def can_handle(self, event_type: EventType) -> bool:
        return event_type == EventType.PROJECT_CREATED

    async def handle(self, event: Event) -> None:
        """프로젝트 생성 이벤트 처리"""
        project_data = event.data

        # 1. 프로젝트 분석 시작
        await self._start_project_analysis(project_data)

        # 2. 에이전트 할당
        await self._assign_agents(project_data)

        # 3. 진행 상황 추적 시작
        await self._start_progress_tracking(project_data['project_id'])

class EventBus:
    """개선된 이벤트 버스"""

    def __init__(self, redis_url: str):
        self.redis_pool = redis.ConnectionPool.from_url(redis_url)
        self.handlers: List[EventHandler] = []
        self.dead_letter_queue = f"maestro:dlq"

    def register_handler(self, handler: EventHandler):
        """이벤트 핸들러 등록"""
        self.handlers.append(handler)

    async def publish(self, event: Event):
        """이벤트 발행"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            # 이벤트 직렬화
            event_data = {
                'id': event.id,
                'type': event.type.value,
                'source': event.source,
                'data': event.data,
                'timestamp': event.timestamp.isoformat(),
                'correlation_id': event.correlation_id
            }

            # Redis Stream에 발행 (순서 보장)
            await r.xadd(
                f"maestro:events:{event.type.value}",
                event_data
            )

    async def consume_events(self):
        """이벤트 소비"""
        async with redis.Redis(connection_pool=self.redis_pool) as r:
            # 모든 이벤트 스트림 구독
            streams = {
                f"maestro:events:{event_type.value}": "0"
                for event_type in EventType
            }

            while True:
                try:
                    # 새 이벤트 대기
                    events = await r.xread(streams, block=1000)

                    for stream_name, stream_events in events:
                        for event_id, event_data in stream_events:
                            await self._process_event(event_data)

                            # 처리 완료된 이벤트 ACK
                            await r.xdel(stream_name, event_id)

                except Exception as e:
                    logger.error(f"이벤트 처리 중 오류: {e}")
                    await asyncio.sleep(1)

    async def _process_event(self, event_data: Dict):
        """개별 이벤트 처리"""
        try:
            # 이벤트 역직렬화
            event = Event(
                id=event_data['id'],
                type=EventType(event_data['type']),
                source=event_data['source'],
                data=event_data['data'],
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                correlation_id=event_data.get('correlation_id')
            )

            # 적절한 핸들러 찾기
            for handler in self.handlers:
                if handler.can_handle(event.type):
                    await handler.handle(event)

        except Exception as e:
            logger.error(f"이벤트 처리 실패: {e}")
            # Dead Letter Queue로 이동
            await self._send_to_dlq(event_data, str(e))
```

### 사가 패턴 (분산 트랜잭션)

```python
from enum import Enum
from typing import List, Callable

class SagaStepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    """사가 단계"""
    name: str
    execute: Callable
    compensate: Callable  # 롤백 함수
    status: SagaStepStatus = SagaStepStatus.PENDING

class ProjectCreationSaga:
    """프로젝트 생성 사가"""

    def __init__(self, project_data: Dict):
        self.project_data = project_data
        self.steps = self._define_steps()
        self.completed_steps = []

    def _define_steps(self) -> List[SagaStep]:
        """사가 단계 정의"""
        return [
            SagaStep(
                name="create_project_record",
                execute=self._create_project_record,
                compensate=self._delete_project_record
            ),
            SagaStep(
                name="allocate_storage",
                execute=self._allocate_storage,
                compensate=self._deallocate_storage
            ),
            SagaStep(
                name="assign_agents",
                execute=self._assign_agents,
                compensate=self._unassign_agents
            ),
            SagaStep(
                name="start_processing",
                execute=self._start_processing,
                compensate=self._stop_processing
            )
        ]

    async def execute(self) -> bool:
        """사가 실행"""
        try:
            for step in self.steps:
                logger.info(f"사가 단계 실행: {step.name}")

                await step.execute(self.project_data)
                step.status = SagaStepStatus.COMPLETED
                self.completed_steps.append(step)

                logger.info(f"사가 단계 완료: {step.name}")

            return True

        except Exception as e:
            logger.error(f"사가 실행 실패: {e}")
            await self._compensate()
            return False

    async def _compensate(self):
        """보상 트랜잭션 실행 (롤백)"""
        logger.warning("사가 보상 트랜잭션 시작")

        # 완료된 단계들을 역순으로 보상
        for step in reversed(self.completed_steps):
            try:
                logger.info(f"보상 트랜잭션 실행: {step.name}")
                await step.compensate(self.project_data)
                step.status = SagaStepStatus.COMPENSATED

            except Exception as e:
                logger.error(f"보상 트랜잭션 실패: {step.name}, {e}")
                # 보상 실패는 수동 개입 필요
                await self._alert_manual_intervention(step, e)

# 사가 사용 예시
async def create_project_with_saga(project_data: Dict):
    """사가를 사용한 프로젝트 생성"""
    saga = ProjectCreationSaga(project_data)

    success = await saga.execute()

    if success:
        # 성공 이벤트 발행
        await event_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.PROJECT_CREATED,
            source="project_service",
            data=project_data,
            timestamp=datetime.utcnow()
        ))
    else:
        # 실패 이벤트 발행
        await event_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.PROJECT_CREATION_FAILED,
            source="project_service",
            data=project_data,
            timestamp=datetime.utcnow()
        ))

    return success
```

---

## 모니터링과 관찰가능성

### Prometheus 메트릭 수집

```python
# src/project_maestro/core/monitoring.py 심화 분석
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import psutil
import time

class MetricsCollector:
    """프로메테우스 메트릭 수집기"""

    def __init__(self):
        # 메트릭 정의
        self.request_count = Counter(
            'maestro_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )

        self.request_duration = Histogram(
            'maestro_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )

        self.active_projects = Gauge(
            'maestro_active_projects',
            'Number of active projects'
        )

        self.agent_status = Gauge(
            'maestro_agent_status',
            'Agent status (1=active, 0=inactive)',
            ['agent_id', 'agent_type']
        )

        self.system_memory = Gauge(
            'maestro_system_memory_usage_percent',
            'System memory usage percentage'
        )

        self.system_cpu = Gauge(
            'maestro_system_cpu_usage_percent',
            'System CPU usage percentage'
        )

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """요청 메트릭 기록"""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        # 메모리 사용률
        memory = psutil.virtual_memory()
        self.system_memory.set(memory.percent)

        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu.set(cpu_percent)

    def update_agent_metrics(self, agents: List[BaseAgent]):
        """에이전트 메트릭 업데이트"""
        for agent in agents:
            status_value = 1 if agent.status == AgentStatus.ACTIVE else 0
            self.agent_status.labels(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type.value
            ).set(status_value)

# FastAPI 메트릭 미들웨어
from fastapi import Request
import time

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """메트릭 수집 미들웨어"""
    start_time = time.time()

    response = await call_next(request)

    # 메트릭 기록
    duration = time.time() - start_time
    metrics_collector.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )

    return response
```

### 구조화된 로깅

```python
# src/project_maestro/core/logging.py
import structlog
import logging.config
from typing import Any, Dict

def configure_logging():
    """구조화된 로깅 설정"""

    # structlog 설정
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.debug
            else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """로거 인스턴스 획득"""
    return structlog.get_logger(name)

# 사용 예시
logger = get_logger("project_service")

async def create_project(project_data: Dict[str, Any]):
    """구조화된 로깅을 사용한 프로젝트 생성"""

    # 컨텍스트 로거 생성
    project_logger = logger.bind(
        project_id=project_data.get('id'),
        user_id=project_data.get('user_id'),
        operation="create_project"
    )

    project_logger.info(
        "프로젝트 생성 시작",
        title=project_data.get('title'),
        complexity=project_data.get('complexity')
    )

    try:
        # 프로젝트 생성 로직
        result = await _create_project_logic(project_data)

        project_logger.info(
            "프로젝트 생성 완료",
            duration_ms=result.get('duration_ms'),
            assets_created=len(result.get('assets', []))
        )

        return result

    except Exception as e:
        project_logger.error(
            "프로젝트 생성 실패",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
```

### 분산 추적 (Distributed Tracing)

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# OpenTelemetry 설정
tracer = trace.get_tracer(__name__)

def setup_tracing(app: FastAPI):
    """분산 추적 설정"""

    # FastAPI 자동 계측
    FastAPIInstrumentor.instrument_app(app)

    # SQLAlchemy 자동 계측
    SQLAlchemyInstrumentor().instrument(engine=engine)

# 수동 스팬 생성
async def process_project_with_tracing(project_id: str):
    """추적이 포함된 프로젝트 처리"""

    with tracer.start_as_current_span("process_project") as span:
        # 스팬에 속성 추가
        span.set_attribute("project.id", project_id)
        span.set_attribute("service.name", "project_maestro")

        # 자식 스팬들
        with tracer.start_as_current_span("analyze_requirements"):
            analysis = await analyze_project_requirements(project_id)
            span.set_attribute("analysis.complexity", analysis.complexity)

        with tracer.start_as_current_span("generate_assets"):
            assets = await generate_project_assets(analysis)
            span.set_attribute("assets.count", len(assets))

        with tracer.start_as_current_span("build_project"):
            build_result = await build_project(assets)
            span.set_attribute("build.success", build_result.success)

        return build_result
```

### 헬스 체크와 알림

```python
from typing import Callable, Dict
from dataclasses import dataclass

@dataclass
class HealthCheckResult:
    """헬스 체크 결과"""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str
    details: Dict[str, Any] = None

class HealthChecker:
    """헬스 체크 관리자"""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable):
        """헬스 체크 등록"""
        self.checks[name] = check_func

    async def run_checks(self) -> Dict[str, HealthCheckResult]:
        """모든 헬스 체크 실행"""
        results = {}

        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status="unhealthy",
                    message=str(e)
                )

        return results

# 헬스 체크 구현
async def database_health_check() -> HealthCheckResult:
    """데이터베이스 헬스 체크"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            return HealthCheckResult(
                name="database",
                status="healthy",
                message="Database connection successful"
            )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status="unhealthy",
            message=f"Database connection failed: {e}"
        )

async def redis_health_check() -> HealthCheckResult:
    """Redis 헬스 체크"""
    try:
        async with redis.Redis.from_url(settings.redis_url) as r:
            await r.ping()
            return HealthCheckResult(
                name="redis",
                status="healthy",
                message="Redis connection successful"
            )
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status="unhealthy",
            message=f"Redis connection failed: {e}"
        )

# FastAPI 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    health_checker = HealthChecker()
    health_checker.register_check("database", database_health_check)
    health_checker.register_check("redis", redis_health_check)

    results = await health_checker.run_checks()

    # 전체 상태 결정
    overall_status = "healthy"
    if any(r.status == "unhealthy" for r in results.values()):
        overall_status = "unhealthy"
    elif any(r.status == "degraded" for r in results.values()):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {name: result.__dict__ for name, result in results.items()}
    }
```

---

## 요약 및 다음 단계

### 이번 단계에서 분석한 아키텍처 패턴

1. **마이크로서비스 아키텍처**: 에이전트별 서비스 분리
2. **이벤트 드리븐**: Redis 기반 비동기 메시징
3. **API Gateway**: FastAPI 기반 통합 엔드포인트
4. **레이어드 아키텍처**: API → 비즈니스 → 데이터 계층
5. **CQRS**: 명령과 쿼리 분리
6. **사가 패턴**: 분산 트랜잭션 관리
7. **관찰가능성**: 메트릭, 로깅, 추적

### 기술 스택 핵심 요소

| 계층 | 기술 | 역할 |
|------|------|------|
| API | FastAPI + Pydantic | REST API, 데이터 검증 |
| 비즈니스 로직 | LangChain + LangGraph | AI 에이전트 오케스트레이션 |
| 데이터 | SQLAlchemy + PostgreSQL | 관계형 데이터 저장 |
| 캐싱 | Redis | 세션, 캐시, 메시지 큐 |
| 파일 저장 | MinIO | 객체 스토리지 |
| 모니터링 | Prometheus + OpenTelemetry | 메트릭, 추적 |
| 태스크 큐 | Celery | 백그라운드 작업 |

### 아키텍처의 장점

- **확장성**: 각 컴포넌트 독립적 스케일링
- **가용성**: 장애 격리 및 회복력
- **유지보수성**: 명확한 책임 분리
- **성능**: 비동기 처리 및 캐싱
- **관찰가능성**: 종합적 모니터링

### 다음 단계 예고

다음 문서에서는 백엔드 개발 베스트 프랙티스를 다룰 예정입니다:
- 테스트 주도 개발 (TDD)
- CI/CD 파이프라인
- 보안 베스트 프랙티스
- 성능 최적화 기법
- 프로덕션 배포 전략

이제 실제 엔터프라이즈급 Python 백엔드 시스템의 구조를 이해했으니, 실무 개발 관행으로 넘어갈 준비가 되었습니다!