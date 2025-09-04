# 대화 메모리 시스템

Project Maestro의 대화 메모리 시스템은 사용자와 AI 에이전트 간의 대화를 저장하고 관리하며, 의미론적 검색과 컨텍스트 인식 기능을 제공합니다.

## 목차

- [시스템 개요](#시스템-개요)
- [아키텍처](#아키텍처)
- [핵심 컴포넌트](#핵심-컴포넌트)
- [데이터베이스 스키마](#데이터베이스-스키마)
- [벡터 임베딩](#벡터-임베딩)
- [API 문서](#api-문서)
- [메모리 인식 에이전트](#메모리-인식-에이전트)
- [개인정보 보호](#개인정보-보호)
- [설치 및 설정](#설치-및-설정)
- [사용 예제](#사용-예제)
- [성능 최적화](#성능-최적화)
- [문제 해결](#문제-해결)

## 시스템 개요

### 주요 기능

- **대화 저장 및 관리**: 사용자와 에이전트 간의 모든 대화를 구조적으로 저장
- **의미론적 검색**: 벡터 임베딩을 활용한 지능적 대화 검색
- **컨텍스트 인식**: 이전 대화 내용을 바탕으로 한 맥락적 응답
- **프라이버시 보호**: GDPR 준수 데이터 관리 및 사용자 권리 보장
- **에이전트 통합**: 모든 AI 에이전트가 메모리 기능을 활용 가능

### 시스템 요구사항

- PostgreSQL 12+ (pgvector 확장 지원)
- Python 3.9+
- Redis 6.0+
- 충분한 저장 공간 (임베딩 벡터 저장용)

## 아키텍처

```mermaid
graph TD
    A[사용자] --> B[FastAPI 엔드포인트]
    B --> C[ConversationMemoryManager]
    C --> D[PostgreSQL with pgvector]
    C --> E[EmbeddingManager]
    E --> F[Sentence Transformers]
    
    G[MemoryAwareAgent] --> C
    H[PrivacyPolicyManager] --> D
    
    subgraph "메모리 시스템"
        C
        E
        H
    end
    
    subgraph "데이터 저장"
        D
        I[Redis 캐시]
    end
```

## 핵심 컴포넌트

### 1. ConversationMemoryManager

대화 메모리 시스템의 핵심 매니저로, 다음 기능을 제공합니다:

```python
from project_maestro.core.conversation_memory import get_memory_manager

memory_manager = get_memory_manager()

# 새 대화 생성
conversation = await memory_manager.create_conversation(
    user_id="user123",
    project_id="proj456",
    title="게임 기획 논의"
)

# 메시지 추가
message = await memory_manager.add_message(
    conversation_id=conversation.id,
    message_type=MessageType.USER,
    content="RPG 게임을 만들고 싶어요"
)
```

### 2. EmbeddingManager

텍스트를 벡터로 변환하여 의미론적 검색을 가능하게 합니다:

```python
from project_maestro.core.conversation_memory import EmbeddingManager

embedding_manager = EmbeddingManager()
await embedding_manager.initialize()

# 단일 텍스트 임베딩
embedding = await embedding_manager.encode_single("안녕하세요")

# 여러 텍스트 일괄 임베딩
embeddings = await embedding_manager.encode([
    "첫 번째 메시지",
    "두 번째 메시지"
])
```

### 3. MemoryAwareAgent

메모리 기능을 갖춘 AI 에이전트:

```python
from project_maestro.core.memory_aware_agent import MemoryAwareAgent
from project_maestro.core.agent_framework import AgentType

agent = MemoryAwareAgent(
    name="게임_디자인_에이전트",
    agent_type=AgentType.CODEX,
    memory_enabled=True
)

# 기억하기
await agent.remember(
    user_id="user123",
    content="사용자가 RPG 게임을 선호함",
    conversation_id="conv456"
)

# 회상하기
memories = await agent.recall(
    user_id="user123",
    query="RPG 게임"
)
```

## 데이터베이스 스키마

### 주요 테이블

#### conversations
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    project_id UUID,
    title VARCHAR NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB
);
```

#### conversation_messages
```sql
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    message_type VARCHAR NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding_vector vector(384),  -- pgvector 임베딩
    created_at TIMESTAMP,
    message_order INTEGER
);
```

#### conversation_summaries
```sql
CREATE TABLE conversation_summaries (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    summary_text TEXT NOT NULL,
    summary_type VARCHAR NOT NULL,
    embedding_vector vector(384),
    created_at TIMESTAMP
);
```

## 벡터 임베딩

### 사용 모델

기본적으로 `sentence-transformers/all-MiniLM-L6-v2` 모델을 사용합니다:
- **차원**: 384
- **언어**: 다국어 지원 (한국어 포함)
- **성능**: 빠른 처리 속도와 적절한 품질

### 유사도 검색

```python
# 코사인 유사도 기반 검색
similar_messages = await memory_manager.search_conversations(
    user_id="user123",
    query="게임 개발",
    limit=10
)

# SQL 함수 직접 사용
cursor.execute("""
    SELECT * FROM find_similar_messages(
        %s,  -- query_vector
        %s,  -- user_id
        NULL,  -- conversation_id (전체 검색)
        0.7,   -- similarity_threshold
        10     -- result_limit
    )
""", (query_embedding, user_id))
```

## API 문서

### 대화 관리 API

#### POST /api/v1/conversations/
새 대화 생성
```json
{
    "title": "게임 기획 회의",
    "project_id": "uuid",
    "metadata": {}
}
```

#### GET /api/v1/conversations/
사용자의 대화 목록 조회
- 파라미터: `project_id`, `limit`, `offset`

#### POST /api/v1/conversations/{id}/messages
대화에 메시지 추가
```json
{
    "message_type": "user",
    "content": "RPG 게임을 만들고 싶어요",
    "metadata": {}
}
```

#### POST /api/v1/conversations/search
대화 내용 검색
```json
{
    "query": "게임 개발",
    "project_id": "uuid",
    "limit": 20
}
```

### 개인정보 관리 API

#### POST /api/v1/privacy/consent
사용자 동의 관리
```json
{
    "consent_type": "functional",
    "granted": true,
    "version": "1.0"
}
```

#### POST /api/v1/privacy/delete-request
데이터 삭제 요청
```json
{
    "request_type": "delete_all",
    "data_categories": ["conversation", "user_profile"],
    "reason": "계정 탈퇴"
}
```

## 메모리 인식 에이전트

### 기본 사용법

```python
# 메모리 인식 에이전트 생성
agent = MemoryAwareAgent(
    name="코드_생성_에이전트",
    agent_type=AgentType.CODEX,
    memory_enabled=True,
    context_window_size=20
)

# 메모리 컨텍스트와 함께 작업 실행
task = AgentTask(
    agent_type=AgentType.CODEX,
    action="generate_code",
    parameters={
        "language": "python",
        "requirements": "게임 인벤토리 시스템"
    }
)

result = await agent.execute_task_with_memory(
    task=task,
    user_id="user123",
    conversation_id="conv456"
)
```

### 대화 세션 관리

```python
# 대화 세션 시작
session = ConversationSession(
    user_id="user123",
    project_id="proj456"
)

conversation_id = await session.start_session("코드 리뷰")

# 사용자 메시지 추가
await session.add_user_message("이 코드를 검토해주세요")

# 에이전트 응답 추가
await session.add_agent_response(
    agent_name="code_reviewer",
    content="코드를 검토한 결과...",
    metadata={"review_score": 8.5}
)

# 대화 히스토리 조회
history = await session.get_conversation_history(limit=50)

# 세션 종료
await session.end_session()
```

## 개인정보 보호

### GDPR 준수 기능

1. **데이터 최소화**: 필요한 데이터만 수집
2. **명시적 동의**: 사용자 동의 관리 시스템
3. **잊혀질 권리**: 데이터 삭제 요청 처리
4. **데이터 이동권**: 사용자 데이터 내보내기
5. **투명성**: 데이터 사용 현황 공개

### 데이터 보존 정책

```python
from project_maestro.core.privacy_policy import (
    get_privacy_manager,
    DataCategory,
    RetentionPeriod
)

privacy_manager = get_privacy_manager()

# 사용자 동의 처리
await privacy_manager.grant_consent(
    user_id="user123",
    consent_type=ConsentType.FUNCTIONAL,
    granted=True
)

# 데이터 내보내기
export_result = await privacy_manager.export_user_data("user123")

# 데이터 삭제 요청
request_id = await privacy_manager.request_data_deletion(
    user_id="user123",
    request_type="delete_all",
    data_categories=[DataCategory.CONVERSATION]
)
```

## 설치 및 설정

### 1. 데이터베이스 설정

```bash
# pgvector 확장 설치 (PostgreSQL)
CREATE EXTENSION vector;

# 마이그레이션 실행
cd database
python run_migrations.py migrate
```

### 2. Python 의존성 설치

```bash
pip install sentence-transformers psycopg2-binary numpy
```

### 3. 환경변수 설정

```bash
# .env 파일
DATABASE_URL=postgresql://user:pass@localhost:5432/maestro
OPENAI_API_KEY=your_openai_key  # 선택사항
ANTHROPIC_API_KEY=your_anthropic_key  # 선택사항
```

### 4. 시스템 상태 확인

```bash
# 데이터베이스 및 pgvector 확인
python database/run_migrations.py check

# 마이그레이션 상태 확인
python database/run_migrations.py status
```

## 사용 예제

### 기본 대화 시나리오

```python
import asyncio
from project_maestro.core.conversation_memory import get_memory_manager
from project_maestro.core.memory_aware_agent import ConversationSession

async def conversation_example():
    # 1. 대화 세션 시작
    session = ConversationSession(user_id="developer123")
    conversation_id = await session.start_session("게임 개발 상담")
    
    # 2. 사용자 질문
    await session.add_user_message(
        "Unity에서 인벤토리 시스템을 어떻게 만들 수 있을까요?"
    )
    
    # 3. AI 응답 (실제로는 에이전트가 생성)
    await session.add_agent_response(
        agent_name="unity_expert",
        content="Unity 인벤토리 시스템을 만들 때는...",
        metadata={
            "code_examples": ["Inventory.cs", "Item.cs"],
            "difficulty": "intermediate"
        }
    )
    
    # 4. 후속 질문
    await session.add_user_message("UI는 어떻게 구성하나요?")
    
    # 5. 대화 히스토리 조회
    history = await session.get_conversation_history()
    print(f"대화 메시지 수: {len(history)}")
    
    # 6. 관련 이전 대화 검색
    memory_manager = get_memory_manager()
    similar_conversations = await memory_manager.search_conversations(
        user_id="developer123",
        query="Unity 인벤토리",
        limit=5
    )
    
    print(f"관련 대화 {len(similar_conversations)}개 발견")
    
    # 7. 세션 종료
    await session.end_session()
    
    return conversation_id

# 실행
conversation_id = asyncio.run(conversation_example())
```

### 메모리 인식 에이전트 활용

```python
from project_maestro.core.memory_aware_agent import MemoryAwareAgent
from project_maestro.core.agent_framework import AgentType, AgentTask

async def memory_agent_example():
    # 메모리 기능이 있는 에이전트 생성
    agent = MemoryAwareAgent(
        name="게임_기획_어시스턴트",
        agent_type=AgentType.ORCHESTRATOR,
        memory_enabled=True
    )
    
    user_id = "game_designer_001"
    
    # 1. 사용자 선호사항 기억
    await agent.remember(
        user_id=user_id,
        content="사용자는 RPG 장르를 선호하며, 픽셀 아트 스타일을 좋아함",
        message_type=MessageType.SYSTEM
    )
    
    # 2. 프로젝트 정보 기억
    await agent.remember(
        user_id=user_id,
        content="현재 프로젝트: 2D RPG '마법사의 모험', 타겟: 모바일",
        message_type=MessageType.AGENT
    )
    
    # 3. 이전 기억 회상
    memories = await agent.recall(
        user_id=user_id,
        query="RPG 게임 선호사항"
    )
    
    print(f"회상된 기억: {len(memories)}개")
    
    # 4. 메모리 컨텍스트와 함께 작업 수행
    task = AgentTask(
        agent_type=AgentType.ORCHESTRATOR,
        action="create_game_concept",
        parameters={
            "genre": "RPG",
            "platform": "mobile"
        }
    )
    
    # 메모리를 활용한 작업 실행
    result = await agent.execute_task_with_memory(
        task=task,
        user_id=user_id
    )
    
    return result

# 실행
result = asyncio.run(memory_agent_example())
```

## 성능 최적화

### 1. 벡터 인덱스 최적화

```sql
-- 인덱스 통계 업데이트
ANALYZE conversation_messages;

-- 인덱스 재구성 (데이터 증가시)
REINDEX INDEX idx_conversation_messages_embedding_cosine;

-- 인덱스 성능 모니터링
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%embedding%';
```

### 2. 임베딩 생성 최적화

```python
# 배치 임베딩 생성 (더 효율적)
texts = ["메시지 1", "메시지 2", "메시지 3"]
embeddings = await embedding_manager.encode(texts)

# 백그라운드 임베딩 생성
import asyncio

async def generate_embeddings_batch(message_ids: List[str]):
    # 백그라운드에서 임베딩 생성
    for message_id in message_ids:
        # 메시지 내용 조회
        # 임베딩 생성 및 저장
        pass

# 비동기 실행
asyncio.create_task(generate_embeddings_batch(pending_message_ids))
```

### 3. 캐싱 전략

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_embedding(expire_time=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(text: str):
            # 캐시 키 생성
            cache_key = f"embedding:{hash(text)}"
            
            # 캐시에서 조회
            cached = redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
            
            # 임베딩 생성
            result = await func(text)
            
            # 캐시에 저장
            redis_client.setex(
                cache_key, 
                expire_time, 
                pickle.dumps(result)
            )
            
            return result
        return wrapper
    return decorator
```

### 4. 데이터베이스 최적화

```sql
-- 연결 풀링 설정
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';

-- 파티셔닝 (데이터가 많은 경우)
CREATE TABLE conversation_messages_2024 
PARTITION OF conversation_messages 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- 자동 vacuum 설정
ALTER TABLE conversation_messages 
SET (autovacuum_vacuum_scale_factor = 0.1);
```

## 문제 해결

### 1. pgvector 설치 문제

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# macOS (Homebrew)
brew install pgvector

# Docker
docker run -p 5432:5432 -e POSTGRES_PASSWORD=password ankane/pgvector
```

### 2. 임베딩 생성 속도 문제

```python
# GPU 가속 사용 (CUDA 사용 가능시)
embedding_manager = EmbeddingManager(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # 또는 "cpu"
)

# 배치 크기 조정
embeddings = await embedding_manager.encode(
    texts,
    batch_size=32  # 메모리에 따라 조정
)
```

### 3. 메모리 사용량 문제

```python
# 메모리 모니터링
import psutil

def check_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")

# 임베딩 모델 메모리 해제
embedding_manager.model = None
import gc
gc.collect()
```

### 4. 검색 성능 문제

```sql
-- 검색 성능 분석
EXPLAIN ANALYZE 
SELECT * FROM find_similar_messages(
    '[0.1, 0.2, ...]'::vector(384),
    'user123',
    NULL,
    0.7,
    10
);

-- 인덱스 사용량 확인
SELECT 
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_tup_read / NULLIF(idx_tup_fetch, 0) as ratio
FROM pg_stat_user_indexes 
WHERE tablename = 'conversation_messages';
```

### 5. 동시성 문제

```python
import asyncio
from asyncio import Semaphore

# 동시 임베딩 생성 제한
embedding_semaphore = Semaphore(5)  # 최대 5개 동시 처리

async def generate_embedding_with_limit(text: str):
    async with embedding_semaphore:
        return await embedding_manager.encode_single(text)
```

## 모니터링 및 로깅

### 1. 메트릭 수집

```python
from prometheus_client import Counter, Histogram, Gauge

# 메트릭 정의
conversation_counter = Counter(
    'conversations_created_total',
    'Total number of conversations created'
)

message_counter = Counter(
    'messages_added_total',
    'Total number of messages added',
    ['message_type']
)

embedding_histogram = Histogram(
    'embedding_generation_duration_seconds',
    'Time spent generating embeddings'
)

memory_usage_gauge = Gauge(
    'memory_system_memory_usage_bytes',
    'Memory usage of the conversation memory system'
)

# 메트릭 업데이트
conversation_counter.inc()
message_counter.labels(message_type='user').inc()
embedding_histogram.observe(embedding_time)
memory_usage_gauge.set(current_memory_usage)
```

### 2. 구조화된 로깅

```python
import structlog

logger = structlog.get_logger("conversation_memory")

# 대화 생성 로깅
logger.info(
    "conversation_created",
    user_id=user_id,
    conversation_id=conversation.id,
    project_id=project_id,
    title=conversation.title
)

# 임베딩 생성 로깅
logger.info(
    "embedding_generated",
    message_id=message_id,
    embedding_dimensions=384,
    generation_time=elapsed_time
)

# 검색 성능 로깅
logger.info(
    "similarity_search_performed",
    user_id=user_id,
    query=query,
    results_count=len(results),
    search_time=search_elapsed_time,
    similarity_threshold=threshold
)
```

## 보안 고려사항

### 1. 데이터 암호화

```python
from cryptography.fernet import Fernet

# 민감한 데이터 암호화
key = Fernet.generate_key()
cipher = Fernet(key)

# 메시지 내용 암호화 (필요시)
encrypted_content = cipher.encrypt(content.encode())

# 복호화
decrypted_content = cipher.decrypt(encrypted_content).decode()
```

### 2. 접근 제어

```python
from fastapi import Depends, HTTPException
from jose import JWTError, jwt

async def verify_user_access(
    conversation_id: str,
    current_user: str = Depends(get_current_user)
):
    """사용자의 대화 접근 권한 확인"""
    conversation = await memory_manager.get_conversation(
        conversation_id, current_user
    )
    if not conversation:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    return conversation
```

### 3. 감사 로깅

```python
async def audit_log(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: Dict[str, Any] = None
):
    """보안 감사 로그"""
    audit_entry = {
        "timestamp": datetime.utcnow(),
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
        "ip_address": request.client.host,  # FastAPI request 객체
        "user_agent": request.headers.get("user-agent")
    }
    
    # 감사 로그 저장
    await save_audit_log(audit_entry)
```

---

## 참고 자료

- [pgvector 공식 문서](https://github.com/pgvector/pgvector)
- [Sentence Transformers 문서](https://www.sbert.net/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [GDPR 가이드라인](https://gdpr.eu/)

## 라이센스

이 시스템은 MIT 라이센스 하에 배포됩니다.