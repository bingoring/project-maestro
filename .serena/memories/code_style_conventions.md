# Project Maestro 코드 스타일 가이드

## 코딩 규칙

### Python 스타일
- **PEP 8 준수**: Python 표준 스타일 가이드
- **Black 포맷팅**: 88자 줄 길이, 자동 포맷팅
- **isort**: import 문 정렬 (black profile)
- **Type Hints**: 모든 함수/메서드에 타입 힌트 필수

### 네이밍 컨벤션
- **클래스**: PascalCase (예: `BaseAgent`, `RAGSystem`)
- **함수/변수**: snake_case (예: `process_task`, `vector_store`)
- **상수**: UPPER_SNAKE_CASE (예: `MAX_RETRY_ATTEMPTS`)
- **Private**: 언더스코어 접두사 (예: `_internal_method`)

### 문서화
- **Docstrings**: Google 스타일 docstring
- **타입 힌트**: Pydantic models for data validation
- **코멘트**: 복잡한 로직에만 간결하게

### 프로젝트 구조
```
src/project_maestro/
├── core/                 # 핵심 프레임워크
│   ├── agent_framework.py
│   ├── rag_system.py
│   └── config.py
├── agents/              # AI 에이전트들
│   ├── orchestrator.py
│   ├── codex_agent.py
│   └── ...
├── api/                 # REST API
│   ├── main.py
│   ├── endpoints/
│   └── models.py
└── cli.py              # CLI 인터페이스
```

### 에러 처리
- **구조화된 예외**: 커스텀 예외 클래스 사용
- **로깅**: structlog를 사용한 구조화된 로깅
- **복구 전략**: Circuit breaker, retry logic 활용
- **모니터링**: Prometheus 메트릭 수집

### 테스트 전략
- **단위 테스트**: pytest + 80% 이상 커버리지
- **통합 테스트**: FastAPI TestClient 사용
- **비동기 테스트**: pytest-asyncio 활용
- **Mock 사용**: API 호출, 외부 서비스 mocking