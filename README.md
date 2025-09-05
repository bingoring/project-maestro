# Project Maestro

**AI 에이전트 기반 게임 프로토타이핑 자동화 시스템**

Project Maestro는 AI 에이전트와 Unity 통합을 사용하여 게임 디자인 문서를 실제 게임 프로토타입으로 자동 변환하는 정교한 멀티 에이전트 오케스트레이션 시스템입니다.

[![Tests](https://github.com/your-org/project-maestro/workflows/tests/badge.svg)](https://github.com/your-org/project-maestro/actions)
[![Coverage](https://codecov.io/gh/your-org/project-maestro/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/project-maestro)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 주요 기능

- **지능형 게임 디자인 문서 파싱**: 게임 디자인 요구사항의 고급 NLP 기반 분석
- **멀티 에이전트 오케스트레이션**: 게임 개발의 다양한 측면을 위한 전문 AI 에이전트
- **자동화된 에셋 생성**: AI 기반 코드, 아트, 오디오, 레벨 에셋 생성
- **Unity 통합**: 완벽한 Unity 프로젝트 생성 및 크로스 플랫폼 빌드
- **실시간 모니터링**: 포괄적인 시스템 모니터링 및 성능 분석
- **확장 가능한 아키텍처**: Redis 메시지 큐잉을 활용한 이벤트 기반 마이크로서비스
- **RESTful API**: 통합 및 외부 도구 지원을 위한 완전한 API

## 🏗️ 아키텍처

### 에이전트 유형

#### 게임 개발 에이전트
- **🎭 오케스트레이터** (`gpt-4-turbo-preview`): 전체 워크플로를 조율하는 마스터 에이전트
- **💻 코덱스** (`gpt-4`): C# 게임 코드 생성 및 Unity 스크립팅
- **🎨 캔버스** (`dall-e-3`, `stable-diffusion-xl`): 시각적 에셋 및 아트워크 생성
- **🎵 소나타** (`musicgen-large`, `bark`): 오디오, 음악 및 사운드 이펙트 생성
- **🗺️ 라비린스** (`gpt-4`): 레벨 디자인 및 게임플레이 진행
- **🔨 빌더** (`gpt-3.5-turbo`): Unity 통합 및 크로스 플랫폼 빌드

#### 기업 지식 관리 에이전트
- **🔍 쿼리 에이전트** (`gpt-4-turbo-preview`): 기업 정보 검색 및 복잡도 기반 캐스케이딩 처리
- **🧠 의도 분석기** (`gpt-3.5-turbo`): 사용자 질의 의도 분석 및 라우팅 결정

### 핵심 시스템

- **이벤트 기반 메시징**: Redis 기반 비동기 통신
- **멀티 백엔드 스토리지**: MinIO, S3, 로컬 스토리지 지원
- **포괄적 모니터링**: 실시간 지표, 알림, 헬스체크
- **고급 오류 처리**: 서킷 브레이커, 재시도 메커니즘, 복구 전략

## 📋 요구사항

- **Python**: 3.9 이상
- **Unity**: 2023.2.0f1 이상
- **Redis**: 6.0 이상
- **스토리지**: MinIO, S3, 또는 로컬 파일시스템

### API 키 (선택사항)

- GPT 모델을 위한 OpenAI API 키
- Claude 모델을 위한 Anthropic API 키
- 이미지 생성을 위한 Stable Diffusion API 액세스

## 🛠️ 설치

### 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/your-org/project-maestro.git
cd project-maestro

# 의존성 설치
pip install -e .

# 환경 변수 설정
cp .env.example .env
# 설정을 위해 .env 편집

# 프로젝트 초기화
maestro init

# API 서버 시작
maestro server start
```

### 개발 설정

```bash
# 개발 의존성 설치
pip install -e ".[dev,test]"

# 테스트 실행
python run_tests.py

# 개발 설정으로 시작
export MAESTRO_ENVIRONMENT=development
maestro server start --reload
```

## 🚦 빠른 시작 가이드

### 1. 게임 디자인 문서 생성

게임을 설명하는 마크다운 파일을 생성하세요:

```markdown
# 내 플랫포머 게임

## 게임 개요
- **장르**: 플랫포머
- **플랫폼**: 모바일 (Android/iOS)
- **아트 스타일**: 픽셀 아트

## 핵심 게임플레이
- 플레이어가 좌우 이동과 점프가 가능한 캐릭터를 조작
- 코인을 수집하고 적들을 피함
- 각 레벨의 끝에 도달하기

## 캐릭터
- **히어로**: 점프와 이동 능력을 가진 메인 플레이어 캐릭터
- **굼바**: 좌우로 이동하는 간단한 적

## 레벨
- **레벨 1**: 3개 플랫폼과 2개 적이 있는 녹색 언덕
```

### 2. 게임 생성

CLI 사용:

```bash
maestro project create "내 플랫포머" \
  --document-file game_design.md \
  --description "간단한 플랫포머 게임"
```

API 사용:

```bash
curl -X POST "http://localhost:8000/api/v1/projects/" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "내 플랫포머",
    "description": "간단한 플랫포머 게임",
    "game_design_document": "# 내 게임..."
  }'
```

### 3. 진행 상황 모니터링

```bash
# 프로젝트 상태 확인
maestro project status <project-id>

# 실시간 모니터링
maestro project status <project-id> --watch

# 시스템 지표 보기
curl http://localhost:8000/metrics
```

## 🔧 설정

### 환경 변수

```bash
# 핵심 설정
MAESTRO_ENVIRONMENT=production
MAESTRO_DEBUG=false
MAESTRO_LOG_LEVEL=INFO

# API 설정
MAESTRO_API_HOST=0.0.0.0
MAESTRO_API_PORT=8000
MAESTRO_API_WORKERS=4

# 데이터베이스
MAESTRO_DATABASE_URL=postgresql://user:pass@localhost/maestro

# Redis
MAESTRO_REDIS_URL=redis://localhost:6379/0

# 스토리지
MAESTRO_STORAGE_TYPE=minio  # minio, s3, local
MAESTRO_MINIO_ENDPOINT=localhost:9000
MAESTRO_MINIO_ACCESS_KEY=minioaccess
MAESTRO_MINIO_SECRET_KEY=miniosecret

# AI 서비스
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
STABLE_DIFFUSION_API_KEY=your_sd_key

# Unity
MAESTRO_UNITY_PATH=/Applications/Unity/Hub/Editor/2023.2.0f1
```

### 고급 설정

상세한 설정 옵션은 [docs/configuration.md](docs/configuration.md)를 참조하세요.

## 📖 API 문서

### 핵심 엔드포인트

- `GET /health` - 시스템 헬스체크
- `GET /metrics` - 시스템 지표 및 모니터링
- `POST /api/v1/projects/` - 새 게임 프로젝트 생성
- `GET /api/v1/projects/{id}` - 프로젝트 상세 정보 조회
- `POST /api/v1/builds/` - 게임 빌드 생성
- `GET /api/v1/agents/` - 에이전트 상태 목록

### 대화형 문서

서버를 시작한 후 방문하세요:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 테스트

### 모든 테스트 실행

```bash
python run_tests.py
```

### 특정 테스트 유형 실행

```bash
# 단위 테스트만
python run_tests.py --unit

# 통합 테스트
python run_tests.py --integration

# 성능 테스트
python run_tests.py --performance

# API 테스트
python run_tests.py --api

# 코드 품질 체크
python run_tests.py --lint
```

### 테스트 커버리지

테스트는 80% 이상의 커버리지를 유지합니다. 커버리지 보고서 보기:

```bash
python run_tests.py --report
open test_reports/coverage/index.html
```

## 📊 모니터링

### 시스템 헬스

```bash
# 전체 시스템 헬스 체크
curl http://localhost:8000/health

# 상세 지표 조회
curl http://localhost:8000/metrics

# 에이전트별 지표
curl http://localhost:8000/metrics/agents/orchestrator
```

### 모니터링 대시보드

Project Maestro는 내장 모니터링을 포함합니다:

- **시스템 지표**: CPU, 메모리, 디스크 사용량
- **에이전트 성능**: 작업 완료율, 응답 시간
- **오류 추적**: 포괄적인 오류 분류 및 복구
- **실시간 알림**: 시스템 이슈에 대한 구성 가능한 알림

## 🔍 문제해결

### 일반적인 문제

**에이전트 응답 없음**
```bash
# 에이전트 상태 확인
maestro agent status orchestrator

# 에이전트 재시작
maestro server restart
```

**빌드 실패**
```bash
# Unity 경로 확인
maestro config

# 빌드 로그 보기
curl http://localhost:8000/api/v1/builds/{build-id}/logs
```

**높은 메모리 사용량**
```bash
# 시스템 지표 확인
curl http://localhost:8000/metrics

# 오류 통계 보기
curl http://localhost:8000/api/v1/analytics/errors/summary
```

### 디버그 모드

```bash
export MAESTRO_DEBUG=true
export MAESTRO_LOG_LEVEL=DEBUG
maestro server start
```

## 🤝 기여하기

기여를 환영합니다! 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

### 개발 워크플로

1. **포크** 저장소 포크
2. **브랜치 생성** 기능 브랜치 생성: `git checkout -b feature/amazing-feature`
3. **커밋** 변경사항 커밋: `git commit -m 'Add amazing feature'`
4. **푸시** 브랜치에 푸시: `git push origin feature/amazing-feature`
5. **풀 리퀘스트** 풀 리퀘스트 생성

### 코드 품질 표준

- **테스트**: 모든 새 코드에는 테스트가 포함되어야 함
- **커버리지**: 80% 이상의 테스트 커버리지 유지
- **문서화**: 새 기능에 대한 문서 업데이트
- **코드 스타일**: PEP 8 준수 및 Black 포맷팅 사용

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 라이센스됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **LangChain** 멀티 에이전트 프레임워크 제공
- **Unity** 게임 엔진 통합 지원
- **OpenAI & Anthropic** AI 모델 액세스 제공
- **FastAPI** REST API 프레임워크 제공
- **Redis** 메시지 큐잉 시스템 제공

## 📞 지원

- **문서**: [docs/](docs/)
- **이슈**: [GitHub Issues](https://github.com/your-org/project-maestro/issues)
- **토론**: [GitHub Discussions](https://github.com/your-org/project-maestro/discussions)

## 🗺️ 로드맵

### 버전 1.0 (현재)
- ✅ 멀티 에이전트 오케스트레이션
- ✅ 기본 에셋 생성
- ✅ Unity 통합
- ✅ REST API

### 버전 1.1 (계획)
- 🔄 고급 AI 모델 통합
- 🔄 웹 기반 UI 대시보드
- 🔄 커스텀 에이전트를 위한 플러그인 시스템
- 🔄 향상된 분석 및 리포팅

### 버전 2.0 (미래)
- 📋 비주얼 스크립팅 지원
- 📋 멀티플레이어 게임 템플릿
- 📋 고급 AI 행동 트리
- 📋 클라우드 배포 자동화

---

**Project Maestro 팀이 ❤️로 만들었습니다**