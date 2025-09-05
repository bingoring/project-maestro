# Project Maestro - 개발 명령어 가이드

## 테스트 명령어
```bash
# 모든 테스트 실행
python run_tests.py

# 특정 테스트 유형
python run_tests.py --unit          # 단위 테스트
python run_tests.py --integration   # 통합 테스트
python run_tests.py --api           # API 테스트
python run_tests.py --performance   # 성능 테스트
python run_tests.py --lint          # 코드 품질 검사

# 테스트 커버리지
python run_tests.py --report
```

## 코드 품질 명령어
```bash
# 포맷팅
black src/ tests/
isort src/ tests/

# 타입 체킹
mypy src/

# 린팅
flake8 src/ tests/

# 전체 품질 체크
python run_tests.py --lint
```

## 서버 실행 명령어
```bash
# API 서버 (개발 모드)
maestro server start --reload

# 프로덕션 모드
maestro server start --workers 4

# 특정 포트로 실행
maestro server start --port 8080 --host 0.0.0.0
```

## 백그라운드 서비스
```bash
# Celery 워커 시작
celery -A project_maestro.api.main:celery_app worker --loglevel=info

# Redis 서버 시작 (macOS)
redis-server

# PostgreSQL 시작 (macOS)
brew services start postgresql
```

## 개발 유틸리티
```bash
# 프로젝트 초기화
maestro init

# 설정 확인
maestro config

# 에이전트 상태 확인
maestro agent status orchestrator

# 시스템 헬스체크
curl http://localhost:8000/health
```

## macOS 특화 명령어
```bash
# 패키지 관리
brew install redis postgresql

# 프로세스 관리
brew services start/stop/restart redis
brew services start/stop/restart postgresql

# 파일 찾기
find . -name "*.py" -type f
mdfind -name "*.py"

# 로그 확인
log stream --process maestro
```

## Git 워크플로우
```bash
# 기능 브랜치 생성
git checkout -b feature/rag-implementation

# 커밋 및 푸시
git add .
git commit -m "feat: implement RAG system with LCEL"
git push origin feature/rag-implementation
```

## 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 개발 의존성 설치
pip install -e ".[dev,test]"

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집 후

# 데이터베이스 초기화
alembic upgrade head
```