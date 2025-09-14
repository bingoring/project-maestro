# Project Maestro - 개발 가이드

## 개발 환경 설정

### 백엔드 설정
```bash
# Python 가상환경 생성 및 활성화
cd project_maestro
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
cd src
uvicorn project_maestro.main:app --reload --host 0.0.0.0 --port 8000
```

### 프론트엔드 설정
```bash
# 프론트엔드 디렉토리로 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

## 실행 확인

1. 백엔드: http://localhost:8000
2. 프론트엔드: http://localhost:5173 (Vite 기본 포트)
3. WebSocket 연결: ws://localhost:8000/ws/{user_id}

## 주요 기능

### 실시간 에이전트 모니터링
- 워크플로우 진행상황 실시간 업데이트
- D3.js 기반 Sankey 다이어그램
- 에이전트별 상태 및 메트릭 표시

### 프롬프트 인터페이스
- IBM Carbon Design System 스타일링
- 파일 업로드 지원
- 복잡도 실시간 분석

### 대시보드
- 통계 타일
- 연결 상태 표시기
- 실시간 로그 모니터링