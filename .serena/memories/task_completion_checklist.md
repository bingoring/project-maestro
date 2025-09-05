# 작업 완료 체크리스트

## 코드 수정 후 실행할 명령어

### 1. 코드 품질 검사
```bash
# 포맷팅 확인/적용
black --check src/ tests/  # 확인만
black src/ tests/          # 적용

# Import 정렬
isort --check-only src/ tests/  # 확인만
isort src/ tests/               # 적용

# 타입 체크
mypy src/

# 린팅
flake8 src/ tests/
```

### 2. 테스트 실행
```bash
# 빠른 단위 테스트
python run_tests.py --unit

# 전체 테스트
python run_tests.py

# 특정 모듈 테스트
pytest tests/test_rag_system.py -v
```

### 3. 의존성 확인
```bash
# 새로운 패키지 설치 후
pip install -e ".[dev,test]"

# 의존성 충돌 확인
pip check
```

### 4. 서비스 상태 확인
```bash
# 개발 서버 시작
maestro server start --reload

# 헬스체크
curl http://localhost:8000/health

# API 문서 확인
open http://localhost:8000/docs
```

## 코드 리뷰 체크포인트

### 필수 검토 항목
- [ ] 타입 힌트 완성
- [ ] Docstring 작성
- [ ] 에러 처리 구현
- [ ] 테스트 케이스 작성
- [ ] 로깅 추가
- [ ] 성능 고려사항 검토
- [ ] 보안 이슈 확인

### AI/ML 특화 체크포인트
- [ ] LLM API 키 보안 처리
- [ ] 토큰 사용량 최적화
- [ ] 스트리밍 응답 지원
- [ ] 에러 복구 전략
- [ ] 캐싱 전략 적용
- [ ] 모니터링 메트릭 추가