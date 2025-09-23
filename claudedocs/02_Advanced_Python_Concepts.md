# Python 기초부터 마스터까지 - 2단계: 고급 Python 개념

## 목차
1. [Python 실행 모델과 GIL](#python-실행-모델과-gil)
2. [비동기 프로그래밍 (AsyncIO)](#비동기-프로그래밍-asyncio)
3. [병렬 처리 (Threading vs Multiprocessing)](#병렬-처리-threading-vs-multiprocessing)
4. [예외 처리와 오류 관리](#예외-처리와-오류-관리)
5. [의존성 주입과 추상화](#의존성-주입과-추상화)
6. [고급 객체지향 패턴](#고급-객체지향-패턴)
7. [프로젝트 예시로 보는 실제 활용](#프로젝트-예시로-보는-실제-활용)

---

## Python 실행 모델과 GIL

### Python의 실행 방식

Python은 **인터프리터 언어**로, JavaScript의 V8 엔진과는 다른 방식으로 동작합니다:

```python
# Python 코드 실행 과정
# 1. 소스 코드 (.py) → 바이트코드 (.pyc) → Python 가상머신(PVM)

# test.py
def hello():
    print("Hello, World!")

hello()

# 실행: python test.py
# 1. test.py를 읽어서 바이트코드로 컴파일
# 2. __pycache__/test.cpython-311.pyc 생성 (캐시)
# 3. PVM에서 바이트코드 실행
```

### GIL (Global Interpreter Lock)

**GIL은 Python의 핵심 특징이자 제약사항입니다:**

```python
import threading
import time

# GIL 때문에 이 코드는 실제로 병렬 실행되지 않음
def cpu_intensive_task(n):
    """CPU 집약적 작업 - GIL로 인해 순차 실행됨"""
    result = 0
    for i in range(n):
        result += i * i
    return result

# 스레드로 실행해도 성능 향상 없음
start_time = time.time()

threads = []
for i in range(4):
    thread = threading.Thread(target=cpu_intensive_task, args=(1000000,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"실행 시간: {time.time() - start_time:.2f}초")
```

**GIL의 영향:**
- **CPU 집약적 작업**: 스레딩으로 성능 향상 불가
- **I/O 집약적 작업**: GIL이 해제되어 스레딩 효과적
- **해결책**: Multiprocessing, AsyncIO, C 확장 사용

### Node.js와의 비교

| 특징 | Python (GIL) | Node.js (Event Loop) |
|------|---------------|----------------------|
| 동시성 모델 | GIL + 스레드 | 단일 스레드 + 이벤트 루프 |
| CPU 집약적 작업 | Multiprocessing 필요 | Worker Threads 필요 |
| I/O 작업 | 스레드 풀 사용 | 비동기 이벤트 기반 |
| 메모리 공유 | 제한적 (프로세스 간) | 공유 (단일 프로세스) |

---

## 비동기 프로그래밍 (AsyncIO)

### AsyncIO 기본 개념

AsyncIO는 Python의 **협력적 멀티태스킹** 시스템입니다:

```python
import asyncio
import aiohttp
import time

# 동기 버전 (느림)
def sync_fetch_data(url):
    """동기 HTTP 요청 - 블로킹"""
    import requests
    response = requests.get(url)
    return response.text

# 비동기 버전 (빠름)
async def async_fetch_data(url):
    """비동기 HTTP 요청 - 논블로킹"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# 여러 요청을 동시에 처리
async def fetch_multiple_urls():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]

    # 모든 요청을 동시에 시작
    tasks = [async_fetch_data(url) for url in urls]

    # 모든 요청이 완료될 때까지 대기
    results = await asyncio.gather(*tasks)

    return results

# 실행
async def main():
    start = time.time()
    results = await fetch_multiple_urls()
    print(f"비동기 실행 시간: {time.time() - start:.2f}초")  # 약 2초

# 이벤트 루프 실행
asyncio.run(main())
```

### 프로젝트에서의 AsyncIO 활용

```python
# src/project_maestro/core/distributed_workflow.py에서 발췌
import asyncio
import aioredis
import aiohttp

class DistributedWorkflowManager:
    """분산 워크플로우 관리자"""

    def __init__(self):
        self.redis_pool = None
        self.node_sessions = {}

    async def initialize(self):
        """비동기 초기화"""
        # Redis 연결 풀 생성 (비동기)
        self.redis_pool = aioredis.ConnectionPool.from_url(
            "redis://localhost:6379"
        )

        # 각 노드에 대한 HTTP 세션 생성
        for node_id in self.nodes:
            self.node_sessions[node_id] = aiohttp.ClientSession()

    async def execute_workflow(self, workflow_id: str):
        """워크플로우 비동기 실행"""
        # 여러 태스크를 동시에 시작
        tasks = []
        for task in workflow.tasks:
            task_coroutine = self.execute_task(task)
            tasks.append(task_coroutine)

        # 모든 태스크 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def execute_task(self, task):
        """개별 태스크 실행"""
        # 적절한 노드 선택
        node = await self.select_best_node(task)

        # 해당 노드에 HTTP 요청
        async with self.node_sessions[node.id] as session:
            async with session.post(f"{node.url}/execute", json=task.data) as response:
                return await response.json()
```

### 이벤트 루프와 코루틴

```python
# 이벤트 루프 동작 원리
async def demonstrate_event_loop():
    print("1. 첫 번째 태스크 시작")

    # await 지점에서 다른 태스크로 제어권 넘김
    await asyncio.sleep(1)

    print("3. 첫 번째 태스크 재개")

async def another_task():
    print("2. 두 번째 태스크 실행")
    await asyncio.sleep(0.5)
    print("4. 두 번째 태스크 완료")

# 동시 실행
async def main():
    await asyncio.gather(
        demonstrate_event_loop(),
        another_task()
    )

# 출력 순서: 1 → 2 → 4 → 3
```

---

## 병렬 처리 (Threading vs Multiprocessing)

### Threading: I/O 집약적 작업에 적합

```python
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor

def fetch_url(url):
    """I/O 집약적 작업 - 스레딩 효과적"""
    response = requests.get(url)
    return f"{url}: {response.status_code}"

# 방법 1: 수동 스레드 관리
def threading_example():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]

    threads = []
    results = {}

    def worker(url):
        results[url] = fetch_url(url)

    # 스레드 생성 및 시작
    for url in urls:
        thread = threading.Thread(target=worker, args=(url,))
        threads.append(thread)
        thread.start()

    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()

    return results

# 방법 2: ThreadPoolExecutor (권장)
def threadpool_example():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        # 모든 작업을 제출하고 Future 객체 받기
        futures = [executor.submit(fetch_url, url) for url in urls]

        # 결과 수집
        results = [future.result() for future in futures]

    return results
```

### Multiprocessing: CPU 집약적 작업에 적합

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math

def cpu_intensive_task(n):
    """CPU 집약적 작업 - 멀티프로세싱 효과적"""
    result = 0
    for i in range(n):
        result += math.sqrt(i)
    return result

# 방법 1: Process 클래스
def multiprocessing_example():
    processes = []
    manager = multiprocessing.Manager()
    results = manager.list()

    def worker(n, results, index):
        result = cpu_intensive_task(n)
        results.append((index, result))

    # 프로세스 생성 및 시작
    for i in range(4):
        process = multiprocessing.Process(
            target=worker,
            args=(1000000, results, i)
        )
        processes.append(process)
        process.start()

    # 모든 프로세스 완료 대기
    for process in processes:
        process.join()

    return list(results)

# 방법 2: ProcessPoolExecutor (권장)
def processpool_example():
    tasks = [1000000, 2000000, 1500000, 1200000]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_intensive_task, task) for task in tasks]
        results = [future.result() for future in futures]

    return results
```

### 프로젝트에서의 병렬 처리 활용

```python
# src/project_maestro/core/intelligent_cache.py에서 발췌
import threading
from concurrent.futures import ThreadPoolExecutor

class IntelligentCache:
    def __init__(self):
        self._lock = threading.RLock()  # 재진입 가능한 락
        self._cache = {}
        self._stats = defaultdict(int)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def get_parallel(self, keys: List[str]) -> Dict[str, Any]:
        """여러 키를 병렬로 조회"""
        with ThreadPoolExecutor(max_workers=len(keys)) as executor:
            # 각 키에 대해 별도 스레드에서 조회
            futures = {
                executor.submit(self.get, key): key
                for key in keys
            }

            results = {}
            for future in futures:
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    self.logger.error(f"키 조회 실패 {key}: {e}")
                    results[key] = None

        return results

    def _background_cleanup(self):
        """백그라운드 스레드에서 캐시 정리"""
        def cleanup_worker():
            while self._running:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_entries()

        cleanup_thread = threading.Thread(
            target=cleanup_worker,
            daemon=True  # 메인 프로세스 종료 시 함께 종료
        )
        cleanup_thread.start()
```

---

## 예외 처리와 오류 관리

### 기본 예외 처리

```python
# 기본 try-except 구조
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다!")
        return None
    except TypeError:
        print("숫자가 아닌 값입니다!")
        return None
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        return None
    finally:
        print("연산 시도 완료")  # 항상 실행

# 사용 예시
print(safe_divide(10, 2))    # 5.0
print(safe_divide(10, 0))    # None (ZeroDivisionError)
print(safe_divide(10, "a"))  # None (TypeError)
```

### 커스텀 예외 클래스

```python
# 프로젝트별 예외 정의
class ProjectMaestroError(Exception):
    """프로젝트 마에스트로 기본 예외"""
    pass

class WorkflowError(ProjectMaestroError):
    """워크플로우 관련 예외"""
    def __init__(self, workflow_id, message):
        self.workflow_id = workflow_id
        self.message = message
        super().__init__(f"워크플로우 {workflow_id}: {message}")

class NodeConnectionError(ProjectMaestroError):
    """노드 연결 예외"""
    def __init__(self, node_id, reason):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"노드 {node_id} 연결 실패: {reason}")

# 사용 예시
def execute_workflow(workflow_id):
    try:
        if not workflow_id:
            raise WorkflowError(workflow_id, "워크플로우 ID가 비어있습니다")

        # 워크플로우 실행 로직
        pass

    except WorkflowError as e:
        logger.error(f"워크플로우 오류: {e}")
        # 특정 워크플로우 오류 처리
    except NodeConnectionError as e:
        logger.error(f"노드 연결 오류: {e}")
        # 노드 재연결 시도
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise  # 상위로 예외 전파
```

### 비동기 예외 처리

```python
import asyncio
import aiohttp

async def async_exception_handling():
    """비동기 함수에서의 예외 처리"""

    async def fetch_with_retry(url, max_retries=3):
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )

            except asyncio.TimeoutError:
                logger.warning(f"시도 {attempt + 1}: 타임아웃")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 지수 백오프

            except aiohttp.ClientError as e:
                logger.error(f"시도 {attempt + 1}: 클라이언트 오류 {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

    # 여러 URL을 동시에 처리하면서 예외 관리
    urls = ["https://api1.com", "https://api2.com", "https://invalid-url"]

    tasks = [fetch_with_retry(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과와 예외 구분 처리
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"URL {urls[i]} 처리 실패: {result}")
        else:
            logger.info(f"URL {urls[i]} 처리 성공")
```

### 프로젝트에서의 오류 관리 패턴

```python
# src/project_maestro/core/monitoring.py에서 발췌
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    name: str
    level: AlertLevel
    message: str
    threshold: float
    current_value: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

class MonitoringSystem:
    def __init__(self):
        self.alerts = {}
        self.handlers = {
            AlertLevel.INFO: self._log_info,
            AlertLevel.WARNING: self._log_warning,
            AlertLevel.ERROR: self._handle_error,
            AlertLevel.CRITICAL: self._handle_critical
        }

    def check_alert(self, metric_name: str, value: float):
        """메트릭 값을 확인하고 필요시 알림 발생"""
        try:
            alert = self.alerts.get(metric_name)
            if not alert:
                return

            if value >= alert.threshold:
                alert.current_value = value
                handler = self.handlers.get(alert.level)
                if handler:
                    handler(alert)

        except Exception as e:
            # 모니터링 시스템 자체의 오류는 로그만 남기고 서비스에 영향 안줌
            logger.error(f"알림 확인 중 오류: {e}")

    def _handle_critical(self, alert: Alert):
        """중요 알림 처리 - 즉시 대응 필요"""
        # 1. 로그 기록
        logger.critical(f"CRITICAL: {alert.message}")

        # 2. 외부 알림 시스템 호출 (이메일, 슬랙 등)
        self._send_notification(alert)

        # 3. 자동 복구 시도
        self._attempt_auto_recovery(alert)

    def _attempt_auto_recovery(self, alert: Alert):
        """자동 복구 시도"""
        try:
            if alert.name == "high_memory_usage":
                self._trigger_memory_cleanup()
            elif alert.name == "node_disconnected":
                self._attempt_node_reconnection(alert.tags.get("node_id"))

        except Exception as e:
            logger.error(f"자동 복구 실패: {e}")
```

---

## 의존성 주입과 추상화

### 의존성 주입 기본 개념

**의존성 주입(DI)**은 객체가 필요한 의존성을 외부에서 주입받는 패턴입니다:

```python
from abc import ABC, abstractmethod

# 잘못된 예시 - 강한 결합
class EmailService:
    def send_email(self, to, subject, body):
        # Gmail SMTP 설정 하드코딩
        import smtplib
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        # 이메일 전송 로직...

class UserService:
    def __init__(self):
        self.email_service = EmailService()  # 직접 생성 - 나쁨!

    def register_user(self, user_data):
        # 사용자 등록 로직...
        self.email_service.send_email(
            user_data['email'],
            "환영합니다",
            "가입을 환영합니다!"
        )

# 올바른 예시 - 의존성 주입
class EmailServiceInterface(ABC):
    """이메일 서비스 인터페이스"""

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool:
        pass

class GmailService(EmailServiceInterface):
    """Gmail 구현체"""

    def send_email(self, to: str, subject: str, body: str) -> bool:
        # Gmail 특화 로직
        print(f"Gmail로 전송: {to}")
        return True

class SlackService(EmailServiceInterface):
    """Slack 구현체"""

    def send_email(self, to: str, subject: str, body: str) -> bool:
        # Slack 특화 로직
        print(f"Slack으로 전송: {to}")
        return True

class UserService:
    def __init__(self, email_service: EmailServiceInterface):
        self.email_service = email_service  # 의존성 주입!

    def register_user(self, user_data):
        # 사용자 등록 로직...
        success = self.email_service.send_email(
            user_data['email'],
            "환영합니다",
            "가입을 환영합니다!"
        )
        return success

# 사용법 - 런타임에 구현체 선택 가능
gmail_service = GmailService()
user_service = UserService(gmail_service)  # Gmail 사용

slack_service = SlackService()
user_service = UserService(slack_service)  # Slack 사용
```

### 고급 DI 패턴: DI 컨테이너

```python
class DIContainer:
    """간단한 의존성 주입 컨테이너"""

    def __init__(self):
        self._services = {}
        self._singletons = {}

    def register(self, interface, implementation, singleton=False):
        """서비스 등록"""
        self._services[interface] = {
            'implementation': implementation,
            'singleton': singleton
        }

    def get(self, interface):
        """서비스 인스턴스 반환"""
        if interface not in self._services:
            raise ValueError(f"서비스 {interface}가 등록되지 않았습니다")

        config = self._services[interface]

        if config['singleton']:
            if interface not in self._singletons:
                self._singletons[interface] = self._create_instance(config['implementation'])
            return self._singletons[interface]
        else:
            return self._create_instance(config['implementation'])

    def _create_instance(self, implementation_class):
        """의존성을 자동으로 해결하여 인스턴스 생성"""
        import inspect

        # 생성자 매개변수 분석
        sig = inspect.signature(implementation_class.__init__)
        args = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # 매개변수 타입을 기반으로 의존성 해결
            param_type = param.annotation
            if param_type in self._services:
                args.append(self.get(param_type))

        return implementation_class(*args)

# 사용 예시
container = DIContainer()

# 서비스 등록
container.register(EmailServiceInterface, GmailService, singleton=True)

# 자동 의존성 해결
user_service = container.get(UserService)  # EmailService가 자동 주입됨
```

### 프로젝트에서의 추상화 활용

```python
# src/project_maestro/core/intelligent_cache.py에서 발췌
from abc import ABC, abstractmethod

class CacheInterface(ABC):
    """캐시 인터페이스 정의"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """키로 값 조회"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """키-값 저장"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """키 삭제"""
        pass

class MemoryCache(CacheInterface):
    """메모리 캐시 구현"""

    def __init__(self):
        self._cache = {}
        self._ttl_info = {}

    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            # TTL 확인
            if self._is_expired(key):
                await self.delete(key)
                return None
            return self._cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._cache[key] = value
        if ttl:
            self._ttl_info[key] = time.time() + ttl
        return True

class RedisCache(CacheInterface):
    """Redis 캐시 구현"""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        data = await self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        data = pickle.dumps(value)
        if ttl:
            await self.redis.setex(key, ttl, data)
        else:
            await self.redis.set(key, data)
        return True

# 계층화된 캐시 구현
class TieredCache(CacheInterface):
    """다층 캐시 시스템"""

    def __init__(self, memory_cache: CacheInterface, redis_cache: CacheInterface):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache

    async def get(self, key: str) -> Optional[Any]:
        # 1차: 메모리 캐시 확인
        value = await self.memory_cache.get(key)
        if value is not None:
            return value

        # 2차: Redis 캐시 확인
        value = await self.redis_cache.get(key)
        if value is not None:
            # 메모리 캐시에 다시 저장 (승격)
            await self.memory_cache.set(key, value, ttl=300)
            return value

        return None
```

---

## 고급 객체지향 패턴

### 추상 기본 클래스 (ABC)

```python
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List

class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """에이전트 초기화 - 반드시 구현해야 함"""
        pass

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """에이전트 실행 - 반드시 구현해야 함"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """에이전트 능력 목록 반환 - 반드시 구현해야 함"""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """에이전트 타입 - 반드시 구현해야 함"""
        pass

    # 공통 기능은 기본 구현 제공
    def is_initialized(self) -> bool:
        return self._initialized

    async def cleanup(self) -> None:
        """리소스 정리 - 선택적으로 오버라이드"""
        pass

# 구체적 구현 클래스
class CodeGeneratorAgent(BaseAgent):
    """코드 생성 에이전트"""

    @property
    def agent_type(self) -> str:
        return "code_generator"

    async def initialize(self) -> bool:
        # 언어 모델 초기화, API 키 확인 등
        self._language_model = await self._setup_language_model()
        self._initialized = True
        return True

    async def execute(self, input_data: Any) -> Any:
        if not self.is_initialized():
            raise RuntimeError("에이전트가 초기화되지 않았습니다")

        # 코드 생성 로직
        prompt = input_data.get('prompt', '')
        language = input_data.get('language', 'python')

        generated_code = await self._generate_code(prompt, language)
        return {
            'code': generated_code,
            'language': language,
            'agent': self.name
        }

    def get_capabilities(self) -> List[str]:
        return [
            "python_code_generation",
            "javascript_code_generation",
            "code_optimization",
            "documentation_generation"
        ]
```

### 데코레이터 패턴

```python
import functools
import time
import asyncio
from typing import Callable, Any

def timeit(func: Callable) -> Callable:
    """함수 실행 시간을 측정하는 데코레이터"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 실행 시간: {end - start:.4f}초")
        return result

    return wrapper

def async_timeit(func: Callable) -> Callable:
    """비동기 함수 실행 시간을 측정하는 데코레이터"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 실행 시간: {end - start:.4f}초")
        return result

    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """함수 재시도 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # 지수 백오프
                    else:
                        raise last_exception

        return wrapper
    return decorator

# 사용 예시
class APIClient:
    @async_timeit
    @retry(max_attempts=3, delay=1.0)
    async def fetch_data(self, url: str) -> dict:
        """API에서 데이터 조회"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                return await response.json()
```

### 팩토리 패턴

```python
from typing import Type, Dict
from enum import Enum

class AgentType(Enum):
    CODE_GENERATOR = "code_generator"
    DOCUMENT_PARSER = "document_parser"
    ASSET_CREATOR = "asset_creator"
    QUALITY_CHECKER = "quality_checker"

class AgentFactory:
    """에이전트 팩토리 클래스"""

    _agent_classes: Dict[AgentType, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_type: AgentType, agent_class: Type[BaseAgent]):
        """에이전트 클래스 등록"""
        cls._agent_classes[agent_type] = agent_class

    @classmethod
    def create_agent(cls, agent_type: AgentType, name: str, **kwargs) -> BaseAgent:
        """에이전트 인스턴스 생성"""
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"등록되지 않은 에이전트 타입: {agent_type}")

        return agent_class(name=name, **kwargs)

    @classmethod
    def get_available_types(cls) -> List[AgentType]:
        """사용 가능한 에이전트 타입 목록"""
        return list(cls._agent_classes.keys())

# 에이전트 클래스들 등록
AgentFactory.register(AgentType.CODE_GENERATOR, CodeGeneratorAgent)
AgentFactory.register(AgentType.DOCUMENT_PARSER, DocumentParserAgent)

# 사용법
agent = AgentFactory.create_agent(
    AgentType.CODE_GENERATOR,
    name="python_generator",
    model="gpt-4"
)
```

---

## 프로젝트 예시로 보는 실제 활용

### 1. 분산 워크플로우 시스템

```python
# src/project_maestro/core/distributed_workflow.py 분석

class DistributedWorkflowManager:
    """실제 프로젝트의 고급 패턴 활용 예시"""

    def __init__(self):
        # 의존성 주입을 통한 서비스 구성
        self.node_manager = NodeManager()
        self.task_scheduler = TaskScheduler()
        self.metrics_collector = MetricsCollector()

        # 스레드 안전성을 위한 락
        self._workflow_lock = asyncio.Lock()
        self._node_locks = defaultdict(asyncio.Lock)

    @async_timeit
    @retry(max_attempts=3)
    async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """워크플로우 실행 - 데코레이터 패턴 활용"""

        async with self._workflow_lock:  # 동시성 제어
            try:
                # 1. 워크플로우 검증
                await self._validate_workflow(workflow)

                # 2. 태스크 의존성 분석 및 병렬 실행 계획
                execution_plan = await self._create_execution_plan(workflow)

                # 3. 병렬 실행
                results = await self._execute_parallel_tasks(execution_plan)

                # 4. 결과 집계
                return await self._aggregate_results(results)

            except Exception as e:
                # 예외 발생 시 정리 작업
                await self._cleanup_failed_workflow(workflow.id)
                raise WorkflowExecutionError(workflow.id, str(e))

    async def _execute_parallel_tasks(self, execution_plan: ExecutionPlan) -> List[TaskResult]:
        """태스크 병렬 실행"""
        all_results = []

        # 의존성 레벨별로 순차 실행
        for level in execution_plan.levels:
            # 같은 레벨의 태스크들은 병렬 실행
            level_tasks = [
                self._execute_task_on_node(task)
                for task in level.tasks
            ]

            # 모든 태스크 완료 대기 (예외 포함)
            level_results = await asyncio.gather(
                *level_tasks,
                return_exceptions=True
            )

            # 결과와 예외 분리 처리
            for task, result in zip(level.tasks, level_results):
                if isinstance(result, Exception):
                    # 실패한 태스크 처리
                    await self._handle_task_failure(task, result)
                else:
                    all_results.append(result)

        return all_results
```

### 2. 인텔리전트 캐싱 시스템

```python
# src/project_maestro/core/intelligent_cache.py 분석

class IntelligentCache:
    """다중 전략과 계층을 가진 고급 캐싱 시스템"""

    def __init__(self, strategies: List[CacheStrategy]):
        # 전략 패턴: 런타임에 캐싱 전략 변경 가능
        self.strategies = {
            strategy.strategy_type: strategy
            for strategy in strategies
        }

        # 관찰자 패턴: 캐시 이벤트 모니터링
        self.observers = []

        # 스레드 안전성
        self._lock = threading.RLock()

        # 백그라운드 태스크 관리
        self._background_tasks = set()

    async def get_semantic(self, query: str, threshold: float = 0.8) -> Optional[Any]:
        """의미론적 유사도 기반 캐시 조회"""

        # 1. 쿼리 임베딩 생성
        query_embedding = await self._get_embedding(query)

        # 2. 기존 캐시 엔트리와 유사도 계산 (병렬 처리)
        similarity_tasks = [
            self._calculate_similarity(query_embedding, entry.embedding)
            for entry in self._cache.values()
            if entry.embedding is not None
        ]

        similarities = await asyncio.gather(*similarity_tasks)

        # 3. 임계값 이상의 가장 유사한 항목 반환
        best_match = None
        best_score = 0

        for entry, score in zip(self._cache.values(), similarities):
            if score > threshold and score > best_score:
                best_match = entry
                best_score = score

        if best_match:
            # 캐시 히트 이벤트 발생
            await self._notify_observers(CacheEvent.HIT, best_match.key)
            return best_match.response

        return None

    async def set_with_quality(self, key: str, value: Any, quality_score: float):
        """품질 점수와 함께 캐시 저장"""

        async with self._lock:
            # 기존 항목보다 품질이 좋은 경우만 교체
            existing = self._cache.get(key)
            if existing and existing.quality_score > quality_score:
                return False

            # 새 엔트리 생성
            entry = CacheEntry(
                key=key,
                query=key,  # 단순화
                response=value,
                quality_score=quality_score,
                timestamp=time.time()
            )

            # 임베딩 생성 (백그라운드)
            task = asyncio.create_task(self._generate_embedding(entry))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            self._cache[key] = entry

            # 관찰자에게 알림
            await self._notify_observers(CacheEvent.SET, key)

            return True
```

### 3. 모니터링 시스템

```python
# src/project_maestro/core/monitoring.py 분석

class MonitoringSystem:
    """포괄적인 모니터링 및 알림 시스템"""

    def __init__(self):
        # 컴포지트 패턴: 여러 메트릭 수집기 조합
        self.collectors = []

        # 옵저버 패턴: 메트릭 변화 감지
        self.metric_observers = defaultdict(list)

        # 스레드 풀: 메트릭 수집 병렬화
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 큐: 비동기 메트릭 처리
        self.metric_queue = asyncio.Queue(maxsize=10000)

        # 백그라운드 태스크
        self._processing_task = None
        self._collection_tasks = set()

    async def start_monitoring(self):
        """모니터링 시작"""
        # 메트릭 처리 백그라운드 태스크 시작
        self._processing_task = asyncio.create_task(
            self._process_metrics()
        )

        # 각 수집기에 대한 수집 태스크 시작
        for collector in self.collectors:
            task = asyncio.create_task(
                self._run_collector(collector)
            )
            self._collection_tasks.add(task)

    async def _process_metrics(self):
        """메트릭 큐 처리 (소비자)"""
        while True:
            try:
                # 메트릭 배치 수집 (성능 최적화)
                metrics_batch = []

                # 1초 또는 100개 메트릭까지 수집
                deadline = time.time() + 1.0
                while len(metrics_batch) < 100 and time.time() < deadline:
                    try:
                        metric = await asyncio.wait_for(
                            self.metric_queue.get(),
                            timeout=deadline - time.time()
                        )
                        metrics_batch.append(metric)
                    except asyncio.TimeoutError:
                        break

                if metrics_batch:
                    # 배치 처리 (I/O 최적화)
                    await self._process_metric_batch(metrics_batch)

            except Exception as e:
                logger.error(f"메트릭 처리 오류: {e}")
                await asyncio.sleep(1)  # 오류 시 잠시 대기

    async def _run_collector(self, collector: MetricCollector):
        """개별 수집기 실행 (생산자)"""
        while True:
            try:
                # 수집기별 간격으로 메트릭 수집
                await asyncio.sleep(collector.interval)

                # 블로킹 수집 작업을 스레드 풀에서 실행
                metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    collector.collect_metrics
                )

                # 수집된 메트릭을 큐에 추가
                for metric in metrics:
                    try:
                        self.metric_queue.put_nowait(metric)
                    except asyncio.QueueFull:
                        # 큐가 가득한 경우 오래된 메트릭 제거
                        try:
                            self.metric_queue.get_nowait()
                            self.metric_queue.put_nowait(metric)
                        except asyncio.QueueEmpty:
                            pass

            except Exception as e:
                logger.error(f"수집기 {collector.name} 오류: {e}")
                await asyncio.sleep(collector.interval)
```

---

## 요약 및 다음 단계

### 이번 단계에서 배운 고급 개념

1. **GIL과 실행 모델**: Python의 동시성 제약과 해결 방법
2. **AsyncIO**: 비동기 프로그래밍과 이벤트 루프
3. **병렬 처리**: Threading vs Multiprocessing 선택 기준
4. **예외 처리**: 견고한 오류 관리 전략
5. **의존성 주입**: 결합도를 낮추는 설계 패턴
6. **고급 OOP**: ABC, 데코레이터, 팩토리 패턴

### 실무에서의 활용 패턴

| 패턴 | 사용 상황 | 프로젝트 예시 |
|------|-----------|---------------|
| AsyncIO | I/O 집약적 작업 | HTTP 요청, 데이터베이스 쿼리 |
| Threading | I/O 대기가 있는 작업 | 파일 읽기, 네트워크 통신 |
| Multiprocessing | CPU 집약적 작업 | 데이터 분석, 이미지 처리 |
| 의존성 주입 | 테스트 가능한 설계 | 서비스 레이어 분리 |
| 추상화 | 다형성 필요 | 다중 구현체 지원 |

### 다음 단계 예고

다음 문서에서는 프로젝트 구조 분석과 백엔드 아키텍처를 다룰 예정입니다:
- 마이크로서비스 아키텍처 패턴
- API 설계와 RESTful 서비스
- 데이터베이스 설계와 ORM
- 보안과 인증/인가
- 성능 최적화와 스케일링

이제 Python의 고급 개념을 이해했으니, 실제 백엔드 시스템 설계로 넘어갈 준비가 되었습니다!