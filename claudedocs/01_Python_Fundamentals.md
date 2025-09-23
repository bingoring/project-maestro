# Python 기초부터 마스터까지 - 1단계: Python 기본 개념

## 목차
1. [Python 언어 기본 구조](#python-언어-기본-구조)
2. [변수와 데이터 타입](#변수와-데이터-타입)
3. [함수 선언과 활용](#함수-선언과-활용)
4. [클래스와 __init__ 메서드](#클래스와-__init__-메서드)
5. [모듈과 패키지 시스템](#모듈과-패키지-시스템)
6. [프로젝트 예시로 보는 실제 활용](#프로젝트-예시로-보는-실제-활용)

---

## Python 언어 기본 구조

### Python의 철학과 특징
Python은 "읽기 쉽고 간결한" 코드를 추구하는 언어입니다. 이는 프로젝트에서도 잘 드러납니다:

```python
# src/project_maestro/__init__.py에서 발췌
"""
Project Maestro: AI Agent-based Game Prototyping Automation System

A sophisticated multi-agent orchestration system that converts game design
documents into playable game prototypes using LangChain and specialized AI agents.
"""
```

**주요 특징:**
- **인터프리터 언어**: 컴파일 없이 바로 실행
- **들여쓰기로 블록 구분**: 중괄호 {} 대신 들여쓰기 사용
- **동적 타이핑**: 런타임에 변수 타입이 결정됨
- **객체지향**: 모든 것이 객체

### 문법 기본 구조

```python
# 1. 주석 - 한 줄 주석
"""
여러 줄 주석
docstring이라고 부르며, 함수나 클래스의 설명에 사용
"""

# 2. 들여쓰기 (매우 중요!)
if True:
    print("이 코드는 실행됩니다")  # 4칸 들여쓰기
    if True:
        print("중첩된 블록")       # 8칸 들여쓰기

# 3. import 문 - 모듈 가져오기
import asyncio
from pathlib import Path
from typing import Optional, List
```

---

## 변수와 데이터 타입

### 기본 데이터 타입

```python
# 숫자 타입
port = 8000                    # int (정수)
version = 0.1                  # float (실수)

# 문자열 타입
environment = "development"    # str
api_host = "0.0.0.0"

# 불린 타입
debug = False                  # bool
langchain_tracing = True

# None 타입 (다른 언어의 null과 같음)
api_key = None
```

### 컬렉션 타입

```python
# 리스트 (배열과 같음)
workers = [1, 2, 3, 4]
endpoints = ["health", "status", "metrics"]

# 딕셔너리 (객체/맵과 같음)
config = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}

# 튜플 (불변 리스트)
coordinates = (100, 200)

# 세트 (중복 없는 집합)
allowed_methods = {"GET", "POST", "PUT", "DELETE"}
```

### 프로젝트에서의 실제 사용 예시

```python
# src/project_maestro/core/config.py에서 발췌
class Settings(BaseSettings):
    # 기본값이 있는 변수 선언
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    api_port: int = Field(default=8000)

    # Optional 타입 - None이 될 수 있음을 명시
    openai_api_key: Optional[str] = Field(default=None)
```

**중요한 차이점:**
- JavaScript와 달리 `var`, `let`, `const` 없이 바로 변수명 사용
- 타입 힌트(`:`로 표시)는 선택사항이지만 권장됨
- `Field(default=...)`는 Pydantic 라이브러리의 기능

---

## 함수 선언과 활용

### 기본 함수 선언

```python
# 1. 가장 간단한 함수
def greet():
    print("안녕하세요!")

# 2. 매개변수가 있는 함수
def greet_user(name):
    print(f"안녕하세요, {name}님!")

# 3. 반환값이 있는 함수
def add(a, b):
    return a + b

# 4. 기본값이 있는 매개변수
def greet_with_default(name="사용자"):
    return f"안녕하세요, {name}님!"
```

### 타입 힌트를 사용한 현대적 함수 선언

```python
# 프로젝트 스타일 - 타입을 명시
def calculate_score(base_score: int, multiplier: float = 1.0) -> float:
    """점수를 계산하는 함수

    Args:
        base_score: 기본 점수
        multiplier: 배수 (기본값: 1.0)

    Returns:
        계산된 최종 점수
    """
    return base_score * multiplier
```

### 프로젝트에서의 실제 함수 예시

```python
# src/project_maestro/cli.py에서 발췌
@app.command()
def version():
    """Show version information."""
    console.print(f"Project Maestro v0.1.0", style="bold green")
    console.print(f"Environment: {settings.environment}")
    console.print(f"Debug: {settings.debug}")
```

**핵심 개념:**
- `@app.command()`: 데코레이터 - 함수에 추가 기능을 부여
- `"""..."""`: docstring - 함수 설명 문서
- `f"문자열 {변수}"`: f-string - 문자열 포맷팅

### 비동기 함수 (Async/Await)

```python
# 비동기 함수 선언 (Node.js와 비슷)
async def fetch_data():
    """데이터를 비동기로 가져오는 함수"""
    await asyncio.sleep(1)  # 1초 대기
    return "데이터"

# 비동기 함수 호출
async def main():
    result = await fetch_data()
    print(result)

# 실행
asyncio.run(main())
```

---

## 클래스와 __init__ 메서드

### 클래스 기본 구조

```python
class GameCharacter:
    """게임 캐릭터 클래스"""

    def __init__(self, name, health=100):
        """생성자 메서드 - 객체가 만들어질 때 자동 호출"""
        self.name = name        # 인스턴스 변수
        self.health = health
        self.level = 1

    def attack(self, damage):
        """공격 메서드"""
        self.health -= damage
        print(f"{self.name}이(가) {damage} 데미지를 받았습니다!")

    def __str__(self):
        """문자열 표현 메서드"""
        return f"{self.name} (체력: {self.health}, 레벨: {self.level})"

# 객체 생성 및 사용
hero = GameCharacter("용사", 150)
print(hero)  # 용사 (체력: 150, 레벨: 1)
hero.attack(30)  # 용사이(가) 30 데미지를 받았습니다!
```

### __init__ 메서드의 역할

`__init__`은 Python에서 생성자 역할을 하는 특별한 메서드입니다:

```python
class Database:
    def __init__(self, host, port, username, password):
        """데이터베이스 연결 초기화"""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None

        # 초기화 시 자동으로 연결 시도
        self.connect()

    def connect(self):
        print(f"데이터베이스 연결 중: {self.host}:{self.port}")
        # 실제 연결 로직...

# 객체 생성 시 __init__이 자동 호출됨
db = Database("localhost", 5432, "admin", "password")
```

### 프로젝트에서의 실제 클래스 사용

```python
# src/project_maestro/core/config.py에서 발췌
class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # 클래스 변수들 (인스턴스가 만들어질 때 자동으로 설정됨)
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
```

**중요한 개념:**
- `BaseSettings`: 상속 - 다른 클래스의 기능을 물려받음
- `Field`: Pydantic의 유효성 검사 및 기본값 설정
- 환경변수 자동 읽기 기능이 내장됨

### 특별한 메서드들 (Magic Methods)

```python
class Config:
    def __init__(self, name):
        self.name = name
        self.settings = {}

    def __getitem__(self, key):
        """config['key'] 형태로 접근 가능"""
        return self.settings[key]

    def __setitem__(self, key, value):
        """config['key'] = value 형태로 설정 가능"""
        self.settings[key] = value

    def __str__(self):
        """str(config) 또는 print(config) 시 호출"""
        return f"Config({self.name}): {self.settings}"

    def __repr__(self):
        """개발자용 문자열 표현"""
        return f"Config(name='{self.name}', settings={self.settings})"

# 사용 예시
config = Config("app")
config['debug'] = True      # __setitem__ 호출
print(config['debug'])      # __getitem__ 호출
print(config)               # __str__ 호출
```

---

## 모듈과 패키지 시스템

### 모듈 기본 개념

Python에서 `.py` 파일 하나가 모듈입니다:

```python
# math_utils.py (모듈 파일)
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

PI = 3.14159
```

```python
# main.py (사용하는 파일)
import math_utils
from math_utils import add, PI

# 사용법 1
result1 = math_utils.add(5, 3)

# 사용법 2
result2 = add(5, 3)
print(PI)
```

### 패키지 구조

프로젝트의 디렉토리 구조는 패키지 시스템을 보여줍니다:

```
src/project_maestro/          # 패키지 루트
├── __init__.py              # 패키지 표시 파일
├── cli.py                   # CLI 모듈
├── core/                    # 하위 패키지
│   ├── __init__.py
│   ├── config.py           # 설정 모듈
│   ├── logging.py          # 로깅 모듈
│   └── metrics.py          # 메트릭 모듈
├── api/                     # API 패키지
│   └── main.py
└── agents/                  # 에이전트 패키지
```

### __init__.py의 역할

```python
# src/project_maestro/__init__.py
"""
Project Maestro: AI Agent-based Game Prototyping Automation System
"""

__version__ = "0.1.0"
__author__ = "Project Maestro Team"

# 패키지에서 자주 사용되는 것들을 미리 가져옴
from .core.config import settings
from .core.logging import logger

# 외부에서 사용 가능한 것들을 명시
__all__ = ["settings", "logger"]
```

**이렇게 하면:**
```python
# 다른 파일에서 간단하게 사용 가능
from project_maestro import settings, logger
```

### 상대 import와 절대 import

```python
# 절대 import (권장)
from project_maestro.core.config import settings
from project_maestro.core.logging import logger

# 상대 import (같은 패키지 내에서)
from .config import settings          # 같은 디렉토리
from ..api.main import run_server     # 상위 디렉토리의 api 패키지
from ...external import some_module   # 두 단계 위의 디렉토리
```

---

## 프로젝트 예시로 보는 실제 활용

### 1. CLI 애플리케이션 구조

```python
# src/project_maestro/cli.py 분석
import typer
from rich.console import Console

# Typer: CLI 애플리케이션 프레임워크
app = typer.Typer(
    name="maestro",
    help="Project Maestro: AI Agent-based Game Prototyping Automation System",
    rich_markup_mode="markdown"
)

# Rich: 터미널 UI 라이브러리
console = Console()

# 하위 명령어 그룹 생성
server_app = typer.Typer(name="server", help="Server management commands")
app.add_typer(server_app)  # maestro server 명령어 그룹

@app.command()  # maestro version 명령어
def version():
    console.print(f"Project Maestro v0.1.0", style="bold green")
```

### 2. 설정 관리 패턴

```python
# src/project_maestro/core/config.py 분석
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경변수와 기본값을 자동으로 관리하는 설정 클래스"""

    # 환경변수 ENVIRONMENT가 있으면 사용, 없으면 기본값
    environment: str = Field(default="development")

    # 타입 검증 자동 수행
    api_port: int = Field(default=8000)

    # 선택적 값 (None 가능)
    openai_api_key: Optional[str] = Field(default=None)

# 전역 설정 객체 생성
settings = Settings()
```

**이 패턴의 장점:**
- 환경변수 자동 읽기
- 타입 안전성
- 기본값 자동 적용
- 유효성 검사 내장

### 3. 모듈 간 의존성 관리

```python
# src/project_maestro/__init__.py
from .core.config import settings
from .core.logging import logger

# 다른 모듈에서 쉽게 사용
# from project_maestro import settings, logger
```

### 4. 타입 힌트 활용

```python
# 프로젝트 전반에서 사용되는 타입 힌트 패턴
from typing import Optional, List, Dict, Any
from pathlib import Path

def process_files(
    file_paths: List[Path],           # Path 객체의 리스트
    options: Optional[Dict[str, Any]] = None  # 선택적 옵션 딕셔너리
) -> bool:                            # 성공/실패 반환
    """파일들을 처리하는 함수"""
    if options is None:
        options = {}

    for file_path in file_paths:
        if file_path.exists():
            # 파일 처리 로직
            pass

    return True
```

---

## 요약 및 다음 단계

### 이번 단계에서 배운 핵심 개념

1. **Python 기본 문법**: 들여쓰기, 변수, 데이터 타입
2. **함수**: 선언, 매개변수, 반환값, 타입 힌트
3. **클래스**: __init__, 메서드, 상속
4. **모듈 시스템**: import, 패키지, __init__.py
5. **프로젝트 구조**: 실제 코드에서의 활용법

### Node.js와의 주요 차이점

| 개념 | Python | Node.js |
|------|---------|---------|
| 변수 선언 | `name = "값"` | `const name = "값"` |
| 함수 선언 | `def func():` | `function func() {}` |
| 비동기 | `async def` / `await` | `async function` / `await` |
| 모듈 가져오기 | `from module import func` | `const { func } = require('module')` |
| 클래스 생성자 | `__init__(self)` | `constructor()` |
| 블록 구분 | 들여쓰기 | 중괄호 `{}` |

### 다음 단계 준비

다음 문서에서는 다음 내용을 다룰 예정입니다:
- Python의 실행 모델과 GIL (Global Interpreter Lock)
- 병렬 처리 (Threading, Multiprocessing, AsyncIO)
- 예외 처리와 런타임 오류 관리
- 의존성 주입 (Dependency Injection) 패턴
- 추상화와 인터페이스 설계

이 기초를 바탕으로 더 복잡한 백엔드 개념들을 이해할 준비가 되었습니다!