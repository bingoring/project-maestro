"""
메모리 최적화 시스템

AI 에이전트들의 메모리 사용량을 지능적으로 관리하고 최적화하는 시스템입니다.
"""

import asyncio
import gc
import psutil
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import weakref
import sys
from collections import defaultdict, deque
import threading
import contextlib

from ..utils.metrics import PrometheusMetrics
from ..core.base_agent import BaseAgent


class MemoryPressure(Enum):
    """메모리 압박 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationStrategy(Enum):
    """최적화 전략"""
    CONSERVATIVE = "conservative"  # 안전한 최적화
    BALANCED = "balanced"  # 균형적 최적화
    AGGRESSIVE = "aggressive"  # 적극적 최적화


@dataclass
class MemorySnapshot:
    """메모리 스냅샷"""
    timestamp: datetime
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    gc_count: Tuple[int, int, int]
    tracemalloc_top: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    strategy: OptimizationStrategy
    memory_freed: int
    objects_collected: int
    execution_time: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class MemoryPool:
    """메모리 풀 관리"""
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self._pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._stats = {
            'created': 0,
            'reused': 0,
            'discarded': 0
        }
    
    def get_object(self, factory_func=None):
        """객체 획득"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._stats['reused'] += 1
                return obj
            elif factory_func:
                obj = factory_func()
                self._stats['created'] += 1
                return obj
            return None
    
    def return_object(self, obj):
        """객체 반납"""
        with self._lock:
            if len(self._pool) < self.max_size:
                # 객체 초기화
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
            else:
                self._stats['discarded'] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """풀 통계"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                **self._stats
            }


class SmartCache:
    """스마트 캐시 시스템"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_times = defaultdict(list)
        self._creation_times = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 획득"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # TTL 확인
            if self._is_expired(key):
                self._remove(key)
                return None
            
            # 접근 시간 기록
            now = datetime.now()
            self._access_times[key].append(now)
            
            # 최근 접근 시간만 유지 (메모리 절약)
            if len(self._access_times[key]) > 100:
                self._access_times[key] = self._access_times[key][-50:]
            
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> bool:
        """캐시에 값 저장"""
        with self._lock:
            # 크기 제한 확인
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._creation_times[key] = datetime.now()
            return True
    
    def _is_expired(self, key: str) -> bool:
        """만료 확인"""
        if key not in self._creation_times:
            return True
        
        age = datetime.now() - self._creation_times[key]
        return age.total_seconds() > self.ttl
    
    def _remove(self, key: str):
        """캐시 항목 제거"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)
    
    def _evict_lru(self):
        """LRU 기반 제거"""
        if not self._cache:
            return
        
        # 가장 오래된 접근 시간을 가진 키 찾기
        oldest_key = None
        oldest_time = datetime.now()
        
        for key in self._cache:
            if key in self._access_times and self._access_times[key]:
                last_access = self._access_times[key][-1]
            else:
                last_access = self._creation_times.get(key, datetime.min)
            
            if last_access < oldest_time:
                oldest_time = last_access
                oldest_key = key
        
        if oldest_key:
            self._remove(oldest_key)
    
    def clear_expired(self) -> int:
        """만료된 항목 정리"""
        with self._lock:
            expired_keys = [
                key for key in self._cache.keys() 
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self._lock:
            total_memory = sum(
                sys.getsizeof(v) for v in self._cache.values()
            )
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': self._calculate_hit_rate(),
                'memory_usage': total_memory,
                'expired_count': sum(1 for key in self._cache if self._is_expired(key))
            }
    
    def _calculate_hit_rate(self) -> float:
        """히트율 계산"""
        total_accesses = sum(len(times) for times in self._access_times.values())
        if total_accesses == 0:
            return 0.0
        
        # 최근 1시간 내 접근만 고려
        recent_hits = 0
        cutoff = datetime.now() - timedelta(hours=1)
        
        for times in self._access_times.values():
            recent_hits += sum(1 for t in times if t > cutoff)
        
        return recent_hits / total_accesses if total_accesses > 0 else 0.0


class MemoryOptimizer:
    """메모리 최적화 시스템"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.metrics = PrometheusMetrics()
        
        # 메모리 모니터링
        self._monitoring_active = False
        self._snapshots = deque(maxlen=1000)
        self._monitor_task = None
        
        # 최적화 컴포넌트
        self.memory_pools = {}
        self.smart_cache = SmartCache()
        self._agent_references = weakref.WeakSet()
        
        # 설정
        self.config = {
            'monitor_interval': 30,  # 초
            'pressure_thresholds': {
                MemoryPressure.LOW: 70,
                MemoryPressure.MEDIUM: 80,
                MemoryPressure.HIGH: 90,
                MemoryPressure.CRITICAL: 95
            },
            'optimization_strategies': {
                MemoryPressure.LOW: OptimizationStrategy.CONSERVATIVE,
                MemoryPressure.MEDIUM: OptimizationStrategy.BALANCED,
                MemoryPressure.HIGH: OptimizationStrategy.AGGRESSIVE,
                MemoryPressure.CRITICAL: OptimizationStrategy.AGGRESSIVE
            }
        }
        
        # tracemalloc 시작
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    async def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("메모리 모니터링 시작됨")
    
    async def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("메모리 모니터링 중지됨")
    
    async def _monitor_loop(self):
        """메모리 모니터링 루프"""
        while self._monitoring_active:
            try:
                snapshot = await self.take_snapshot()
                pressure = self.analyze_pressure(snapshot)
                
                # 메트릭 업데이트
                self.metrics.gauge(
                    'memory_usage_percent',
                    snapshot.memory_percent,
                    {'pressure': pressure.value}
                )
                
                # 압박 상황 시 최적화 수행
                if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
                    await self.optimize_memory(pressure)
                
                await asyncio.sleep(self.config['monitor_interval'])
                
            except Exception as e:
                self.logger.error("메모리 모니터링 오류", error=str(e))
                await asyncio.sleep(5)
    
    async def take_snapshot(self) -> MemorySnapshot:
        """메모리 스냅샷 생성"""
        memory = psutil.virtual_memory()
        gc_stats = gc.get_stats()
        
        # tracemalloc 정보
        top_stats = []
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            top_stats = [(stat.filename, stat.size) 
                        for stat in tracemalloc.take_snapshot().statistics('filename')[:10]]
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            gc_count=gc.get_count(),
            tracemalloc_top=top_stats
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def analyze_pressure(self, snapshot: MemorySnapshot) -> MemoryPressure:
        """메모리 압박 상황 분석"""
        percent = snapshot.memory_percent
        thresholds = self.config['pressure_thresholds']
        
        if percent >= thresholds[MemoryPressure.CRITICAL]:
            return MemoryPressure.CRITICAL
        elif percent >= thresholds[MemoryPressure.HIGH]:
            return MemoryPressure.HIGH
        elif percent >= thresholds[MemoryPressure.MEDIUM]:
            return MemoryPressure.MEDIUM
        else:
            return MemoryPressure.LOW
    
    async def optimize_memory(self, pressure: MemoryPressure) -> OptimizationResult:
        """메모리 최적화 수행"""
        start_time = datetime.now()
        strategy = self.config['optimization_strategies'][pressure]
        
        self.logger.info(
            "메모리 최적화 시작",
            pressure=pressure.value,
            strategy=strategy.value
        )
        
        memory_before = psutil.virtual_memory().used
        objects_before = len(gc.get_objects())
        
        try:
            # 전략별 최적화 수행
            if strategy == OptimizationStrategy.CONSERVATIVE:
                await self._conservative_optimization()
            elif strategy == OptimizationStrategy.BALANCED:
                await self._balanced_optimization()
            else:  # AGGRESSIVE
                await self._aggressive_optimization()
            
            # 결과 측정
            memory_after = psutil.virtual_memory().used
            objects_after = len(gc.get_objects())
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                strategy=strategy,
                memory_freed=max(0, memory_before - memory_after),
                objects_collected=max(0, objects_before - objects_after),
                execution_time=execution_time,
                success=True
            )
            
            self.logger.info(
                "메모리 최적화 완료",
                memory_freed=result.memory_freed,
                objects_collected=result.objects_collected,
                execution_time=result.execution_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error("메모리 최적화 실패", error=str(e))
            return OptimizationResult(
                strategy=strategy,
                memory_freed=0,
                objects_collected=0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                details={'error': str(e)}
            )
    
    async def _conservative_optimization(self):
        """보수적 최적화"""
        # 만료된 캐시 항목만 정리
        expired_count = self.smart_cache.clear_expired()
        
        # 약한 참조로 정리 가능한 객체들 정리
        collected = gc.collect(0)  # generation 0만
        
        self.logger.debug(
            "보수적 최적화 수행",
            expired_cache=expired_count,
            gc_collected=collected
        )
    
    async def _balanced_optimization(self):
        """균형적 최적화"""
        # 캐시 정리
        expired_count = self.smart_cache.clear_expired()
        
        # GC 수행 (generation 0, 1)
        collected = gc.collect(1)
        
        # 에이전트 메모리 정리
        agent_cleaned = await self._cleanup_agents()
        
        self.logger.debug(
            "균형적 최적화 수행",
            expired_cache=expired_count,
            gc_collected=collected,
            agents_cleaned=agent_cleaned
        )
    
    async def _aggressive_optimization(self):
        """적극적 최적화"""
        # 전체 캐시 크기 줄이기
        cache_stats = self.smart_cache.get_stats()
        if cache_stats['size'] > cache_stats['max_size'] * 0.7:
            # 캐시 크기를 70%로 줄임
            await self._reduce_cache_size(0.3)
        
        # 전체 GC 수행
        collected = gc.collect()
        
        # 에이전트 메모리 강제 정리
        agent_cleaned = await self._cleanup_agents(force=True)
        
        # 메모리 풀 크기 조정
        pool_reduced = self._reduce_memory_pools()
        
        self.logger.debug(
            "적극적 최적화 수행",
            gc_collected=collected,
            agents_cleaned=agent_cleaned,
            pools_reduced=pool_reduced
        )
    
    async def _cleanup_agents(self, force: bool = False) -> int:
        """에이전트 메모리 정리"""
        cleaned_count = 0
        
        # 약한 참조로 살아있는 에이전트들 확인
        for agent in list(self._agent_references):
            try:
                if hasattr(agent, 'cleanup_memory'):
                    await agent.cleanup_memory(force=force)
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(
                    "에이전트 메모리 정리 실패",
                    agent_id=getattr(agent, 'agent_id', 'unknown'),
                    error=str(e)
                )
        
        return cleaned_count
    
    async def _reduce_cache_size(self, reduction_ratio: float):
        """캐시 크기 축소"""
        # 스마트 캐시에서 오래된 항목들 제거
        # 현재 구현에서는 LRU 제거를 통해 처리
        current_size = self.smart_cache.get_stats()['size']
        target_removals = int(current_size * reduction_ratio)
        
        for _ in range(target_removals):
            self.smart_cache._evict_lru()
    
    def _reduce_memory_pools(self) -> int:
        """메모리 풀 크기 축소"""
        reduced_count = 0
        
        for pool_name, pool in self.memory_pools.items():
            original_size = len(pool._pool)
            # 풀 크기를 절반으로 축소
            target_size = max(1, original_size // 2)
            
            with pool._lock:
                while len(pool._pool) > target_size:
                    pool._pool.pop()
                    reduced_count += 1
        
        return reduced_count
    
    def register_agent(self, agent: BaseAgent):
        """에이전트 등록"""
        self._agent_references.add(agent)
    
    def get_memory_pool(self, name: str, max_size: int = 1000) -> MemoryPool:
        """메모리 풀 획득"""
        if name not in self.memory_pools:
            self.memory_pools[name] = MemoryPool(name, max_size)
        return self.memory_pools[name]
    
    def get_cache(self) -> SmartCache:
        """스마트 캐시 획득"""
        return self.smart_cache
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        memory = psutil.virtual_memory()
        
        # 메모리 풀 통계
        pool_stats = {
            name: pool.get_stats() 
            for name, pool in self.memory_pools.items()
        }
        
        # 캐시 통계
        cache_stats = self.smart_cache.get_stats()
        
        # 최근 스냅샷
        recent_snapshots = list(self._snapshots)[-10:]
        
        return {
            'system': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'pools': pool_stats,
            'cache': cache_stats,
            'agents': len(self._agent_references),
            'recent_snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'percent': s.memory_percent,
                    'pressure': self.analyze_pressure(s).value
                }
                for s in recent_snapshots
            ]
        }
    
    @contextlib.asynccontextmanager
    async def memory_context(self, name: str):
        """메모리 컨텍스트 관리자"""
        snapshot_before = await self.take_snapshot()
        
        try:
            yield
        finally:
            snapshot_after = await self.take_snapshot()
            memory_diff = snapshot_after.used_memory - snapshot_before.used_memory
            
            self.logger.info(
                "메모리 컨텍스트 완료",
                context=name,
                memory_change=memory_diff,
                duration=(snapshot_after.timestamp - snapshot_before.timestamp).total_seconds()
            )
            
            # 메모리 증가가 클 경우 정리 수행
            if memory_diff > 100 * 1024 * 1024:  # 100MB
                pressure = self.analyze_pressure(snapshot_after)
                if pressure != MemoryPressure.LOW:
                    await self.optimize_memory(pressure)


# 전역 메모리 옵티마이저 인스턴스
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """글로벌 메모리 옵티마이저 획득"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer