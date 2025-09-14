"""인텔리전트 캐싱 시스템"""

import asyncio
import time
import hashlib
import pickle
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod
import threading
import redis
import sqlite3

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings import OpenAIEmbeddings


class CacheStrategy(Enum):
    """캐시 전략"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    SEMANTIC = "semantic"          # 의미론적 유사도
    ADAPTIVE = "adaptive"          # 적응형
    QUALITY_BASED = "quality"      # 품질 기반


class CacheLevel(Enum):
    """캐시 레벨"""
    MEMORY = "memory"              # 메모리 캐시
    REDIS = "redis"                # Redis 캐시
    DISK = "disk"                  # 디스크 캐시
    DISTRIBUTED = "distributed"    # 분산 캐시


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    
    key: str
    query: str
    response: Any
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    access_count: int = 0
    last_access_time: float = 0.0
    creation_time: float = 0.0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class CacheConfig:
    """캐시 설정"""
    
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_memory_size: int = 1000  # 메모리 캐시 최대 엔트리
    max_disk_size: int = 10000   # 디스크 캐시 최대 엔트리
    default_ttl: float = 3600    # 기본 TTL (1시간)
    semantic_threshold: float = 0.85  # 의미론적 유사도 임계값
    quality_threshold: float = 0.7    # 품질 임계값
    embedding_cache_size: int = 5000  # 임베딩 캐시 크기
    redis_url: Optional[str] = None
    disk_cache_path: str = "./cache"
    enable_compression: bool = True
    async_writes: bool = True


class QualityEvaluator(ABC):
    """응답 품질 평가자 추상 클래스"""
    
    @abstractmethod
    async def evaluate(self, query: str, response: Any, context: Dict[str, Any] = None) -> float:
        """응답 품질 평가 (0.0 - 1.0)"""
        pass


class SimpleQualityEvaluator(QualityEvaluator):
    """간단한 품질 평가자"""
    
    def __init__(self):
        self.response_history = deque(maxlen=1000)
    
    async def evaluate(self, query: str, response: Any, context: Dict[str, Any] = None) -> float:
        """간단한 휴리스틱 기반 품질 평가"""
        
        quality_factors = []
        
        # 응답 길이 평가
        response_str = str(response)
        query_len = len(query.split())
        response_len = len(response_str.split())
        
        # 적절한 길이 비율 (1:3 ~ 1:10 정도가 적당)
        length_ratio = response_len / max(query_len, 1)
        if 3 <= length_ratio <= 15:
            length_score = 1.0
        elif 2 <= length_ratio <= 20:
            length_score = 0.8
        else:
            length_score = 0.6
        quality_factors.append(length_score * 0.2)
        
        # 구조화 정도 평가
        structure_score = 0.7  # 기본값
        if '\n' in response_str:
            structure_score += 0.1
        if any(marker in response_str.lower() for marker in ['1.', '2.', '-', '*']):
            structure_score += 0.1
        if any(keyword in response_str.lower() for keyword in ['however', 'therefore', 'because']):
            structure_score += 0.1
        quality_factors.append(min(structure_score, 1.0) * 0.3)
        
        # 쿼리 관련성 평가 (간단한 키워드 매칭)
        query_words = set(query.lower().split())
        response_words = set(response_str.lower().split())
        
        if query_words:
            relevance_score = len(query_words & response_words) / len(query_words)
            quality_factors.append(relevance_score * 0.3)
        else:
            quality_factors.append(0.5 * 0.3)
        
        # 에러 또는 예외 응답 체크
        error_indicators = ['error', 'exception', 'failed', 'unable to', 'cannot', 'sorry']
        has_error = any(indicator in response_str.lower() for indicator in error_indicators)
        error_score = 0.2 if has_error else 1.0
        quality_factors.append(error_score * 0.2)
        
        final_score = sum(quality_factors)
        
        # 히스토리 업데이트
        self.response_history.append({
            'query': query,
            'response_length': response_len,
            'quality_score': final_score,
            'timestamp': time.time()
        })
        
        return min(max(final_score, 0.0), 1.0)


class TTLManager:
    """TTL 관리자"""
    
    def __init__(self):
        self.ttl_entries: Dict[str, float] = {}  # key -> expiry_time
        self.cleanup_interval = 60  # 1분마다 정리
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
    
    async def start_cleanup(self):
        """주기적 정리 시작"""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup(self):
        """정리 중단"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    def set_ttl(self, key: str, ttl_seconds: float):
        """TTL 설정"""
        expiry_time = time.time() + ttl_seconds
        with self._lock:
            self.ttl_entries[key] = expiry_time
    
    def is_expired(self, key: str) -> bool:
        """만료 확인"""
        with self._lock:
            if key not in self.ttl_entries:
                return False
            return time.time() > self.ttl_entries[key]
    
    def remove_ttl(self, key: str):
        """TTL 제거"""
        with self._lock:
            self.ttl_entries.pop(key, None)
    
    def get_expired_keys(self) -> List[str]:
        """만료된 키들 반환"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, expiry_time in self.ttl_entries.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
        
        return expired_keys
    
    async def _cleanup_loop(self):
        """정리 루프"""
        while True:
            try:
                expired_keys = self.get_expired_keys()
                
                # TTL 엔트리에서 제거
                with self._lock:
                    for key in expired_keys:
                        self.ttl_entries.pop(key, None)
                
                # 외부에서 실제 캐시 엔트리 제거 처리
                if hasattr(self, 'cache_instance') and expired_keys:
                    await self.cache_instance._remove_expired_entries(expired_keys)
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"TTL cleanup error: {e}")
                await asyncio.sleep(self.cleanup_interval)


class IntelligentCacheSystem:
    """의미론적 유사도 기반 인텔리전트 캐싱"""
    
    def __init__(self, config: CacheConfig = None, embeddings: OpenAIEmbeddings = None):
        self.config = config or CacheConfig()
        self.embeddings = embeddings or OpenAIEmbeddings()
        
        # 멀티레벨 캐시 저장소
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}  # 쿼리 임베딩 캐시
        
        # Redis 연결 (옵션)
        self.redis_client = None
        if self.config.redis_url:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                self.redis_client.ping()
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # 디스크 캐시 (SQLite)
        self.disk_db_path = f"{self.config.disk_cache_path}/cache.db"
        self._init_disk_cache()
        
        # TTL 관리
        self.ttl_manager = TTLManager()
        self.ttl_manager.cache_instance = self
        
        # 품질 평가자
        self.quality_evaluator = SimpleQualityEvaluator()
        
        # 통계 및 성능 추적
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_hits': 0,
            'exact_hits': 0,
            'avg_quality_score': 0.0,
            'storage_distribution': defaultdict(int)
        }
        
        # 접근 패턴 학습
        self.access_patterns = defaultdict(list)
        self.popularity_scores = defaultdict(float)
        
        # 스레드 안전성
        self._lock = threading.RLock()
    
    def _init_disk_cache(self):
        """디스크 캐시 초기화"""
        import os
        os.makedirs(self.config.disk_cache_path, exist_ok=True)
        
        conn = sqlite3.connect(self.disk_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response BLOB NOT NULL,
                embedding BLOB,
                quality_score REAL,
                access_count INTEGER,
                last_access_time REAL,
                creation_time REAL,
                metadata TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_query ON cache_entries(query)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_access_time ON cache_entries(last_access_time)')
        conn.commit()
        conn.close()
    
    async def start(self):
        """캐시 시스템 시작"""
        await self.ttl_manager.start_cleanup()
    
    async def stop(self):
        """캐시 시스템 중단"""
        await self.ttl_manager.stop_cleanup()
    
    async def semantic_lookup(
        self, 
        query: str, 
        threshold: float = None
    ) -> Optional[Tuple[CacheEntry, float]]:
        """의미론적 캐시 검색"""
        
        threshold = threshold or self.config.semantic_threshold
        
        try:
            # 쿼리 임베딩 생성 또는 조회
            query_embedding = await self._get_embedding(query)
            
            best_match = None
            best_score = 0.0
            
            # 메모리 캐시에서 검색
            with self._lock:
                for cache_key, entry in self.memory_cache.items():
                    if entry.embedding is not None:
                        similarity = cosine_similarity(
                            [query_embedding], 
                            [entry.embedding]
                        )[0][0]
                        
                        if similarity > threshold and similarity > best_score:
                            best_match = entry
                            best_score = similarity
            
            # Redis 캐시에서 검색 (구현 생략 - 임베딩 저장/조회 복잡)
            
            # 디스크 캐시에서 검색 (상위 후보들만)
            if not best_match or best_score < 0.95:  # 완전 일치가 아닌 경우
                disk_candidates = await self._get_disk_cache_candidates(query)
                
                for entry in disk_candidates:
                    if entry.embedding is not None:
                        similarity = cosine_similarity(
                            [query_embedding], 
                            [entry.embedding]
                        )[0][0]
                        
                        if similarity > threshold and similarity > best_score:
                            best_match = entry
                            best_score = similarity
            
            if best_match:
                # 접근 정보 업데이트
                await self._update_access_info(best_match.key)
                
                # 통계 업데이트
                self.stats['cache_hits'] += 1
                if best_score > 0.95:
                    self.stats['exact_hits'] += 1
                else:
                    self.stats['semantic_hits'] += 1
                
                return best_match, best_score
            
            return None
            
        except Exception as e:
            print(f"Semantic lookup error: {e}")
            return None
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """임베딩 생성 또는 캐시에서 조회"""
        
        # 임베딩 캐시 확인
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # 새 임베딩 생성
        embedding = await self.embeddings.aembed_query(text)
        embedding_array = np.array(embedding)
        
        # 캐시 크기 관리
        if len(self.embedding_cache) >= self.config.embedding_cache_size:
            # LRU 방식으로 가장 오래된 것 제거
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[text_hash] = embedding_array
        return embedding_array
    
    async def _get_disk_cache_candidates(self, query: str, limit: int = 50) -> List[CacheEntry]:
        """디스크 캐시에서 후보 엔트리들 조회"""
        
        candidates = []
        
        try:
            conn = sqlite3.connect(self.disk_db_path)
            
            # 최근 접근한 엔트리들 중에서 검색
            cursor = conn.execute('''
                SELECT key, query, response, embedding, quality_score,
                       access_count, last_access_time, creation_time, metadata
                FROM cache_entries
                WHERE last_access_time > ?
                ORDER BY access_count DESC, last_access_time DESC
                LIMIT ?
            ''', (time.time() - 86400, limit))  # 24시간 이내
            
            for row in cursor.fetchall():
                embedding_blob = row[3]
                embedding = None
                if embedding_blob:
                    embedding = pickle.loads(embedding_blob)
                
                metadata = json.loads(row[8]) if row[8] else {}
                
                entry = CacheEntry(
                    key=row[0],
                    query=row[1],
                    response=pickle.loads(row[2]),
                    embedding=embedding,
                    quality_score=row[4],
                    access_count=row[5],
                    last_access_time=row[6],
                    creation_time=row[7],
                    metadata=metadata
                )
                candidates.append(entry)
            
            conn.close()
            
        except Exception as e:
            print(f"Disk cache lookup error: {e}")
        
        return candidates
    
    async def intelligent_store(
        self, 
        query: str, 
        response: Any,
        metadata: Dict[str, Any] = None,
        force_cache: bool = False
    ) -> bool:
        """지능적 캐시 저장"""
        
        # 응답 품질 평가
        quality_score = await self.quality_evaluator.evaluate(query, response, metadata)
        
        # 품질이 낮으면 캐싱하지 않음 (강제가 아닌 경우)
        if not force_cache and quality_score < self.config.quality_threshold:
            return False
        
        # 쿼리 임베딩 생성
        embedding = await self._get_embedding(query)
        
        # 캐시 키 생성
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        
        # TTL 계산
        ttl = self._calculate_adaptive_ttl(quality_score, metadata or {})
        
        # 캐시 엔트리 생성
        entry = CacheEntry(
            key=cache_key,
            query=query,
            response=response,
            embedding=embedding,
            quality_score=quality_score,
            creation_time=time.time(),
            last_access_time=time.time(),
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # 저장 레벨 결정
        storage_level = self._determine_storage_level(entry)
        
        success = False
        
        # 메모리 캐시에 저장
        if storage_level in [CacheLevel.MEMORY, CacheLevel.DISTRIBUTED]:
            success = await self._store_in_memory(entry)
            self.stats['storage_distribution']['memory'] += 1
        
        # Redis 캐시에 저장
        if self.redis_client and storage_level in [CacheLevel.REDIS, CacheLevel.DISTRIBUTED]:
            success = await self._store_in_redis(entry) or success
            self.stats['storage_distribution']['redis'] += 1
        
        # 디스크 캐시에 저장 (비동기)
        if storage_level in [CacheLevel.DISK, CacheLevel.DISTRIBUTED]:
            if self.config.async_writes:
                asyncio.create_task(self._store_in_disk(entry))
            else:
                success = await self._store_in_disk(entry) or success
            self.stats['storage_distribution']['disk'] += 1
        
        # TTL 설정
        if ttl:
            self.ttl_manager.set_ttl(cache_key, ttl)
        
        # 통계 업데이트
        self.stats['avg_quality_score'] = (
            (self.stats['avg_quality_score'] * (self.stats['total_requests'] - 1) + quality_score)
            / self.stats['total_requests']
        )
        
        return success
    
    def _determine_storage_level(self, entry: CacheEntry) -> CacheLevel:
        """저장 레벨 결정"""
        
        # 품질과 접근 빈도 기반
        if entry.quality_score > 0.9:
            return CacheLevel.DISTRIBUTED  # 모든 레벨에 저장
        elif entry.quality_score > 0.8:
            return CacheLevel.REDIS if self.redis_client else CacheLevel.MEMORY
        elif entry.quality_score > 0.6:
            return CacheLevel.MEMORY
        else:
            return CacheLevel.DISK
    
    async def _store_in_memory(self, entry: CacheEntry) -> bool:
        """메모리 캐시 저장"""
        
        try:
            with self._lock:
                # 용량 초과시 LRU 제거
                if len(self.memory_cache) >= self.config.max_memory_size:
                    oldest_key = min(
                        self.memory_cache.keys(),
                        key=lambda k: self.memory_cache[k].last_access_time
                    )
                    del self.memory_cache[oldest_key]
                
                self.memory_cache[entry.key] = entry
            
            return True
            
        except Exception as e:
            print(f"Memory cache store error: {e}")
            return False
    
    async def _store_in_redis(self, entry: CacheEntry) -> bool:
        """Redis 캐시 저장"""
        
        if not self.redis_client:
            return False
        
        try:
            # 데이터 직렬화
            serialized_data = {
                'query': entry.query,
                'response': pickle.dumps(entry.response),
                'quality_score': entry.quality_score,
                'creation_time': entry.creation_time,
                'metadata': json.dumps(entry.metadata)
            }
            
            # Redis에 저장 (TTL 포함)
            pipe = self.redis_client.pipeline()
            pipe.hset(f"cache:{entry.key}", mapping=serialized_data)
            
            if entry.ttl:
                pipe.expire(f"cache:{entry.key}", int(entry.ttl))
            
            pipe.execute()
            
            return True
            
        except Exception as e:
            print(f"Redis cache store error: {e}")
            return False
    
    async def _store_in_disk(self, entry: CacheEntry) -> bool:
        """디스크 캐시 저장"""
        
        try:
            conn = sqlite3.connect(self.disk_db_path)
            
            # 용량 관리
            conn.execute('DELETE FROM cache_entries WHERE key IN (SELECT key FROM cache_entries ORDER BY last_access_time ASC LIMIT (SELECT MAX(0, COUNT(*) - ?)))', (self.config.max_disk_size,))
            
            # 엔트리 저장
            conn.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, query, response, embedding, quality_score, access_count, 
                 last_access_time, creation_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.key,
                entry.query,
                pickle.dumps(entry.response),
                pickle.dumps(entry.embedding) if entry.embedding is not None else None,
                entry.quality_score,
                entry.access_count,
                entry.last_access_time,
                entry.creation_time,
                json.dumps(entry.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Disk cache store error: {e}")
            return False
    
    def _calculate_adaptive_ttl(
        self, 
        quality_score: float,
        metadata: Dict[str, Any]
    ) -> float:
        """품질과 메타데이터 기반 적응형 TTL"""
        
        base_ttl = self.config.default_ttl
        
        # 품질 기반 조정 (높은 품질 = 더 긴 TTL)
        quality_multiplier = 0.5 + (quality_score * 1.5)
        
        # 컨텐츠 타입 기반 조정
        content_type = metadata.get('content_type', 'dynamic')
        if content_type == 'static':
            type_multiplier = 3.0
        elif content_type == 'semi_static':
            type_multiplier = 2.0
        elif content_type == 'dynamic':
            type_multiplier = 1.0
        else:
            type_multiplier = 0.5  # 휘발성 데이터
        
        # 사용 빈도 기반 조정
        usage_frequency = metadata.get('usage_frequency', 1.0)
        frequency_multiplier = min(usage_frequency, 3.0)
        
        # 응답 크기 기반 조정 (큰 응답 = 더 긴 TTL)
        response_size = metadata.get('response_size', 1000)
        size_multiplier = min(1.0 + (response_size / 10000), 2.0)
        
        # 최종 TTL 계산
        final_ttl = base_ttl * quality_multiplier * type_multiplier * frequency_multiplier * size_multiplier
        
        # 범위 제한
        return max(300, min(final_ttl, 86400 * 7))  # 5분 ~ 7일
    
    async def _update_access_info(self, cache_key: str):
        """접근 정보 업데이트"""
        
        current_time = time.time()
        
        # 메모리 캐시 업데이트
        with self._lock:
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                entry.access_count += 1
                entry.last_access_time = current_time
        
        # 디스크 캐시 업데이트 (비동기)
        async def update_disk():
            try:
                conn = sqlite3.connect(self.disk_db_path)
                conn.execute('''
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_access_time = ?
                    WHERE key = ?
                ''', (current_time, cache_key))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Disk access update error: {e}")
        
        asyncio.create_task(update_disk())
        
        # 접근 패턴 학습
        self.access_patterns[cache_key].append(current_time)
        if len(self.access_patterns[cache_key]) > 100:
            self.access_patterns[cache_key] = self.access_patterns[cache_key][-50:]
        
        # 인기도 점수 업데이트
        self.popularity_scores[cache_key] = self._calculate_popularity(cache_key)
    
    def _calculate_popularity(self, cache_key: str) -> float:
        """인기도 점수 계산"""
        
        if cache_key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[cache_key]
        if not accesses:
            return 0.0
        
        current_time = time.time()
        
        # 최근성과 빈도 조합
        recent_accesses = [t for t in accesses if current_time - t < 3600]  # 1시간 이내
        frequency_score = len(accesses) / 100.0  # 정규화
        recency_score = len(recent_accesses) / max(len(accesses), 1)
        
        return min(frequency_score * 0.6 + recency_score * 0.4, 1.0)
    
    async def _remove_expired_entries(self, expired_keys: List[str]):
        """만료된 엔트리들 제거"""
        
        # 메모리 캐시에서 제거
        with self._lock:
            for key in expired_keys:
                self.memory_cache.pop(key, None)
        
        # Redis 캐시는 자동 만료
        
        # 디스크 캐시에서 제거 (백그라운드)
        async def remove_from_disk():
            try:
                conn = sqlite3.connect(self.disk_db_path)
                placeholders = ','.join(['?'] * len(expired_keys))
                conn.execute(f'DELETE FROM cache_entries WHERE key IN ({placeholders})', expired_keys)
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Disk cache cleanup error: {e}")
        
        asyncio.create_task(remove_from_disk())
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        
        hit_rate = self.stats['cache_hits'] / max(self.stats['total_requests'], 1)
        
        # 메모리 사용량
        memory_usage = len(self.memory_cache)
        
        # 디스크 사용량
        disk_usage = 0
        try:
            conn = sqlite3.connect(self.disk_db_path)
            cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
            disk_usage = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
        
        return {
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'semantic_hits': self.stats['semantic_hits'],
            'exact_hits': self.stats['exact_hits'],
            'avg_quality_score': self.stats['avg_quality_score'],
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'redis_available': self.redis_client is not None,
            'storage_distribution': dict(self.stats['storage_distribution']),
            'embedding_cache_size': len(self.embedding_cache),
            'top_popular_queries': await self._get_top_popular_queries()
        }
    
    async def _get_top_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """인기 쿼리 상위 목록"""
        
        popular_items = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        result = []
        for cache_key, score in popular_items:
            # 메모리에서 쿼리 찾기
            query = "Unknown"
            if cache_key in self.memory_cache:
                query = self.memory_cache[cache_key].query
            
            result.append({
                'cache_key': cache_key,
                'query_preview': query[:50] + "..." if len(query) > 50 else query,
                'popularity_score': score,
                'access_count': len(self.access_patterns.get(cache_key, []))
            })
        
        return result
    
    async def clear_cache(self, level: Optional[CacheLevel] = None):
        """캐시 정리"""
        
        if level is None or level == CacheLevel.MEMORY:
            with self._lock:
                self.memory_cache.clear()
                self.embedding_cache.clear()
        
        if level is None or level == CacheLevel.REDIS:
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    print(f"Redis clear error: {e}")
        
        if level is None or level == CacheLevel.DISK:
            try:
                conn = sqlite3.connect(self.disk_db_path)
                conn.execute('DELETE FROM cache_entries')
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Disk cache clear error: {e}")
        
        # 통계 초기화
        if level is None:
            self.stats = {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'semantic_hits': 0,
                'exact_hits': 0,
                'avg_quality_score': 0.0,
                'storage_distribution': defaultdict(int)
            }
            self.access_patterns.clear()
            self.popularity_scores.clear()


class CacheMiddleware:
    """캐시 미들웨어"""
    
    def __init__(self, cache_system: IntelligentCacheSystem):
        self.cache = cache_system
    
    async def __call__(
        self, 
        query: str, 
        handler_func,
        cache_metadata: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> Any:
        """캐시 미들웨어 실행"""
        
        self.cache.stats['total_requests'] += 1
        
        if not use_cache:
            return await handler_func(query)
        
        # 캐시 조회
        cache_result = await self.cache.semantic_lookup(query)
        
        if cache_result:
            entry, similarity = cache_result
            
            # TTL 확인
            if entry.ttl and self.cache.ttl_manager.is_expired(entry.key):
                # 만료된 캐시 제거
                await self.cache._remove_expired_entries([entry.key])
            else:
                return entry.response
        
        # 캐시 미스 - 실제 처리 실행
        self.cache.stats['cache_misses'] += 1
        response = await handler_func(query)
        
        # 응답 캐싱
        await self.cache.intelligent_store(
            query, 
            response, 
            metadata=cache_metadata
        )
        
        return response