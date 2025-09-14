"""캐시 관리 API 엔드포인트"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from project_maestro.core.intelligent_cache import (
    IntelligentCacheSystem,
    CacheConfig,
    CacheLevel,
    CacheStrategy,
    CacheMiddleware
)
from langchain.embeddings import OpenAIEmbeddings

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])

# 글로벌 캐시 시스템 인스턴스
cache_system = None
cache_middleware = None

async def get_cache_system():
    """캐시 시스템 인스턴스 생성/반환"""
    global cache_system, cache_middleware
    
    if cache_system is None:
        config = CacheConfig(
            strategy=CacheStrategy.ADAPTIVE,
            max_memory_size=1000,
            max_disk_size=10000,
            semantic_threshold=0.85,
            quality_threshold=0.7
        )
        
        embeddings = OpenAIEmbeddings()
        cache_system = IntelligentCacheSystem(config, embeddings)
        cache_middleware = CacheMiddleware(cache_system)
        
        # 캐시 시스템 시작
        await cache_system.start()
    
    return cache_system


@router.get("/stats")
async def get_cache_statistics(
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 통계 조회"""
    
    try:
        stats = await cache_system.get_cache_stats()
        return JSONResponse(content={
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_cache_config(
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """현재 캐시 설정 조회"""
    
    return JSONResponse(content={
        "status": "success", 
        "data": {
            "strategy": cache_system.config.strategy.value,
            "max_memory_size": cache_system.config.max_memory_size,
            "max_disk_size": cache_system.config.max_disk_size,
            "default_ttl": cache_system.config.default_ttl,
            "semantic_threshold": cache_system.config.semantic_threshold,
            "quality_threshold": cache_system.config.quality_threshold,
            "embedding_cache_size": cache_system.config.embedding_cache_size,
            "redis_available": cache_system.redis_client is not None,
            "enable_compression": cache_system.config.enable_compression,
            "async_writes": cache_system.config.async_writes
        },
        "timestamp": datetime.now().isoformat()
    })


@router.post("/config")
async def update_cache_config(
    config_update: Dict[str, Any],
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 설정 업데이트"""
    
    try:
        updated_fields = []
        
        if "max_memory_size" in config_update:
            cache_system.config.max_memory_size = config_update["max_memory_size"]
            updated_fields.append("max_memory_size")
        
        if "max_disk_size" in config_update:
            cache_system.config.max_disk_size = config_update["max_disk_size"]
            updated_fields.append("max_disk_size")
        
        if "default_ttl" in config_update:
            cache_system.config.default_ttl = config_update["default_ttl"]
            updated_fields.append("default_ttl")
        
        if "semantic_threshold" in config_update:
            threshold = config_update["semantic_threshold"]
            if 0.0 <= threshold <= 1.0:
                cache_system.config.semantic_threshold = threshold
                updated_fields.append("semantic_threshold")
            else:
                raise ValueError("semantic_threshold must be between 0.0 and 1.0")
        
        if "quality_threshold" in config_update:
            threshold = config_update["quality_threshold"]
            if 0.0 <= threshold <= 1.0:
                cache_system.config.quality_threshold = threshold
                updated_fields.append("quality_threshold")
            else:
                raise ValueError("quality_threshold must be between 0.0 and 1.0")
        
        if "enable_compression" in config_update:
            cache_system.config.enable_compression = config_update["enable_compression"]
            updated_fields.append("enable_compression")
        
        if "async_writes" in config_update:
            cache_system.config.async_writes = config_update["async_writes"]
            updated_fields.append("async_writes")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Updated configuration fields: {', '.join(updated_fields)}",
            "updated_fields": updated_fields,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def semantic_cache_search(
    query_data: Dict[str, Any],
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """의미론적 캐시 검색"""
    
    try:
        query = query_data.get("query", "")
        threshold = query_data.get("threshold", cache_system.config.semantic_threshold)
        
        if not query:
            raise ValueError("Query is required")
        
        result = await cache_system.semantic_lookup(query, threshold)
        
        if result:
            entry, similarity = result
            return JSONResponse(content={
                "status": "hit",
                "data": {
                    "cache_key": entry.key,
                    "query": entry.query,
                    "response": entry.response,
                    "similarity_score": float(similarity),
                    "quality_score": entry.quality_score,
                    "access_count": entry.access_count,
                    "last_access_time": entry.last_access_time,
                    "creation_time": entry.creation_time,
                    "ttl_remaining": None,  # TODO: TTL 남은 시간 계산
                    "metadata": entry.metadata
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "status": "miss",
                "message": "No matching cache entry found",
                "timestamp": datetime.now().isoformat()
            })
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store")
async def store_in_cache(
    cache_data: Dict[str, Any],
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시에 데이터 저장"""
    
    try:
        query = cache_data.get("query", "")
        response = cache_data.get("response")
        metadata = cache_data.get("metadata", {})
        force_cache = cache_data.get("force_cache", False)
        
        if not query or response is None:
            raise ValueError("Query and response are required")
        
        success = await cache_system.intelligent_store(
            query=query,
            response=response,
            metadata=metadata,
            force_cache=force_cache
        )
        
        return JSONResponse(content={
            "status": "success" if success else "failed",
            "message": "Data stored successfully" if success else "Data not cached due to quality threshold",
            "cached": success,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_cache(
    level: Optional[str] = None,
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 정리"""
    
    try:
        cache_level = None
        if level:
            try:
                cache_level = CacheLevel(level)
            except ValueError:
                raise ValueError(f"Invalid cache level: {level}. Valid options: memory, redis, disk")
        
        await cache_system.clear_cache(cache_level)
        
        level_desc = level or "all levels"
        return JSONResponse(content={
            "status": "success",
            "message": f"Cache cleared for {level_desc}",
            "cleared_level": level_desc,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular")
async def get_popular_queries(
    limit: int = 10,
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """인기 쿼리 조회"""
    
    try:
        popular_queries = await cache_system._get_top_popular_queries(limit)
        
        return JSONResponse(content={
            "status": "success",
            "data": {
                "popular_queries": popular_queries,
                "total_queries": len(cache_system.popularity_scores),
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup")
async def warmup_cache(
    warmup_data: Dict[str, Any],
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 예열"""
    
    try:
        query_response_pairs = warmup_data.get("data", [])
        
        if not isinstance(query_response_pairs, list):
            raise ValueError("Data must be a list of query-response pairs")
        
        results = []
        successful = 0
        failed = 0
        
        for pair in query_response_pairs:
            if not isinstance(pair, dict) or "query" not in pair or "response" not in pair:
                failed += 1
                results.append({
                    "status": "failed",
                    "error": "Invalid format - requires query and response fields"
                })
                continue
            
            try:
                success = await cache_system.intelligent_store(
                    query=pair["query"],
                    response=pair["response"],
                    metadata=pair.get("metadata", {"source": "warmup"}),
                    force_cache=warmup_data.get("force_cache", True)
                )
                
                if success:
                    successful += 1
                    results.append({"status": "success", "query": pair["query"][:50] + "..."})
                else:
                    failed += 1
                    results.append({"status": "failed", "error": "Quality threshold not met"})
                    
            except Exception as e:
                failed += 1
                results.append({"status": "failed", "error": str(e)})
        
        return JSONResponse(content={
            "status": "completed",
            "summary": {
                "total": len(query_response_pairs),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / max(len(query_response_pairs), 1)
            },
            "details": results if warmup_data.get("include_details", False) else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def cache_health_check(
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 시스템 헬스 체크"""
    
    try:
        stats = await cache_system.get_cache_stats()
        
        # 헬스 점수 계산
        health_score = 1.0
        issues = []
        
        # 히트율 체크
        if stats['hit_rate'] < 0.3:
            health_score -= 0.2
            issues.append("Low cache hit rate")
        
        # 메모리 사용량 체크
        memory_usage_ratio = stats['memory_usage'] / cache_system.config.max_memory_size
        if memory_usage_ratio > 0.9:
            health_score -= 0.3
            issues.append("High memory usage")
        
        # 평균 품질 점수 체크
        if stats['avg_quality_score'] < 0.6:
            health_score -= 0.2
            issues.append("Low average quality score")
        
        # Redis 연결 체크
        redis_healthy = True
        if cache_system.redis_client:
            try:
                cache_system.redis_client.ping()
            except Exception:
                redis_healthy = False
                health_score -= 0.3
                issues.append("Redis connection failed")
        
        health_status = "healthy"
        if health_score < 0.5:
            health_status = "unhealthy"
        elif health_score < 0.8:
            health_status = "degraded"
        
        return JSONResponse(content={
            "status": health_status,
            "health_score": round(health_score, 2),
            "components": {
                "memory_cache": "healthy",
                "disk_cache": "healthy",
                "redis_cache": "healthy" if redis_healthy else "unhealthy",
                "ttl_manager": "healthy",
                "embeddings": "healthy"
            },
            "metrics": {
                "hit_rate": stats['hit_rate'],
                "memory_usage_ratio": memory_usage_ratio,
                "avg_quality_score": stats['avg_quality_score'],
                "total_requests": stats['total_requests']
            },
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/benchmark")
async def run_cache_benchmark(
    benchmark_config: Dict[str, Any],
    cache_system: IntelligentCacheSystem = Depends(get_cache_system)
) -> Dict[str, Any]:
    """캐시 성능 벤치마크"""
    
    try:
        num_queries = benchmark_config.get("num_queries", 100)
        query_pattern = benchmark_config.get("query_pattern", "test query {}")
        include_semantic_test = benchmark_config.get("include_semantic_test", True)
        
        # 벤치마크 데이터 생성
        test_queries = []
        test_responses = []
        
        for i in range(num_queries):
            query = query_pattern.format(i)
            response = f"This is test response number {i} for benchmarking purposes. " * 5
            test_queries.append(query)
            test_responses.append(response)
        
        # 저장 성능 테스트
        store_start = time.perf_counter()
        for i in range(num_queries):
            await cache_system.intelligent_store(
                test_queries[i], 
                test_responses[i],
                metadata={"source": "benchmark", "index": i},
                force_cache=True
            )
        store_time = time.perf_counter() - store_start
        
        # 조회 성능 테스트 (정확한 매치)
        lookup_start = time.perf_counter()
        exact_hits = 0
        for query in test_queries:
            result = await cache_system.semantic_lookup(query, threshold=0.95)
            if result:
                exact_hits += 1
        exact_lookup_time = time.perf_counter() - lookup_start
        
        results = {
            "benchmark_config": {
                "num_queries": num_queries,
                "query_pattern": query_pattern
            },
            "performance_metrics": {
                "store_operations": {
                    "total_time": store_time,
                    "avg_time_per_operation": store_time / num_queries,
                    "operations_per_second": num_queries / store_time
                },
                "lookup_operations": {
                    "total_time": exact_lookup_time,
                    "avg_time_per_operation": exact_lookup_time / num_queries,
                    "operations_per_second": num_queries / exact_lookup_time,
                    "exact_hit_rate": exact_hits / num_queries
                }
            }
        }
        
        # 의미론적 유사도 테스트
        if include_semantic_test:
            semantic_queries = [
                f"different phrasing for test query {i}" 
                for i in range(min(10, num_queries))
            ]
            
            semantic_start = time.perf_counter()
            semantic_hits = 0
            for query in semantic_queries:
                result = await cache_system.semantic_lookup(query, threshold=0.7)
                if result:
                    semantic_hits += 1
            semantic_time = time.perf_counter() - semantic_start
            
            results["performance_metrics"]["semantic_lookup"] = {
                "total_time": semantic_time,
                "avg_time_per_operation": semantic_time / len(semantic_queries),
                "semantic_hit_rate": semantic_hits / len(semantic_queries),
                "tested_queries": len(semantic_queries)
            }
        
        return JSONResponse(content={
            "status": "success",
            "benchmark_results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 시스템 종료 시 정리
@router.on_event("shutdown")
async def shutdown_cache_system():
    """캐시 시스템 종료"""
    global cache_system
    
    if cache_system:
        await cache_system.stop()