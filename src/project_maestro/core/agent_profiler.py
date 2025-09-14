"""에이전트 성능 프로파일링 시스템"""

import asyncio
import time
import cProfile
import pstats
import io
import psutil
import threading
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
from abc import ABC, abstractmethod
import sys
import gc

from project_maestro.core.base_agent import BaseAgent
from project_maestro.models.workflow_models import Task


class ProfilerType(Enum):
    """프로파일러 유형"""
    CPU = "cpu"                    # CPU 프로파일링
    MEMORY = "memory"              # 메모리 프로파일링
    TIME = "time"                  # 실행 시간 측정
    TOKEN = "token"                # 토큰 사용량
    API_CALLS = "api_calls"        # API 호출 추적
    COMPREHENSIVE = "comprehensive" # 종합 프로파일링


class PerformanceLevel(Enum):
    """성능 레벨"""
    EXCELLENT = "excellent"        # 90% 이상
    GOOD = "good"                 # 70-90%
    AVERAGE = "average"           # 50-70%
    POOR = "poor"                 # 30-50%
    CRITICAL = "critical"         # 30% 미만


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    api_call_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    error_count: int = 0
    success_count: int = 0
    throughput_ops_per_sec: float = 0.0
    latency_percentiles: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """프로파일링 결과"""
    
    agent_name: str
    task_id: str
    profiler_type: ProfilerType
    metrics: PerformanceMetrics
    cpu_profile: Optional[str] = None
    memory_profile: Optional[Dict[str, Any]] = None
    function_stats: Optional[List[Dict[str, Any]]] = None
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0


@dataclass
class PerformanceBaseline:
    """성능 기준선"""
    
    agent_name: str
    expected_execution_time: float
    max_memory_usage_mb: float
    max_token_usage: int
    min_success_rate: float
    max_error_rate: float
    baseline_timestamp: float = field(default_factory=time.time)


class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self):
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.thresholds = {
            'execution_time_multiplier': 2.0,      # 기준의 2배 이상이면 이상
            'memory_usage_multiplier': 1.5,        # 기준의 1.5배 이상이면 이상
            'token_usage_multiplier': 2.0,         # 기준의 2배 이상이면 이상
            'error_rate_threshold': 0.1,           # 10% 이상 에러율
            'success_rate_threshold': 0.9          # 90% 미만 성공률
        }
    
    def set_baseline(self, agent_name: str, baseline: PerformanceBaseline):
        """성능 기준선 설정"""
        self.baselines[agent_name] = baseline
    
    def detect_anomalies(
        self, 
        agent_name: str, 
        current_metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """성능 이상 감지"""
        
        anomalies = []
        
        # 기준선과 비교
        if agent_name in self.baselines:
            baseline = self.baselines[agent_name]
            
            # 실행 시간 이상
            if current_metrics.execution_time > baseline.expected_execution_time * self.thresholds['execution_time_multiplier']:
                anomalies.append({
                    'type': 'slow_execution',
                    'severity': 'high',
                    'current_value': current_metrics.execution_time,
                    'baseline_value': baseline.expected_execution_time,
                    'description': f'Execution time {current_metrics.execution_time:.2f}s exceeds baseline {baseline.expected_execution_time:.2f}s by {current_metrics.execution_time/baseline.expected_execution_time:.1f}x'
                })
            
            # 메모리 사용량 이상
            if current_metrics.memory_usage_mb > baseline.max_memory_usage_mb * self.thresholds['memory_usage_multiplier']:
                anomalies.append({
                    'type': 'high_memory',
                    'severity': 'medium',
                    'current_value': current_metrics.memory_usage_mb,
                    'baseline_value': baseline.max_memory_usage_mb,
                    'description': f'Memory usage {current_metrics.memory_usage_mb:.1f}MB exceeds baseline {baseline.max_memory_usage_mb:.1f}MB'
                })
            
            # 토큰 사용량 이상
            total_tokens = sum(current_metrics.token_usage.values())
            if total_tokens > baseline.max_token_usage * self.thresholds['token_usage_multiplier']:
                anomalies.append({
                    'type': 'excessive_tokens',
                    'severity': 'medium',
                    'current_value': total_tokens,
                    'baseline_value': baseline.max_token_usage,
                    'description': f'Token usage {total_tokens} exceeds baseline {baseline.max_token_usage}'
                })
        
        # 통계적 이상 감지 (이력 데이터 기반)
        if agent_name in self.historical_data:
            historical_metrics = list(self.historical_data[agent_name])
            if len(historical_metrics) >= 10:  # 최소 10개 데이터 포인트 필요
                statistical_anomalies = self._detect_statistical_anomalies(
                    current_metrics, 
                    historical_metrics
                )
                anomalies.extend(statistical_anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(
        self, 
        current: PerformanceMetrics, 
        historical: List[PerformanceMetrics]
    ) -> List[Dict[str, Any]]:
        """통계적 이상 감지"""
        
        anomalies = []
        
        # 실행 시간 Z-점수 계산
        exec_times = [m.execution_time for m in historical]
        if exec_times:
            mean_time = np.mean(exec_times)
            std_time = np.std(exec_times)
            
            if std_time > 0:
                z_score = (current.execution_time - mean_time) / std_time
                if abs(z_score) > 2.5:  # 2.5 표준편차 이상
                    anomalies.append({
                        'type': 'statistical_execution_time',
                        'severity': 'medium',
                        'z_score': z_score,
                        'current_value': current.execution_time,
                        'mean_value': mean_time,
                        'description': f'Execution time Z-score: {z_score:.2f} (>2.5σ from mean)'
                    })
        
        # 메모리 사용량 Z-점수
        memory_usage = [m.memory_usage_mb for m in historical]
        if memory_usage:
            mean_memory = np.mean(memory_usage)
            std_memory = np.std(memory_usage)
            
            if std_memory > 0:
                z_score = (current.memory_usage_mb - mean_memory) / std_memory
                if abs(z_score) > 2.0:
                    anomalies.append({
                        'type': 'statistical_memory_usage',
                        'severity': 'low',
                        'z_score': z_score,
                        'current_value': current.memory_usage_mb,
                        'mean_value': mean_memory,
                        'description': f'Memory usage Z-score: {z_score:.2f} (>2σ from mean)'
                    })
        
        return anomalies
    
    def generate_recommendations(
        self, 
        agent_name: str, 
        anomalies: List[Dict[str, Any]],
        historical_trend: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """최적화 권장사항 생성"""
        
        recommendations = []
        
        # 이상 항목별 권장사항
        for anomaly in anomalies:
            if anomaly['type'] in ['slow_execution', 'statistical_execution_time']:
                recommendations.append(
                    "Consider implementing result caching or optimizing computation-heavy operations"
                )
                recommendations.append(
                    "Review algorithm complexity and consider parallel processing for independent tasks"
                )
            
            elif anomaly['type'] in ['high_memory', 'statistical_memory_usage']:
                recommendations.append(
                    "Implement streaming responses to reduce memory footprint"
                )
                recommendations.append(
                    "Optimize data structures and consider memory-efficient alternatives"
                )
                recommendations.append(
                    "Add garbage collection optimization at key points"
                )
            
            elif anomaly['type'] == 'excessive_tokens':
                recommendations.append(
                    "Implement dynamic context pruning to reduce token usage"
                )
                recommendations.append(
                    "Use summary techniques for long conversation histories"
                )
                recommendations.append(
                    "Consider using smaller models for simpler tasks"
                )
        
        # 일반적인 성능 개선 권장사항
        if not anomalies:
            recommendations.append(
                "Performance is within normal range - consider proactive optimizations"
            )
        
        # 트렌드 기반 권장사항
        if historical_trend:
            if historical_trend.get('execution_time_trend', 0) > 0.1:
                recommendations.append(
                    "Execution time is trending upward - investigate for gradual performance degradation"
                )
            
            if historical_trend.get('memory_trend', 0) > 0.1:
                recommendations.append(
                    "Memory usage is trending upward - check for memory leaks or accumulating state"
                )
        
        return list(set(recommendations))  # 중복 제거
    
    def calculate_performance_score(
        self, 
        metrics: PerformanceMetrics,
        baseline: Optional[PerformanceBaseline] = None
    ) -> Tuple[float, PerformanceLevel]:
        """성능 점수 계산"""
        
        score_components = []
        
        # 성공률 점수 (0-40점)
        total_ops = metrics.success_count + metrics.error_count
        if total_ops > 0:
            success_rate = metrics.success_count / total_ops
            success_score = min(success_rate * 40, 40)
        else:
            success_score = 20  # 기본값
        score_components.append(success_score)
        
        # 처리량 점수 (0-25점)
        if metrics.throughput_ops_per_sec > 0:
            # 1 ops/sec을 기준으로 로그 스케일 적용
            throughput_score = min(np.log10(max(metrics.throughput_ops_per_sec, 0.1)) * 10 + 15, 25)
        else:
            throughput_score = 10
        score_components.append(throughput_score)
        
        # 리소스 효율성 점수 (0-25점)
        resource_score = 25
        if baseline:
            # 실행 시간 효율성
            if metrics.execution_time > 0 and baseline.expected_execution_time > 0:
                time_ratio = baseline.expected_execution_time / metrics.execution_time
                resource_score *= min(time_ratio, 1.0) * 0.5
            
            # 메모리 효율성
            if metrics.memory_usage_mb > 0 and baseline.max_memory_usage_mb > 0:
                memory_ratio = baseline.max_memory_usage_mb / max(metrics.memory_usage_mb, 1)
                resource_score *= min(memory_ratio, 1.0) * 0.5
        else:
            resource_score = 15  # 기준선이 없으면 중간 점수
        
        score_components.append(resource_score)
        
        # 캐시 효율성 점수 (0-10점)
        total_cache_ops = metrics.cache_hit_count + metrics.cache_miss_count
        if total_cache_ops > 0:
            cache_hit_rate = metrics.cache_hit_count / total_cache_ops
            cache_score = cache_hit_rate * 10
        else:
            cache_score = 5  # 기본값
        score_components.append(cache_score)
        
        # 총 점수 계산
        total_score = sum(score_components)
        
        # 성능 레벨 결정
        if total_score >= 90:
            level = PerformanceLevel.EXCELLENT
        elif total_score >= 70:
            level = PerformanceLevel.GOOD
        elif total_score >= 50:
            level = PerformanceLevel.AVERAGE
        elif total_score >= 30:
            level = PerformanceLevel.POOR
        else:
            level = PerformanceLevel.CRITICAL
        
        return total_score, level
    
    def add_historical_data(self, agent_name: str, metrics: PerformanceMetrics):
        """이력 데이터 추가"""
        self.historical_data[agent_name].append(metrics)
    
    def get_performance_trends(
        self, 
        agent_name: str, 
        window_size: int = 50
    ) -> Dict[str, float]:
        """성능 트렌드 분석"""
        
        if agent_name not in self.historical_data:
            return {}
        
        historical = list(self.historical_data[agent_name])
        if len(historical) < window_size:
            return {}
        
        recent_data = historical[-window_size:]
        older_data = historical[-window_size*2:-window_size] if len(historical) >= window_size*2 else historical[:-window_size]
        
        if not older_data:
            return {}
        
        trends = {}
        
        # 실행 시간 트렌드
        recent_exec_time = np.mean([m.execution_time for m in recent_data])
        older_exec_time = np.mean([m.execution_time for m in older_data])
        if older_exec_time > 0:
            trends['execution_time_trend'] = (recent_exec_time - older_exec_time) / older_exec_time
        
        # 메모리 사용량 트렌드
        recent_memory = np.mean([m.memory_usage_mb for m in recent_data])
        older_memory = np.mean([m.memory_usage_mb for m in older_data])
        if older_memory > 0:
            trends['memory_trend'] = (recent_memory - older_memory) / older_memory
        
        # 성공률 트렌드
        recent_success_rate = np.mean([
            m.success_count / max(m.success_count + m.error_count, 1) 
            for m in recent_data
        ])
        older_success_rate = np.mean([
            m.success_count / max(m.success_count + m.error_count, 1) 
            for m in older_data
        ])
        trends['success_rate_trend'] = recent_success_rate - older_success_rate
        
        return trends


class AgentPerformanceProfiler:
    """에이전트 성능 상세 프로파일링"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.active_profiles: Dict[str, Dict] = {}
        self.profile_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
    
    async def profile_agent_execution(
        self,
        agent: BaseAgent,
        task: Task,
        profiler_type: ProfilerType = ProfilerType.COMPREHENSIVE
    ) -> ProfileResult:
        """에이전트 실행 프로파일링"""
        
        profile_id = f"{agent.name}_{task.task_id}_{time.time()}"
        start_time = time.perf_counter()
        
        # 프로파일링 설정
        cpu_profiler = None
        memory_tracker = None
        initial_memory = None
        
        try:
            # CPU 프로파일링 시작
            if profiler_type in [ProfilerType.CPU, ProfilerType.COMPREHENSIVE]:
                cpu_profiler = cProfile.Profile()
                cpu_profiler.enable()
            
            # 메모리 추적 시작
            if profiler_type in [ProfilerType.MEMORY, ProfilerType.COMPREHENSIVE]:
                tracemalloc.start()
                initial_memory = tracemalloc.get_traced_memory()
            
            # 시스템 리소스 측정 시작
            process = psutil.Process()
            initial_cpu_percent = process.cpu_percent()
            
            # 활성 프로파일 등록
            with self._lock:
                self.active_profiles[profile_id] = {
                    'agent_name': agent.name,
                    'task_id': task.task_id,
                    'start_time': start_time,
                    'profiler_type': profiler_type
                }
            
            # 에이전트 실행
            initial_token_count = getattr(agent, 'token_count', 0)
            initial_api_calls = getattr(agent, 'api_call_count', 0)
            initial_cache_hits = getattr(agent, 'cache_hit_count', 0)
            initial_cache_misses = getattr(agent, 'cache_miss_count', 0)
            
            try:
                result = await agent.execute_task(task)
                execution_successful = True
                error_count = 0
            except Exception as e:
                result = None
                execution_successful = False
                error_count = 1
                print(f"Agent execution failed: {e}")
            
            end_time = time.perf_counter()
            execution_duration = end_time - start_time
            
            # 최종 메트릭 수집
            final_cpu_percent = process.cpu_percent()
            final_token_count = getattr(agent, 'token_count', initial_token_count)
            final_api_calls = getattr(agent, 'api_call_count', initial_api_calls)
            final_cache_hits = getattr(agent, 'cache_hit_count', initial_cache_hits)
            final_cache_misses = getattr(agent, 'cache_miss_count', initial_cache_misses)
            
            # CPU 프로파일링 중지
            cpu_profile_stats = None
            if cpu_profiler:
                cpu_profiler.disable()
                stats_stream = io.StringIO()
                stats = pstats.Stats(cpu_profiler, stream=stats_stream)
                stats.sort_stats('cumulative')
                stats.print_stats(20)  # 상위 20개 함수
                cpu_profile_stats = stats_stream.getvalue()
            
            # 메모리 프로파일링 결과
            memory_usage_mb = 0.0
            memory_profile_data = None
            if initial_memory:
                final_memory = tracemalloc.get_traced_memory()
                memory_usage_mb = (final_memory[0] - initial_memory[0]) / 1024 / 1024
                
                # 메모리 사용량 상위 10개
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                memory_profile_data = {
                    'peak_memory_mb': final_memory[1] / 1024 / 1024,
                    'current_memory_mb': final_memory[0] / 1024 / 1024,
                    'top_allocations': []
                }
                
                for stat in top_stats[:10]:
                    memory_profile_data['top_allocations'].append({
                        'file': stat.traceback.format(),
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count
                    })
                
                tracemalloc.stop()
            
            # 함수별 통계 추출
            function_stats = []
            if cpu_profiler:
                stats = pstats.Stats(cpu_profiler)
                stats.sort_stats('cumulative')
                
                for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                    filename, line, func_name = func
                    function_stats.append({
                        'function': f"{filename}:{line}({func_name})",
                        'call_count': cc,
                        'total_time': tt,
                        'cumulative_time': ct,
                        'per_call_time': ct / cc if cc > 0 else 0
                    })
            
            # 성능 메트릭 생성
            metrics = PerformanceMetrics(
                execution_time=execution_duration,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=(final_cpu_percent + initial_cpu_percent) / 2,
                token_usage={
                    'total': final_token_count - initial_token_count,
                    'input': getattr(agent, 'input_token_count', 0) - getattr(agent, '_initial_input_tokens', 0),
                    'output': getattr(agent, 'output_token_count', 0) - getattr(agent, '_initial_output_tokens', 0)
                },
                api_call_count=final_api_calls - initial_api_calls,
                cache_hit_count=final_cache_hits - initial_cache_hits,
                cache_miss_count=final_cache_misses - initial_cache_misses,
                error_count=error_count,
                success_count=1 if execution_successful else 0,
                throughput_ops_per_sec=1.0 / execution_duration if execution_duration > 0 else 0.0
            )
            
            # 이상 감지
            anomalies = self.analyzer.detect_anomalies(agent.name, metrics)
            
            # 권장사항 생성
            trends = self.analyzer.get_performance_trends(agent.name)
            recommendations = self.analyzer.generate_recommendations(agent.name, anomalies, trends)
            
            # 프로파일 결과 생성
            profile_result = ProfileResult(
                agent_name=agent.name,
                task_id=task.task_id,
                profiler_type=profiler_type,
                metrics=metrics,
                cpu_profile=cpu_profile_stats,
                memory_profile=memory_profile_data,
                function_stats=function_stats[:20],  # 상위 20개만
                anomalies=anomalies,
                recommendations=recommendations,
                timestamp=start_time,
                duration=execution_duration
            )
            
            # 이력 데이터 추가
            self.analyzer.add_historical_data(agent.name, metrics)
            self.profile_history[agent.name].append(profile_result)
            
            return profile_result
            
        except Exception as e:
            print(f"Profiling error: {e}")
            return ProfileResult(
                agent_name=agent.name,
                task_id=task.task_id,
                profiler_type=profiler_type,
                metrics=PerformanceMetrics(error_count=1),
                anomalies=[{
                    'type': 'profiling_error',
                    'severity': 'high',
                    'description': str(e)
                }],
                timestamp=start_time,
                duration=time.perf_counter() - start_time
            )
            
        finally:
            # 정리
            if cpu_profiler:
                cpu_profiler.disable()
            
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            with self._lock:
                self.active_profiles.pop(profile_id, None)
    
    async def profile_agent_continuously(
        self,
        agent: BaseAgent,
        duration_minutes: int = 60,
        sample_interval_seconds: int = 30
    ) -> List[ProfileResult]:
        """에이전트 연속 프로파일링"""
        
        results = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # 샘플 작업 생성 (실제 작업 시뮬레이션)
                sample_task = Task(
                    task_id=f"continuous_profile_{int(time.time())}",
                    content={"type": "sample_task", "operation": "health_check"},
                    priority=5
                )
                
                # 성능 프로파일링 실행
                result = await self.profile_agent_execution(
                    agent,
                    sample_task,
                    ProfilerType.TIME  # 연속 모니터링에서는 가벼운 프로파일링
                )
                
                results.append(result)
                
                # 다음 샘플까지 대기
                await asyncio.sleep(sample_interval_seconds)
                
            except Exception as e:
                print(f"Continuous profiling error: {e}")
                await asyncio.sleep(sample_interval_seconds)
        
        return results
    
    def set_performance_baseline(
        self, 
        agent_name: str,
        expected_execution_time: float,
        max_memory_usage_mb: float,
        max_token_usage: int,
        min_success_rate: float = 0.95,
        max_error_rate: float = 0.05
    ):
        """성능 기준선 설정"""
        
        baseline = PerformanceBaseline(
            agent_name=agent_name,
            expected_execution_time=expected_execution_time,
            max_memory_usage_mb=max_memory_usage_mb,
            max_token_usage=max_token_usage,
            min_success_rate=min_success_rate,
            max_error_rate=max_error_rate
        )
        
        self.analyzer.set_baseline(agent_name, baseline)
    
    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """에이전트 성능 요약"""
        
        if agent_name not in self.profile_history:
            return {"error": "No profiling history found for agent"}
        
        history = list(self.profile_history[agent_name])
        if not history:
            return {"error": "No profiling data available"}
        
        # 최근 프로파일 결과들 분석
        recent_profiles = history[-10:]  # 최근 10개
        
        # 평균 메트릭 계산
        avg_execution_time = np.mean([p.metrics.execution_time for p in recent_profiles])
        avg_memory_usage = np.mean([p.metrics.memory_usage_mb for p in recent_profiles])
        avg_token_usage = np.mean([sum(p.metrics.token_usage.values()) for p in recent_profiles])
        
        total_operations = sum(p.metrics.success_count + p.metrics.error_count for p in recent_profiles)
        total_successes = sum(p.metrics.success_count for p in recent_profiles)
        success_rate = total_successes / max(total_operations, 1)
        
        # 성능 점수 계산
        latest_metrics = recent_profiles[-1].metrics
        baseline = self.analyzer.baselines.get(agent_name)
        performance_score, performance_level = self.analyzer.calculate_performance_score(
            latest_metrics, baseline
        )
        
        # 트렌드 분석
        trends = self.analyzer.get_performance_trends(agent_name)
        
        # 이상 항목 집계
        all_anomalies = []
        for profile in recent_profiles:
            all_anomalies.extend(profile.anomalies)
        
        anomaly_counts = defaultdict(int)
        for anomaly in all_anomalies:
            anomaly_counts[anomaly['type']] += 1
        
        # 최신 권장사항
        latest_recommendations = recent_profiles[-1].recommendations if recent_profiles else []
        
        return {
            'agent_name': agent_name,
            'profile_count': len(history),
            'recent_profiles_analyzed': len(recent_profiles),
            'performance_summary': {
                'score': round(performance_score, 1),
                'level': performance_level.value,
                'avg_execution_time': round(avg_execution_time, 3),
                'avg_memory_usage_mb': round(avg_memory_usage, 2),
                'avg_token_usage': int(avg_token_usage),
                'success_rate': round(success_rate, 3)
            },
            'trends': trends,
            'anomaly_summary': dict(anomaly_counts),
            'recommendations': latest_recommendations,
            'baseline_set': agent_name in self.analyzer.baselines,
            'last_profiled': recent_profiles[-1].timestamp if recent_profiles else None
        }
    
    def get_comparative_analysis(self, agent_names: List[str]) -> Dict[str, Any]:
        """에이전트 간 비교 분석"""
        
        comparison_data = {}
        
        for agent_name in agent_names:
            summary = self.get_agent_performance_summary(agent_name)
            if 'error' not in summary:
                comparison_data[agent_name] = summary['performance_summary']
        
        if not comparison_data:
            return {"error": "No valid agent data for comparison"}
        
        # 비교 메트릭
        metrics_comparison = {}
        
        for metric in ['score', 'avg_execution_time', 'avg_memory_usage_mb', 'avg_token_usage', 'success_rate']:
            values = {agent: data[metric] for agent, data in comparison_data.items() if metric in data}
            
            if values:
                best_agent = max(values, key=values.get) if metric in ['score', 'success_rate'] else min(values, key=values.get)
                worst_agent = min(values, key=values.get) if metric in ['score', 'success_rate'] else max(values, key=values.get)
                
                metrics_comparison[metric] = {
                    'best_agent': best_agent,
                    'best_value': values[best_agent],
                    'worst_agent': worst_agent,
                    'worst_value': values[worst_agent],
                    'average': np.mean(list(values.values())),
                    'std_deviation': np.std(list(values.values()))
                }
        
        return {
            'agents_compared': agent_names,
            'comparison_data': comparison_data,
            'metrics_comparison': metrics_comparison,
            'timestamp': time.time()
        }
    
    def export_performance_report(
        self, 
        agent_name: str,
        format_type: str = "json"
    ) -> str:
        """성능 리포트 내보내기"""
        
        summary = self.get_agent_performance_summary(agent_name)
        
        if format_type.lower() == "json":
            return json.dumps(summary, indent=2)
        
        elif format_type.lower() == "markdown":
            # 마크다운 형식 리포트 생성
            if 'error' in summary:
                return f"# Performance Report\n\nError: {summary['error']}"
            
            report = f"""# Performance Report: {agent_name}

## Performance Summary
- **Score**: {summary['performance_summary']['score']}/100 ({summary['performance_summary']['level'].title()})
- **Average Execution Time**: {summary['performance_summary']['avg_execution_time']}s
- **Average Memory Usage**: {summary['performance_summary']['avg_memory_usage_mb']}MB
- **Average Token Usage**: {summary['performance_summary']['avg_token_usage']}
- **Success Rate**: {summary['performance_summary']['success_rate']*100:.1f}%

## Trends
"""
            
            for trend_name, trend_value in summary['trends'].items():
                direction = "↑" if trend_value > 0.05 else "↓" if trend_value < -0.05 else "→"
                report += f"- **{trend_name.replace('_', ' ').title()}**: {direction} {trend_value:+.2%}\n"
            
            if summary['anomaly_summary']:
                report += "\n## Recent Anomalies\n"
                for anomaly_type, count in summary['anomaly_summary'].items():
                    report += f"- **{anomaly_type.replace('_', ' ').title()}**: {count} occurrences\n"
            
            if summary['recommendations']:
                report += "\n## Recommendations\n"
                for i, recommendation in enumerate(summary['recommendations'], 1):
                    report += f"{i}. {recommendation}\n"
            
            return report
        
        else:
            return json.dumps({"error": f"Unsupported format: {format_type}"})
    
    def clear_agent_history(self, agent_name: str):
        """에이전트 히스토리 정리"""
        self.profile_history[agent_name].clear()
        if agent_name in self.analyzer.historical_data:
            self.analyzer.historical_data[agent_name].clear()
    
    def get_system_resource_usage(self) -> Dict[str, Any]:
        """시스템 리소스 사용량 조회"""
        
        process = psutil.Process()
        system = psutil
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_info': {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'percent': process.memory_percent()
            },
            'system_memory': {
                'total_mb': system.virtual_memory().total / 1024 / 1024,
                'available_mb': system.virtual_memory().available / 1024 / 1024,
                'percent_used': system.virtual_memory().percent
            },
            'system_cpu': {
                'percent': system.cpu_percent(),
                'count': system.cpu_count()
            },
            'disk_io': system.disk_io_counters()._asdict() if system.disk_io_counters() else {},
            'network_io': system.net_io_counters()._asdict() if system.net_io_counters() else {},
            'open_files': len(process.open_files()),
            'num_threads': process.num_threads(),
            'timestamp': time.time()
        }