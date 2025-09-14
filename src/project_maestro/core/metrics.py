"""Prometheus 메트릭 시스템"""

import time
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from dataclasses import dataclass
import threading
import asyncio


@dataclass
class MetricConfig:
    """메트릭 설정"""
    name: str
    description: str
    labels: List[str] = None


class PrometheusMetrics:
    """Prometheus 메트릭 수집 및 관리"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics = {}
        self._setup_default_metrics()
        self._server_started = False
    
    def _setup_default_metrics(self):
        """기본 메트릭 설정"""
        
        # 에이전트 실행 시간
        self.metrics['agent_execution_time'] = Histogram(
            'agent_execution_time_seconds',
            'Time spent executing agent tasks',
            ['agent', 'event_type']
        )
        
        # 에이전트 이벤트 카운터
        self.metrics['agent_events_total'] = Counter(
            'agent_events_total',
            'Total number of agent events',
            ['agent', 'event_type', 'status']
        )
        
        # 에러 카운터
        self.metrics['agent_errors_total'] = Counter(
            'agent_errors_total',
            'Total number of agent errors',
            ['agent']
        )
        
        # 활성 워크플로우 수
        self.metrics['active_workflows'] = Gauge(
            'active_workflows',
            'Number of currently active workflows'
        )
        
        # 메모리 사용량
        self.metrics['memory_usage_bytes'] = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        # 토큰 사용량
        self.metrics['token_usage_total'] = Counter(
            'token_usage_total',
            'Total tokens used',
            ['agent', 'type']  # type: input, output
        )
        
        # 캐시 히트율
        self.metrics['cache_hits_total'] = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.metrics['cache_misses_total'] = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        # 처리량 (throughput)
        self.metrics['requests_per_second'] = Gauge(
            'requests_per_second',
            'Current requests per second'
        )
        
        # 응답 시간 분위수
        self.metrics['response_time_quantiles'] = Histogram(
            'response_time_seconds',
            'Response time distribution',
            ['endpoint']
        )
        
        # 시스템 정보
        self.metrics['system_info'] = Info(
            'system_info',
            'System information'
        )
        
        # 워크플로우 상태별 카운터
        self.metrics['workflow_status_total'] = Counter(
            'workflow_status_total',
            'Total workflows by status',
            ['status']  # pending, running, completed, failed
        )
    
    def start_server(self):
        """메트릭 서버 시작"""
        if not self._server_started:
            start_http_server(self.port)
            self._server_started = True
            print(f"Prometheus metrics server started on port {self.port}")
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None):
        """카운터 증가"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def observe_histogram(
        self, 
        metric_name: str, 
        value: float, 
        labels: Dict[str, str] = None
    ):
        """히스토그램 관찰"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def set_gauge(
        self, 
        metric_name: str, 
        value: float, 
        labels: Dict[str, str] = None
    ):
        """게이지 값 설정"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def inc_gauge(
        self, 
        metric_name: str, 
        amount: float = 1, 
        labels: Dict[str, str] = None
    ):
        """게이지 값 증가"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).inc(amount)
            else:
                metric.inc(amount)
    
    def dec_gauge(
        self, 
        metric_name: str, 
        amount: float = 1, 
        labels: Dict[str, str] = None
    ):
        """게이지 값 감소"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).dec(amount)
            else:
                metric.dec(amount)
    
    def set_info(self, metric_name: str, info: Dict[str, str]):
        """정보 메트릭 설정"""
        if metric_name in self.metrics:
            self.metrics[metric_name].info(info)
    
    def time_function(self, metric_name: str, labels: Dict[str, str] = None):
        """함수 실행 시간 측정 데코레이터"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    self.observe_histogram(metric_name, duration, labels)
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    self.observe_histogram(metric_name, duration, labels)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def track_workflow_status(self, status: str):
        """워크플로우 상태 추적"""
        self.increment_counter('workflow_status_total', {'status': status})
    
    def track_token_usage(self, agent: str, tokens: int, token_type: str = 'total'):
        """토큰 사용량 추적"""
        self.increment_counter('token_usage_total', {
            'agent': agent,
            'type': token_type
        })
        
        # 토큰 수만큼 카운터 증가
        for _ in range(tokens):
            self.increment_counter('token_usage_total', {
                'agent': agent,
                'type': token_type
            })
    
    def track_cache_performance(self, cache_type: str, hit: bool):
        """캐시 성능 추적"""
        if hit:
            self.increment_counter('cache_hits_total', {'cache_type': cache_type})
        else:
            self.increment_counter('cache_misses_total', {'cache_type': cache_type})
    
    def update_throughput(self, rps: float):
        """처리량 업데이트"""
        self.set_gauge('requests_per_second', rps)
    
    def track_memory_usage(self, component: str, bytes_used: int):
        """메모리 사용량 추적"""
        self.set_gauge('memory_usage_bytes', bytes_used, {'component': component})
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭 요약 조회"""
        summary = {}
        
        for name, metric in self.metrics.items():
            try:
                # 메트릭 타입에 따른 처리
                if hasattr(metric, '_value'):
                    # Counter/Gauge
                    summary[name] = metric._value.get()
                elif hasattr(metric, '_sum'):
                    # Histogram
                    summary[name] = {
                        'count': metric._count.get(),
                        'sum': metric._sum.get()
                    }
                elif hasattr(metric, '_info'):
                    # Info
                    summary[name] = metric._info
            except Exception as e:
                summary[name] = f"Error retrieving metric: {e}"
        
        return summary


class MetricsCollector:
    """비동기 메트릭 수집기"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self.running = False
        self.collection_interval = 30  # 30초마다 수집
    
    async def start_collection(self):
        """메트릭 수집 시작"""
        self.running = True
        
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def stop_collection(self):
        """메트릭 수집 중지"""
        self.running = False
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        import psutil
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent()
        self.metrics.set_gauge('system_cpu_percent', cpu_percent)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        self.metrics.set_gauge('system_memory_percent', memory.percent)
        self.metrics.set_gauge('system_memory_bytes', memory.used)
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        self.metrics.set_gauge('system_disk_percent', disk.percent)
        self.metrics.set_gauge('system_disk_bytes', disk.used)
        
        # 네트워크 통계
        network = psutil.net_io_counters()
        self.metrics.set_gauge('system_network_bytes_sent', network.bytes_sent)
        self.metrics.set_gauge('system_network_bytes_recv', network.bytes_recv)