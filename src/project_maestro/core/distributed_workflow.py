"""
분산 워크플로우 관리 시스템

다중 노드에서 워크플로우를 분산 실행하고 관리하는 시스템입니다.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
import aiohttp
import aioredis
from concurrent.futures import ThreadPoolExecutor
import hashlib

from ..utils.metrics import PrometheusMetrics
from ..core.base_agent import BaseAgent


class WorkflowStatus(Enum):
    """워크플로우 상태"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """태스크 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """노드 상태"""
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """로드 밸런싱 전략"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    CUSTOM = "custom"


@dataclass
class WorkerNode:
    """워커 노드"""
    node_id: str
    hostname: str
    port: int
    status: NodeStatus = NodeStatus.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_tasks: int = 0
    max_tasks: int = 10
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.hostname}:{self.port}"
    
    @property
    def is_available(self) -> bool:
        return (self.status in [NodeStatus.ACTIVE, NodeStatus.IDLE] and 
                self.current_tasks < self.max_tasks)
    
    @property
    def load_factor(self) -> float:
        """로드 팩터 (0.0 - 1.0)"""
        task_load = self.current_tasks / self.max_tasks
        resource_load = max(self.cpu_usage, self.memory_usage) / 100
        return max(task_load, resource_load)


@dataclass
class WorkflowTask:
    """워크플로우 태스크"""
    task_id: str
    workflow_id: str
    task_type: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 3600  # seconds
    priority: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_ready(self) -> bool:
        """의존성이 모두 완료되었는지 확인"""
        return self.status == TaskStatus.PENDING and not self.dependencies


@dataclass
class DistributedWorkflow:
    """분산 워크플로우"""
    workflow_id: str
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: WorkflowTask):
        """태스크 추가"""
        self.tasks[task.task_id] = task
    
    def get_ready_tasks(self) -> List[WorkflowTask]:
        """실행 준비된 태스크들"""
        return [task for task in self.tasks.values() if task.is_ready]
    
    def get_completed_tasks(self) -> List[WorkflowTask]:
        """완료된 태스크들"""
        return [task for task in self.tasks.values() 
                if task.status == TaskStatus.COMPLETED]
    
    def get_failed_tasks(self) -> List[WorkflowTask]:
        """실패한 태스크들"""
        return [task for task in self.tasks.values() 
                if task.status == TaskStatus.FAILED]
    
    @property
    def is_completed(self) -> bool:
        """워크플로우 완료 여부"""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] 
                  for task in self.tasks.values())
    
    @property
    def is_failed(self) -> bool:
        """워크플로우 실패 여부"""
        return any(task.status == TaskStatus.FAILED and task.retry_count >= task.max_retries 
                  for task in self.tasks.values())
    
    @property
    def completion_percentage(self) -> float:
        """완료 비율"""
        if not self.tasks:
            return 0.0
        
        completed = len(self.get_completed_tasks())
        return (completed / len(self.tasks)) * 100


class DistributedWorkflowManager:
    """분산 워크플로우 매니저"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.logger = structlog.get_logger()
        self.metrics = PrometheusMetrics()
        
        # Redis 연결
        self.redis_url = redis_url
        self.redis = None
        
        # 워크플로우 및 노드 관리
        self.workflows: Dict[str, DistributedWorkflow] = {}
        self.nodes: Dict[str, WorkerNode] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # 스케줄링 및 모니터링
        self.load_balancer = LoadBalancer()
        self.scheduler_task = None
        self.monitor_task = None
        self.heartbeat_task = None
        
        # 설정
        self.config = {
            'scheduler_interval': 1.0,  # 초
            'monitor_interval': 10.0,  # 초
            'heartbeat_interval': 30.0,  # 초
            'node_timeout': 120.0,  # 초
            'task_timeout': 3600,  # 초
            'max_concurrent_tasks': 1000,
            'retry_delay': 5.0  # 초
        }
        
        # HTTP 세션
        self.session = None
        
        # 실행 중인 태스크 추적
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """워크플로우 매니저 시작"""
        # Redis 연결
        self.redis = await aioredis.from_url(self.redis_url)
        
        # HTTP 세션 생성
        self.session = aiohttp.ClientSession()
        
        # 백그라운드 태스크 시작
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.logger.info("분산 워크플로우 매니저 시작됨")
    
    async def stop(self):
        """워크플로우 매니저 중지"""
        # 백그라운드 태스크 중지
        for task in [self.scheduler_task, self.monitor_task, self.heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 실행 중인 태스크들 취소
        for task in self.running_tasks.values():
            task.cancel()
        
        # 연결 종료
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()
        
        self.logger.info("분산 워크플로우 매니저 중지됨")
    
    async def register_node(self, node: WorkerNode) -> bool:
        """워커 노드 등록"""
        try:
            # 노드 상태 확인
            if await self._ping_node(node):
                self.nodes[node.node_id] = node
                node.status = NodeStatus.ACTIVE
                node.last_heartbeat = datetime.now()
                
                # Redis에 노드 정보 저장
                await self.redis.hset(
                    "workflow:nodes", 
                    node.node_id, 
                    json.dumps(asdict(node), default=str)
                )
                
                self.logger.info("워커 노드 등록됨", node_id=node.node_id, endpoint=node.endpoint)
                return True
            else:
                self.logger.warning("워커 노드 접근 불가", node_id=node.node_id, endpoint=node.endpoint)
                return False
                
        except Exception as e:
            self.logger.error("워커 노드 등록 실패", node_id=node.node_id, error=str(e))
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """워커 노드 등록 해제"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = NodeStatus.OFFLINE
            
            # 실행 중인 태스크들 다시 스케줄링
            await self._reschedule_node_tasks(node_id)
            
            # Redis에서 제거
            await self.redis.hdel("workflow:nodes", node_id)
            del self.nodes[node_id]
            
            self.logger.info("워커 노드 등록 해제됨", node_id=node_id)
            return True
        
        return False
    
    async def submit_workflow(self, workflow: DistributedWorkflow) -> str:
        """워크플로우 제출"""
        workflow.status = WorkflowStatus.PENDING
        workflow.started_at = datetime.now()
        
        self.workflows[workflow.workflow_id] = workflow
        
        # Redis에 워크플로우 저장
        await self.redis.hset(
            "workflow:workflows",
            workflow.workflow_id,
            json.dumps(asdict(workflow), default=str)
        )
        
        # 준비된 태스크들을 큐에 추가
        for task in workflow.get_ready_tasks():
            await self.task_queue.put((-task.priority, time.time(), task))
        
        self.logger.info(
            "워크플로우 제출됨",
            workflow_id=workflow.workflow_id,
            task_count=len(workflow.tasks)
        )
        
        return workflow.workflow_id
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """워크플로우 취소"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        
        # 실행 중인 태스크들 취소
        for task in workflow.tasks.values():
            if task.status == TaskStatus.RUNNING:
                await self._cancel_task(task)
        
        self.logger.info("워크플로우 취소됨", workflow_id=workflow_id)
        return True
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """워크플로우 상태 조회"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        return {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'completion_percentage': workflow.completion_percentage,
            'total_tasks': len(workflow.tasks),
            'completed_tasks': len(workflow.get_completed_tasks()),
            'failed_tasks': len(workflow.get_failed_tasks()),
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    async def _scheduler_loop(self):
        """스케줄러 루프"""
        while True:
            try:
                # 큐에서 태스크 가져오기
                try:
                    priority, timestamp, task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=self.config['scheduler_interval']
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 태스크 할당
                assigned_node = await self.load_balancer.select_node(
                    self.nodes, 
                    task.task_type
                )
                
                if assigned_node:
                    await self._assign_task(task, assigned_node)
                else:
                    # 사용 가능한 노드가 없으면 다시 큐에 넣기
                    await self.task_queue.put((priority, timestamp, task))
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error("스케줄러 오류", error=str(e))
                await asyncio.sleep(1)
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while True:
            try:
                await self._check_workflow_completion()
                await self._check_task_timeouts()
                await self._check_failed_tasks()
                
                # 메트릭 업데이트
                await self._update_metrics()
                
                await asyncio.sleep(self.config['monitor_interval'])
                
            except Exception as e:
                self.logger.error("모니터링 오류", error=str(e))
                await asyncio.sleep(5)
    
    async def _heartbeat_loop(self):
        """하트비트 루프"""
        while True:
            try:
                current_time = datetime.now()
                offline_nodes = []
                
                for node_id, node in self.nodes.items():
                    time_since_heartbeat = current_time - node.last_heartbeat
                    
                    if time_since_heartbeat.total_seconds() > self.config['node_timeout']:
                        offline_nodes.append(node_id)
                        continue
                    
                    # 노드 상태 업데이트
                    try:
                        status = await self._get_node_status(node)
                        if status:
                            node.cpu_usage = status.get('cpu_usage', 0)
                            node.memory_usage = status.get('memory_usage', 0)
                            node.current_tasks = status.get('current_tasks', 0)
                            node.last_heartbeat = current_time
                    except Exception as e:
                        self.logger.warning(
                            "노드 상태 업데이트 실패",
                            node_id=node_id,
                            error=str(e)
                        )
                
                # 오프라인 노드 처리
                for node_id in offline_nodes:
                    await self.unregister_node(node_id)
                
                await asyncio.sleep(self.config['heartbeat_interval'])
                
            except Exception as e:
                self.logger.error("하트비트 오류", error=str(e))
                await asyncio.sleep(5)
    
    async def _assign_task(self, task: WorkflowTask, node: WorkerNode):
        """태스크 할당"""
        task.status = TaskStatus.ASSIGNED
        task.assigned_node = node.node_id
        node.current_tasks += 1
        
        # 태스크 실행
        run_task = asyncio.create_task(self._execute_task(task, node))
        self.running_tasks[task.task_id] = run_task
        
        self.logger.info(
            "태스크 할당됨",
            task_id=task.task_id,
            node_id=node.node_id,
            task_type=task.task_type
        )
    
    async def _execute_task(self, task: WorkflowTask, node: WorkerNode):
        """태스크 실행"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # 워커 노드에 태스크 전송
            async with self.session.post(
                f"{node.endpoint}/execute",
                json={
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'payload': task.payload,
                    'timeout': task.timeout
                },
                timeout=aiohttp.ClientTimeout(total=task.timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    await self._complete_task(task, result)
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    await self._fail_task(task, error_msg)
        
        except asyncio.TimeoutError:
            await self._fail_task(task, "Task execution timeout")
        except Exception as e:
            await self._fail_task(task, str(e))
        finally:
            node.current_tasks -= 1
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    async def _complete_task(self, task: WorkflowTask, result: Dict[str, Any]):
        """태스크 완료 처리"""
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result
        
        # 의존성이 해결된 다른 태스크들을 큐에 추가
        workflow = self.workflows[task.workflow_id]
        await self._update_task_dependencies(workflow, task.task_id)
        
        self.logger.info(
            "태스크 완료됨",
            task_id=task.task_id,
            execution_time=task.execution_time
        )
    
    async def _fail_task(self, task: WorkflowTask, error: str):
        """태스크 실패 처리"""
        task.error = error
        task.retry_count += 1
        
        if task.retry_count < task.max_retries:
            task.status = TaskStatus.RETRYING
            # 재시도를 위해 지연 후 큐에 추가
            await asyncio.sleep(self.config['retry_delay'])
            await self.task_queue.put((-task.priority, time.time(), task))
            
            self.logger.warning(
                "태스크 재시도",
                task_id=task.task_id,
                retry_count=task.retry_count,
                error=error
            )
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            self.logger.error(
                "태스크 최종 실패",
                task_id=task.task_id,
                retry_count=task.retry_count,
                error=error
            )
    
    async def _cancel_task(self, task: WorkflowTask):
        """태스크 취소"""
        if task.task_id in self.running_tasks:
            self.running_tasks[task.task_id].cancel()
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        # 워커 노드에 취소 신호 전송
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            try:
                async with self.session.post(
                    f"{node.endpoint}/cancel",
                    json={'task_id': task.task_id}
                ) as response:
                    if response.status != 200:
                        self.logger.warning(
                            "태스크 취소 실패",
                            task_id=task.task_id,
                            node_id=task.assigned_node
                        )
            except Exception as e:
                self.logger.error("태스크 취소 오류", error=str(e))
    
    async def _update_task_dependencies(self, workflow: DistributedWorkflow, completed_task_id: str):
        """태스크 의존성 업데이트"""
        for task in workflow.tasks.values():
            if completed_task_id in task.dependencies:
                task.dependencies.remove(completed_task_id)
                
                if task.is_ready:
                    await self.task_queue.put((-task.priority, time.time(), task))
    
    async def _check_workflow_completion(self):
        """워크플로우 완료 확인"""
        for workflow in self.workflows.values():
            if workflow.status == WorkflowStatus.RUNNING:
                if workflow.is_completed:
                    workflow.status = WorkflowStatus.COMPLETED
                    workflow.completed_at = datetime.now()
                    
                    self.logger.info(
                        "워크플로우 완료됨",
                        workflow_id=workflow.workflow_id,
                        execution_time=(workflow.completed_at - workflow.started_at).total_seconds()
                    )
                elif workflow.is_failed:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.completed_at = datetime.now()
                    
                    self.logger.error(
                        "워크플로우 실패됨",
                        workflow_id=workflow.workflow_id
                    )
    
    async def _check_task_timeouts(self):
        """태스크 타임아웃 확인"""
        current_time = datetime.now()
        
        for workflow in self.workflows.values():
            for task in workflow.tasks.values():
                if (task.status == TaskStatus.RUNNING and 
                    task.started_at and 
                    (current_time - task.started_at).total_seconds() > task.timeout):
                    
                    await self._fail_task(task, "Task timeout")
    
    async def _check_failed_tasks(self):
        """실패한 태스크 확인"""
        # 구현: 실패한 태스크들에 대한 추가 처리
        pass
    
    async def _update_metrics(self):
        """메트릭 업데이트"""
        # 워크플로우 메트릭
        workflow_stats = {}
        for status in WorkflowStatus:
            count = sum(1 for w in self.workflows.values() if w.status == status)
            workflow_stats[status.value] = count
            self.metrics.gauge(f'workflows_{status.value}', count)
        
        # 노드 메트릭
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
        self.metrics.gauge('active_nodes', active_nodes)
        
        # 태스크 메트릭
        running_tasks = len(self.running_tasks)
        self.metrics.gauge('running_tasks', running_tasks)
    
    async def _ping_node(self, node: WorkerNode) -> bool:
        """노드 접근성 확인"""
        try:
            async with self.session.get(
                f"{node.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _get_node_status(self, node: WorkerNode) -> Optional[Dict[str, Any]]:
        """노드 상태 조회"""
        try:
            async with self.session.get(
                f"{node.endpoint}/status",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            pass
        return None
    
    async def _reschedule_node_tasks(self, node_id: str):
        """노드의 태스크들 재스케줄링"""
        for workflow in self.workflows.values():
            for task in workflow.tasks.values():
                if (task.assigned_node == node_id and 
                    task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]):
                    
                    task.assigned_node = None
                    task.status = TaskStatus.PENDING
                    await self.task_queue.put((-task.priority, time.time(), task))


class LoadBalancer:
    """로드 밸런서"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.round_robin_index = 0
    
    async def select_node(
        self, 
        nodes: Dict[str, WorkerNode], 
        task_type: str = None
    ) -> Optional[WorkerNode]:
        """노드 선택"""
        available_nodes = [
            node for node in nodes.values() 
            if node.is_available and (not task_type or task_type in node.capabilities)
        ]
        
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.CPU_BASED:
            return self._cpu_based_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.MEMORY_BASED:
            return self._memory_based_select(available_nodes)
        else:
            return available_nodes[0]  # 기본값
    
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """라운드 로빈 선택"""
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return node
    
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """최소 연결 선택"""
        return min(nodes, key=lambda n: n.current_tasks)
    
    def _cpu_based_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """CPU 기반 선택"""
        return min(nodes, key=lambda n: n.cpu_usage)
    
    def _memory_based_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """메모리 기반 선택"""
        return min(nodes, key=lambda n: n.memory_usage)


# 헬퍼 함수들
def create_workflow(name: str, created_by: str = None) -> DistributedWorkflow:
    """워크플로우 생성"""
    workflow_id = str(uuid.uuid4())
    return DistributedWorkflow(
        workflow_id=workflow_id,
        name=name,
        created_by=created_by
    )


def create_task(
    workflow_id: str,
    task_type: str,
    payload: Dict[str, Any],
    dependencies: List[str] = None,
    priority: int = 0,
    timeout: int = 3600
) -> WorkflowTask:
    """태스크 생성"""
    task_id = str(uuid.uuid4())
    return WorkflowTask(
        task_id=task_id,
        workflow_id=workflow_id,
        task_type=task_type,
        payload=payload,
        dependencies=dependencies or [],
        priority=priority,
        timeout=timeout
    )