"""에이전트 협업 프로토콜 강화 시스템"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, deque

from project_maestro.core.base_agent import BaseAgent
from project_maestro.models.workflow_models import Task, WorkflowRequest


class CollaborationType(Enum):
    """협업 유형"""
    SEQUENTIAL = "sequential"        # 순차적 실행
    PARALLEL = "parallel"           # 병렬 실행  
    HIERARCHICAL = "hierarchical"   # 계층적 위임
    CONSENSUS = "consensus"         # 합의 기반
    PIPELINE = "pipeline"           # 파이프라인
    SWARM = "swarm"                # 스웜 인텔리전스


class AgentRole(Enum):
    """에이전트 역할"""
    COORDINATOR = "coordinator"     # 조정자
    EXECUTOR = "executor"          # 실행자
    VALIDATOR = "validator"        # 검증자
    SPECIALIST = "specialist"      # 전문가
    MONITOR = "monitor"           # 모니터


class TaskPriority(Enum):
    """작업 우선순위"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class CollaborationProtocol:
    """협업 프로토콜 정의"""
    
    type: CollaborationType
    agents: List[str]
    coordination_rules: Dict[str, Any]
    conflict_resolution: str = "majority_vote"
    timeout: float = 300.0  # 5분
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    quality_gates: List[str] = field(default_factory=list)


@dataclass 
class CollaborationTask:
    """협업 작업"""
    
    id: str
    content: Dict[str, Any]
    priority: TaskPriority
    assigned_agents: List[str]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 60.0
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """에이전트 역량 정보"""
    
    agent_name: str
    expertise_domains: List[str]
    skill_ratings: Dict[str, float]  # 0.0 - 1.0
    availability: float = 1.0        # 0.0 - 1.0
    current_load: float = 0.0        # 0.0 - 1.0 
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)


@dataclass
class CollaborationResult:
    """협업 결과"""
    
    task_id: str
    success: bool
    results: Dict[str, Any]
    participating_agents: List[str]
    execution_time: float
    quality_score: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolver(ABC):
    """충돌 해결 추상 클래스"""
    
    @abstractmethod
    async def resolve(
        self, 
        conflicting_results: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        pass


class MajorityVoteResolver(ConflictResolver):
    """다수결 충돌 해결"""
    
    async def resolve(
        self, 
        conflicting_results: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        if not conflicting_results:
            return None
        
        # 결과 빈도 계산
        result_counts = defaultdict(int)
        for agent, result in conflicting_results.items():
            result_key = json.dumps(result, sort_keys=True) if isinstance(result, dict) else str(result)
            result_counts[result_key] += 1
        
        # 가장 많이 선택된 결과 반환
        most_common = max(result_counts.items(), key=lambda x: x[1])
        return json.loads(most_common[0]) if most_common[0].startswith('{') else most_common[0]


class WeightedVoteResolver(ConflictResolver):
    """가중 투표 충돌 해결"""
    
    def __init__(self, agent_weights: Dict[str, float]):
        self.agent_weights = agent_weights
    
    async def resolve(
        self, 
        conflicting_results: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        if not conflicting_results:
            return None
        
        # 가중 점수 계산
        weighted_scores = defaultdict(float)
        
        for agent, result in conflicting_results.items():
            weight = self.agent_weights.get(agent, 1.0)
            result_key = json.dumps(result, sort_keys=True) if isinstance(result, dict) else str(result)
            weighted_scores[result_key] += weight
        
        # 가장 높은 가중 점수를 받은 결과 반환
        best_result = max(weighted_scores.items(), key=lambda x: x[1])
        return json.loads(best_result[0]) if best_result[0].startswith('{') else best_result[0]


class ConsensusResolver(ConflictResolver):
    """합의 기반 충돌 해결"""
    
    async def resolve(
        self, 
        conflicting_results: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        # 간단한 합의 알고리즘: 결과들의 평균 또는 교집합
        if not conflicting_results:
            return None
        
        results = list(conflicting_results.values())
        
        # 숫자 결과의 경우 평균
        if all(isinstance(r, (int, float)) for r in results):
            return sum(results) / len(results)
        
        # 리스트 결과의 경우 교집합
        elif all(isinstance(r, list) for r in results):
            if results:
                common_elements = set(results[0])
                for result in results[1:]:
                    common_elements &= set(result)
                return list(common_elements)
        
        # 기본적으로 첫 번째 결과 반환
        return results[0] if results else None


class EnhancedAgentCollaboration:
    """향상된 에이전트 협업 시스템"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.collaboration_history: List[CollaborationResult] = []
        self.active_collaborations: Dict[str, Dict] = {}
        
        # 충돌 해결자들
        self.conflict_resolvers = {
            'majority_vote': MajorityVoteResolver(),
            'weighted_vote': WeightedVoteResolver({}),
            'consensus': ConsensusResolver()
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            'collaboration_success_rate': 0.95,
            'avg_execution_time': 0.0,
            'agent_utilization': defaultdict(float)
        }
    
    def register_agent(self, agent: BaseAgent, capability: AgentCapability):
        """에이전트 등록"""
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = capability
    
    async def negotiate_task_distribution(
        self,
        task: CollaborationTask,
        available_agents: List[str],
        protocol: CollaborationProtocol
    ) -> Dict[str, List[CollaborationTask]]:
        """작업 분배 협상"""
        
        # 에이전트 능력 평가
        agent_scores = {}
        
        for agent_name in available_agents:
            if agent_name not in self.agent_capabilities:
                continue
            
            capability = self.agent_capabilities[agent_name]
            score = await self._calculate_agent_score(task, capability)
            agent_scores[agent_name] = score
        
        # 작업 복잡도 분석
        complexity_score = await self._analyze_task_complexity(task)
        
        # 협업 유형에 따른 분배
        if protocol.type == CollaborationType.SEQUENTIAL:
            return await self._sequential_distribution(task, agent_scores, protocol)
        elif protocol.type == CollaborationType.PARALLEL:
            return await self._parallel_distribution(task, agent_scores, protocol)
        elif protocol.type == CollaborationType.HIERARCHICAL:
            return await self._hierarchical_distribution(task, agent_scores, protocol)
        else:
            # 기본 분배
            best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
            return {best_agent: [task]}
    
    async def _calculate_agent_score(
        self, 
        task: CollaborationTask, 
        capability: AgentCapability
    ) -> float:
        """에이전트 점수 계산"""
        
        # 기본 점수
        base_score = 0.5
        
        # 전문성 점수 (작업과 관련된 도메인 전문성)
        expertise_score = 0.0
        task_domains = task.context.get('domains', [])
        
        for domain in task_domains:
            if domain in capability.expertise_domains:
                domain_rating = capability.skill_ratings.get(domain, 0.5)
                expertise_score += domain_rating
        
        if task_domains:
            expertise_score /= len(task_domains)
        
        # 가용성 점수
        availability_score = capability.availability * (1 - capability.current_load)
        
        # 성능 이력 점수
        performance_score = 0.0
        if capability.performance_history:
            recent_performance = []
            for metric, history in capability.performance_history.items():
                if history:
                    recent_performance.extend(history[-5:])  # 최근 5개
            
            if recent_performance:
                performance_score = sum(recent_performance) / len(recent_performance)
        
        # 가중 평균 계산
        weights = {
            'base': 0.1,
            'expertise': 0.4,
            'availability': 0.3,
            'performance': 0.2
        }
        
        total_score = (
            weights['base'] * base_score +
            weights['expertise'] * expertise_score +
            weights['availability'] * availability_score +
            weights['performance'] * performance_score
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    async def _analyze_task_complexity(self, task: CollaborationTask) -> float:
        """작업 복잡도 분석"""
        
        complexity_factors = {
            'dependencies': len(task.dependencies) * 0.1,
            'estimated_duration': min(task.estimated_duration / 3600, 1.0) * 0.3,  # 시간 기반
            'priority': (5 - task.priority.value) / 4 * 0.2,  # 우선순위 기반
            'quality_requirements': len(task.quality_requirements) * 0.05,
            'content_size': min(len(str(task.content)) / 10000, 1.0) * 0.1
        }
        
        # 복합 복잡도 점수
        total_complexity = sum(complexity_factors.values())
        return min(max(total_complexity, 0.0), 1.0)
    
    async def _sequential_distribution(
        self, 
        task: CollaborationTask,
        agent_scores: Dict[str, float],
        protocol: CollaborationProtocol
    ) -> Dict[str, List[CollaborationTask]]:
        """순차적 작업 분배"""
        
        # 점수순으로 정렬
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 작업을 단계별로 분해
        subtasks = await self._decompose_task(task, len(sorted_agents))
        
        distribution = {}
        for i, (agent_name, score) in enumerate(sorted_agents):
            if i < len(subtasks):
                distribution[agent_name] = [subtasks[i]]
        
        return distribution
    
    async def _parallel_distribution(
        self, 
        task: CollaborationTask,
        agent_scores: Dict[str, float],
        protocol: CollaborationProtocol
    ) -> Dict[str, List[CollaborationTask]]:
        """병렬 작업 분배"""
        
        # 병렬 처리 가능한 하위 작업 생성
        parallel_subtasks = await self._create_parallel_subtasks(task)
        
        # 에이전트에게 균등 분배
        distribution = {}
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, subtask in enumerate(parallel_subtasks):
            agent_name = sorted_agents[i % len(sorted_agents)][0]
            if agent_name not in distribution:
                distribution[agent_name] = []
            distribution[agent_name].append(subtask)
        
        return distribution
    
    async def _hierarchical_distribution(
        self, 
        task: CollaborationTask,
        agent_scores: Dict[str, float],
        protocol: CollaborationProtocol
    ) -> Dict[str, List[CollaborationTask]]:
        """계층적 작업 분배"""
        
        # 최고 점수 에이전트를 코디네이터로 선정
        coordinator = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        # 나머지 에이전트들을 실행자로 배정
        executors = [agent for agent in agent_scores.keys() if agent != coordinator]
        
        # 코디네이터 작업 생성
        coordination_task = CollaborationTask(
            id=f"{task.id}_coordination",
            content={
                "type": "coordination",
                "original_task": task.content,
                "executors": executors
            },
            priority=task.priority,
            assigned_agents=[coordinator]
        )
        
        # 실행자 작업들 생성
        executor_tasks = []
        for executor in executors:
            executor_task = CollaborationTask(
                id=f"{task.id}_execution_{executor}",
                content={
                    "type": "execution", 
                    "assigned_by": coordinator,
                    "task_subset": task.content
                },
                priority=task.priority,
                assigned_agents=[executor]
            )
            executor_tasks.append(executor_task)
        
        distribution = {coordinator: [coordination_task]}
        for i, executor in enumerate(executors):
            distribution[executor] = [executor_tasks[i]]
        
        return distribution
    
    async def consensus_decision(
        self,
        agents: List[str],
        proposal: Dict[str, Any],
        decision_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """합의 기반 의사결정"""
        
        votes = []
        voting_timeout = 30.0  # 30초 투표 제한시간
        
        # 병렬 투표 수집
        async def collect_vote(agent_name: str):
            try:
                agent = self.agents[agent_name]
                
                # 에이전트에게 제안 평가 요청
                evaluation_task = {
                    "type": "proposal_evaluation",
                    "proposal": proposal,
                    "context": decision_context or {},
                    "voting_agent": agent_name
                }
                
                # 타임아웃 설정하여 평가 실행
                vote_result = await asyncio.wait_for(
                    agent.evaluate_proposal(evaluation_task),
                    timeout=voting_timeout
                )
                
                return {
                    'agent': agent_name,
                    'decision': vote_result.get('decision', 'abstain'),
                    'confidence': vote_result.get('confidence', 0.5),
                    'reasoning': vote_result.get('reasoning', ''),
                    'timestamp': time.time()
                }
                
            except asyncio.TimeoutError:
                return {
                    'agent': agent_name,
                    'decision': 'abstain',
                    'confidence': 0.0,
                    'reasoning': 'Timeout during voting',
                    'timestamp': time.time()
                }
            except Exception as e:
                return {
                    'agent': agent_name,
                    'decision': 'error',
                    'confidence': 0.0,
                    'reasoning': f'Error during voting: {str(e)}',
                    'timestamp': time.time()
                }
        
        # 모든 에이전트로부터 병렬로 투표 수집
        vote_tasks = [collect_vote(agent_name) for agent_name in agents]
        votes = await asyncio.gather(*vote_tasks)
        
        # 투표 집계
        decision_counts = defaultdict(int)
        total_confidence = 0.0
        valid_votes = 0
        
        for vote in votes:
            if vote['decision'] not in ['abstain', 'error']:
                decision_counts[vote['decision']] += 1
                total_confidence += vote['confidence']
                valid_votes += 1
        
        # 결과 계산
        if not decision_counts:
            return {
                'decision': 'no_consensus',
                'confidence': 0.0,
                'votes': votes,
                'reasoning': 'No valid votes received'
            }
        
        # 다수결 결정
        winning_decision = max(decision_counts.items(), key=lambda x: x[1])
        avg_confidence = total_confidence / valid_votes if valid_votes > 0 else 0.0
        
        # 합의 강도 계산 (최고 득표 / 총 투표)
        consensus_strength = winning_decision[1] / len(agents)
        
        return {
            'decision': winning_decision[0],
            'confidence': avg_confidence,
            'consensus_strength': consensus_strength,
            'vote_distribution': dict(decision_counts),
            'votes': votes,
            'reasoning': f'Decision reached with {consensus_strength:.2%} consensus'
        }
    
    async def execute_collaborative_workflow(
        self,
        task: CollaborationTask,
        protocol: CollaborationProtocol
    ) -> CollaborationResult:
        """협업 워크플로우 실행"""
        
        start_time = time.time()
        collaboration_id = str(uuid.uuid4())
        
        # 활성 협업 등록
        self.active_collaborations[collaboration_id] = {
            'task': task,
            'protocol': protocol,
            'start_time': start_time,
            'status': 'running'
        }
        
        try:
            # 작업 분배
            available_agents = [
                name for name, cap in self.agent_capabilities.items()
                if cap.availability > 0.1 and cap.current_load < 0.9
            ]
            
            task_distribution = await self.negotiate_task_distribution(
                task, available_agents, protocol
            )
            
            # 협업 실행
            if protocol.type == CollaborationType.SEQUENTIAL:
                results = await self._execute_sequential(task_distribution, protocol)
            elif protocol.type == CollaborationType.PARALLEL:
                results = await self._execute_parallel(task_distribution, protocol)
            elif protocol.type == CollaborationType.CONSENSUS:
                results = await self._execute_consensus(task_distribution, protocol)
            else:
                results = await self._execute_default(task_distribution, protocol)
            
            # 품질 검증
            quality_score = await self._validate_collaboration_quality(results, task)
            
            # 결과 통합
            final_result = await self._integrate_results(results, protocol)
            
            execution_time = time.time() - start_time
            
            collaboration_result = CollaborationResult(
                task_id=task.id,
                success=True,
                results=final_result,
                participating_agents=list(task_distribution.keys()),
                execution_time=execution_time,
                quality_score=quality_score
            )
            
            # 히스토리 업데이트
            self.collaboration_history.append(collaboration_result)
            
            # 에이전트 성능 업데이트
            await self._update_agent_performance(task_distribution, collaboration_result)
            
            return collaboration_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            collaboration_result = CollaborationResult(
                task_id=task.id,
                success=False,
                results={},
                participating_agents=list(task_distribution.keys()) if 'task_distribution' in locals() else [],
                execution_time=execution_time,
                quality_score=0.0,
                errors=[str(e)]
            )
            
            self.collaboration_history.append(collaboration_result)
            return collaboration_result
            
        finally:
            # 활성 협업 정리
            if collaboration_id in self.active_collaborations:
                del self.active_collaborations[collaboration_id]
    
    async def _decompose_task(self, task: CollaborationTask, num_parts: int) -> List[CollaborationTask]:
        """작업 분해"""
        subtasks = []
        
        for i in range(min(num_parts, 5)):  # 최대 5개로 제한
            subtask = CollaborationTask(
                id=f"{task.id}_subtask_{i}",
                content={
                    "parent_task": task.id,
                    "subtask_index": i,
                    "content": f"Subtask {i+1} of {task.content.get('description', 'main task')}"
                },
                priority=task.priority,
                assigned_agents=[]
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _create_parallel_subtasks(self, task: CollaborationTask) -> List[CollaborationTask]:
        """병렬 처리 가능한 하위 작업 생성"""
        # 간단한 병렬 분해 로직
        return await self._decompose_task(task, 3)
    
    async def _execute_parallel(
        self, 
        task_distribution: Dict[str, List[CollaborationTask]], 
        protocol: CollaborationProtocol
    ) -> Dict[str, Any]:
        """병렬 실행"""
        
        async def execute_agent_tasks(agent_name: str, tasks: List[CollaborationTask]):
            agent = self.agents[agent_name]
            results = []
            
            for task in tasks:
                try:
                    result = await agent.execute_task(task)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "task_id": task.id})
            
            return {agent_name: results}
        
        # 모든 에이전트 작업을 병렬 실행
        execution_tasks = [
            execute_agent_tasks(agent, tasks) 
            for agent, tasks in task_distribution.items()
        ]
        
        parallel_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # 결과 통합
        combined_results = {}
        for result in parallel_results:
            if isinstance(result, dict):
                combined_results.update(result)
            else:
                # 예외 처리
                combined_results[f"error_{len(combined_results)}"] = str(result)
        
        return combined_results
    
    async def _execute_sequential(
        self, 
        task_distribution: Dict[str, List[CollaborationTask]], 
        protocol: CollaborationProtocol
    ) -> Dict[str, Any]:
        """순차 실행"""
        
        results = {}
        previous_result = None
        
        for agent_name, tasks in task_distribution.items():
            agent = self.agents[agent_name]
            
            for task in tasks:
                # 이전 결과를 컨텍스트로 추가
                if previous_result:
                    task.context['previous_result'] = previous_result
                
                try:
                    result = await agent.execute_task(task)
                    results[f"{agent_name}_{task.id}"] = result
                    previous_result = result
                except Exception as e:
                    error_result = {"error": str(e), "task_id": task.id}
                    results[f"{agent_name}_{task.id}"] = error_result
                    previous_result = error_result
        
        return results
    
    async def _execute_consensus(
        self, 
        task_distribution: Dict[str, List[CollaborationTask]], 
        protocol: CollaborationProtocol
    ) -> Dict[str, Any]:
        """합의 기반 실행"""
        
        # 모든 에이전트가 같은 작업 수행
        agent_results = {}
        
        for agent_name in task_distribution.keys():
            agent = self.agents[agent_name]
            
            # 첫 번째 작업만 사용 (합의를 위해)
            task = list(task_distribution[agent_name])[0] if task_distribution[agent_name] else None
            
            if task:
                try:
                    result = await agent.execute_task(task)
                    agent_results[agent_name] = result
                except Exception as e:
                    agent_results[agent_name] = {"error": str(e)}
        
        # 충돌 해결
        resolver = self.conflict_resolvers.get(protocol.conflict_resolution, self.conflict_resolvers['majority_vote'])
        consensus_result = await resolver.resolve(agent_results, {})
        
        return {
            'consensus_result': consensus_result,
            'individual_results': agent_results
        }
    
    async def _execute_default(
        self, 
        task_distribution: Dict[str, List[CollaborationTask]], 
        protocol: CollaborationProtocol
    ) -> Dict[str, Any]:
        """기본 실행 (단일 에이전트)"""
        
        if not task_distribution:
            return {}
        
        # 첫 번째 에이전트만 사용
        agent_name, tasks = next(iter(task_distribution.items()))
        agent = self.agents[agent_name]
        
        results = {}
        for task in tasks:
            try:
                result = await agent.execute_task(task)
                results[task.id] = result
            except Exception as e:
                results[task.id] = {"error": str(e)}
        
        return results
    
    async def _validate_collaboration_quality(
        self, 
        results: Dict[str, Any], 
        original_task: CollaborationTask
    ) -> float:
        """협업 품질 검증"""
        
        quality_factors = []
        
        # 성공률
        total_results = len(results)
        successful_results = len([r for r in results.values() if not isinstance(r, dict) or "error" not in r])
        success_rate = successful_results / total_results if total_results > 0 else 0.0
        quality_factors.append(success_rate * 0.4)
        
        # 완성도 (결과에 필요한 정보가 포함되어 있는지)
        completeness = 0.8  # 기본값
        if original_task.quality_requirements:
            # 품질 요구사항 체크 로직
            met_requirements = 0
            for requirement in original_task.quality_requirements:
                # 간단한 키워드 기반 체크
                if any(str(requirement).lower() in str(result).lower() for result in results.values()):
                    met_requirements += 1
            completeness = met_requirements / len(original_task.quality_requirements)
        quality_factors.append(completeness * 0.3)
        
        # 일관성 (결과들 간의 일관성)
        consistency = 0.7  # 기본값
        if len(results) > 1:
            # 결과의 유사성 간단 체크
            result_strings = [str(r) for r in results.values() if "error" not in str(r)]
            if len(result_strings) > 1:
                avg_length = sum(len(s) for s in result_strings) / len(result_strings)
                length_variance = sum((len(s) - avg_length) ** 2 for s in result_strings) / len(result_strings)
                consistency = max(0, 1 - length_variance / (avg_length ** 2)) if avg_length > 0 else 0.5
        quality_factors.append(consistency * 0.3)
        
        return sum(quality_factors)
    
    async def _integrate_results(
        self, 
        results: Dict[str, Any], 
        protocol: CollaborationProtocol
    ) -> Dict[str, Any]:
        """결과 통합"""
        
        if protocol.type == CollaborationType.CONSENSUS:
            return results  # 이미 합의된 결과
        
        # 기본 통합: 모든 결과를 포함
        integrated = {
            'type': 'integrated_results',
            'collaboration_type': protocol.type.value,
            'individual_results': results,
            'summary': self._create_result_summary(results)
        }
        
        return integrated
    
    def _create_result_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """결과 요약 생성"""
        
        total_results = len(results)
        error_count = len([r for r in results.values() if isinstance(r, dict) and "error" in r])
        success_count = total_results - error_count
        
        return {
            'total_results': total_results,
            'successful_results': success_count,
            'failed_results': error_count,
            'success_rate': success_count / total_results if total_results > 0 else 0.0
        }
    
    async def _update_agent_performance(
        self, 
        task_distribution: Dict[str, List[CollaborationTask]], 
        result: CollaborationResult
    ):
        """에이전트 성능 업데이트"""
        
        for agent_name in task_distribution.keys():
            if agent_name in self.agent_capabilities:
                capability = self.agent_capabilities[agent_name]
                
                # 성능 히스토리 업데이트
                if 'collaboration_success' not in capability.performance_history:
                    capability.performance_history['collaboration_success'] = deque(maxlen=50)
                
                success_score = 1.0 if result.success and result.quality_score > 0.7 else 0.0
                capability.performance_history['collaboration_success'].append(success_score)
                
                # 현재 부하 감소 (작업 완료)
                capability.current_load = max(0.0, capability.current_load - 0.1)
    
    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """협업 분석 정보"""
        
        if not self.collaboration_history:
            return {"message": "No collaboration history available"}
        
        total_collaborations = len(self.collaboration_history)
        successful_collaborations = len([r for r in self.collaboration_history if r.success])
        
        avg_execution_time = sum(r.execution_time for r in self.collaboration_history) / total_collaborations
        avg_quality_score = sum(r.quality_score for r in self.collaboration_history) / total_collaborations
        
        # 에이전트별 통계
        agent_stats = defaultdict(lambda: {'collaborations': 0, 'successes': 0, 'avg_quality': 0.0})
        
        for result in self.collaboration_history:
            for agent in result.participating_agents:
                agent_stats[agent]['collaborations'] += 1
                if result.success:
                    agent_stats[agent]['successes'] += 1
                agent_stats[agent]['avg_quality'] += result.quality_score
        
        for agent, stats in agent_stats.items():
            if stats['collaborations'] > 0:
                stats['success_rate'] = stats['successes'] / stats['collaborations']
                stats['avg_quality'] /= stats['collaborations']
        
        return {
            'total_collaborations': total_collaborations,
            'success_rate': successful_collaborations / total_collaborations,
            'avg_execution_time': avg_execution_time,
            'avg_quality_score': avg_quality_score,
            'active_collaborations': len(self.active_collaborations),
            'agent_statistics': dict(agent_stats)
        }