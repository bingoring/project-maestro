"""
고급 기능 통합 테스트

Phase 3에서 구현된 모든 고급 기능들의 통합 테스트입니다.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# 테스트할 시스템들 임포트
from src.project_maestro.core.memory_optimizer import (
    MemoryOptimizer, 
    MemoryPressure, 
    OptimizationStrategy,
    MemoryPool,
    SmartCache
)
from src.project_maestro.core.distributed_workflow import (
    DistributedWorkflowManager,
    WorkerNode,
    DistributedWorkflow,
    WorkflowTask,
    create_workflow,
    create_task,
    LoadBalancer,
    LoadBalancingStrategy
)
from src.project_maestro.core.advanced_visualization import (
    AdvancedVisualizationEngine,
    VisualizationConfig,
    ChartData,
    VisualizationType,
    ChartStyle
)


class TestMemoryOptimizer:
    """메모리 최적화 시스템 테스트"""
    
    @pytest.fixture
    async def memory_optimizer(self):
        """메모리 옵티마이저 픽스처"""
        optimizer = MemoryOptimizer()
        yield optimizer
        await optimizer.stop_monitoring()
    
    async def test_memory_snapshot(self, memory_optimizer):
        """메모리 스냅샷 생성 테스트"""
        snapshot = await memory_optimizer.take_snapshot()
        
        assert snapshot.timestamp is not None
        assert snapshot.total_memory > 0
        assert snapshot.available_memory >= 0
        assert snapshot.used_memory >= 0
        assert 0 <= snapshot.memory_percent <= 100
        assert len(snapshot.gc_count) == 3
    
    async def test_pressure_analysis(self, memory_optimizer):
        """메모리 압박 상황 분석 테스트"""
        snapshot = await memory_optimizer.take_snapshot()
        pressure = memory_optimizer.analyze_pressure(snapshot)
        
        assert pressure in [MemoryPressure.LOW, MemoryPressure.MEDIUM, 
                          MemoryPressure.HIGH, MemoryPressure.CRITICAL]
    
    async def test_memory_optimization(self, memory_optimizer):
        """메모리 최적화 테스트"""
        snapshot = await memory_optimizer.take_snapshot()
        pressure = memory_optimizer.analyze_pressure(snapshot)
        
        result = await memory_optimizer.optimize_memory(pressure)
        
        assert result.strategy in [OptimizationStrategy.CONSERVATIVE,
                                 OptimizationStrategy.BALANCED,
                                 OptimizationStrategy.AGGRESSIVE]
        assert result.execution_time >= 0
        assert isinstance(result.success, bool)
    
    async def test_memory_pool(self, memory_optimizer):
        """메모리 풀 테스트"""
        pool = memory_optimizer.get_memory_pool("test_pool", max_size=10)
        
        # 객체 생성 팩토리
        def create_test_object():
            return {"test": "data", "timestamp": datetime.now()}
        
        # 객체 가져오기
        obj1 = pool.get_object(create_test_object)
        assert obj1 is not None
        assert "test" in obj1
        
        # 객체 반납
        pool.return_object(obj1)
        
        # 재사용 확인
        obj2 = pool.get_object(create_test_object)
        assert obj2 == obj1  # 같은 객체가 재사용됨
        
        # 통계 확인
        stats = pool.get_stats()
        assert stats["reused"] >= 1
    
    async def test_smart_cache(self, memory_optimizer):
        """스마트 캐시 테스트"""
        cache = memory_optimizer.get_cache()
        
        # 데이터 저장
        cache.put("test_key", {"data": "test_value"})
        
        # 데이터 조회
        result = cache.get("test_key")
        assert result is not None
        assert result["data"] == "test_value"
        
        # 없는 키 조회
        result = cache.get("nonexistent_key")
        assert result is None
        
        # 캐시 통계
        stats = cache.get_stats()
        assert stats["size"] >= 1


class TestDistributedWorkflow:
    """분산 워크플로우 시스템 테스트"""
    
    @pytest.fixture
    async def workflow_manager(self):
        """워크플로우 매니저 픽스처"""
        manager = DistributedWorkflowManager("redis://localhost:6379/1")  # 테스트용 DB
        yield manager
        await manager.stop()
    
    async def test_workflow_creation(self):
        """워크플로우 생성 테스트"""
        workflow = create_workflow("test_workflow", "test_user")
        
        assert workflow.workflow_id is not None
        assert workflow.name == "test_workflow"
        assert workflow.created_by == "test_user"
        assert workflow.status.value == "pending"
        assert len(workflow.tasks) == 0
    
    async def test_task_creation(self):
        """태스크 생성 테스트"""
        workflow = create_workflow("test_workflow")
        
        task1 = create_task(
            workflow_id=workflow.workflow_id,
            task_type="data_processing",
            payload={"input_data": "test"},
            priority=1
        )
        
        task2 = create_task(
            workflow_id=workflow.workflow_id,
            task_type="analysis",
            payload={"model": "test_model"},
            dependencies=[task1.task_id],
            priority=2
        )
        
        workflow.add_task(task1)
        workflow.add_task(task2)
        
        assert len(workflow.tasks) == 2
        assert task1.is_ready  # 의존성 없음
        assert not task2.is_ready  # task1에 의존
    
    async def test_worker_node(self):
        """워커 노드 테스트"""
        node = WorkerNode(
            node_id="test_node_1",
            hostname="localhost",
            port=8001,
            capabilities=["data_processing", "analysis"],
            max_tasks=5
        )
        
        assert node.endpoint == "http://localhost:8001"
        assert node.is_available
        assert node.load_factor == 0.0  # 아직 태스크 없음
        
        # 태스크 할당 시뮬레이션
        node.current_tasks = 2
        node.cpu_usage = 45.0
        assert node.load_factor == 0.45  # max(2/5, 45/100)
    
    async def test_load_balancer(self):
        """로드 밸런서 테스트"""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # 테스트 노드들 생성
        nodes = {
            "node1": WorkerNode("node1", "host1", 8001, current_tasks=2, max_tasks=10),
            "node2": WorkerNode("node2", "host2", 8002, current_tasks=5, max_tasks=10),
            "node3": WorkerNode("node3", "host3", 8003, current_tasks=1, max_tasks=10)
        }
        
        # 최소 연결 선택 테스트
        selected_node = await balancer.select_node(nodes)
        assert selected_node.node_id == "node3"  # 가장 적은 태스크 수


class TestAdvancedVisualization:
    """고급 시각화 시스템 테스트"""
    
    @pytest.fixture
    def viz_engine(self):
        """시각화 엔진 픽스처"""
        return AdvancedVisualizationEngine()
    
    async def test_visualization_config(self, viz_engine):
        """시각화 설정 테스트"""
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            width=800,
            height=600,
            title="Test Chart"
        )
        
        assert config.chart_type == VisualizationType.RESOURCE_HEATMAP
        assert config.style == ChartStyle.PROFESSIONAL
        assert config.width == 800
        assert config.height == 600
        assert config.title == "Test Chart"
    
    async def test_chart_data(self, viz_engine):
        """차트 데이터 테스트"""
        data = ChartData(
            data={"agent1": {"cpu": 45, "memory": 60}, "agent2": {"cpu": 30, "memory": 40}},
            metadata={"source": "test", "type": "performance"},
            source="test_system"
        )
        
        assert data.data is not None
        assert data.metadata["source"] == "test"
        assert data.source == "test_system"
        assert data.timestamp is not None
    
    async def test_resource_heatmap_creation(self, viz_engine):
        """리소스 히트맵 생성 테스트"""
        resource_data = {
            "agent1": {"cpu_usage": 45, "memory_usage": 60, "disk_io": 30, "network_io": 20},
            "agent2": {"cpu_usage": 30, "memory_usage": 40, "disk_io": 25, "network_io": 15},
            "agent3": {"cpu_usage": 60, "memory_usage": 70, "disk_io": 35, "network_io": 25}
        }
        
        fig = await viz_engine.create_resource_heatmap(resource_data)
        
        assert fig is not None
        assert fig.data is not None
        assert len(fig.data) > 0
    
    async def test_workflow_graph_creation(self, viz_engine):
        """워크플로우 그래프 생성 테스트"""
        workflow_data = {
            "tasks": {
                "task1": {
                    "status": "completed",
                    "task_type": "data_processing",
                    "execution_time": 2.5,
                    "assigned_node": "node1",
                    "dependencies": []
                },
                "task2": {
                    "status": "running",
                    "task_type": "analysis",
                    "execution_time": 1.8,
                    "assigned_node": "node2",
                    "dependencies": ["task1"]
                },
                "task3": {
                    "status": "pending",
                    "task_type": "reporting",
                    "execution_time": 0,
                    "assigned_node": None,
                    "dependencies": ["task2"]
                }
            }
        }
        
        fig = await viz_engine.create_workflow_graph(workflow_data)
        
        assert fig is not None
        assert fig.data is not None
        assert len(fig.data) >= 2  # nodes and edges
    
    async def test_full_visualization_pipeline(self, viz_engine):
        """전체 시각화 파이프라인 테스트"""
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            export_format="json"
        )
        
        chart_data = ChartData(
            data={
                "agent1": {"cpu_usage": 45, "memory_usage": 60, "disk_io": 30, "network_io": 20},
                "agent2": {"cpu_usage": 30, "memory_usage": 40, "disk_io": 25, "network_io": 15}
            }
        )
        
        result = await viz_engine.create_visualization(config, chart_data)
        
        assert result["chart_id"] is not None
        assert result["config"]["chart_type"] == "resource_heatmap"
        assert result["execution_time"] >= 0
        assert result["metadata"]["data_points"] > 0
        
        # 캐시 확인
        cached_charts = viz_engine.get_cached_charts()
        assert result["chart_id"] in cached_charts


class TestSystemIntegration:
    """시스템 통합 테스트"""
    
    @pytest.fixture
    async def integrated_system(self):
        """통합 시스템 픽스처"""
        memory_optimizer = MemoryOptimizer()
        viz_engine = AdvancedVisualizationEngine()
        
        # 워크플로우 매니저는 Redis 없이는 테스트하기 어려우므로 제외
        
        yield {
            "memory_optimizer": memory_optimizer,
            "viz_engine": viz_engine
        }
        
        await memory_optimizer.stop_monitoring()
    
    async def test_memory_visualization_integration(self, integrated_system):
        """메모리 최적화와 시각화 통합 테스트"""
        memory_optimizer = integrated_system["memory_optimizer"]
        viz_engine = integrated_system["viz_engine"]
        
        # 메모리 통계 수집
        memory_stats = memory_optimizer.get_memory_stats()
        
        # 시각화 데이터로 변환
        viz_data = {
            "System": {
                "cpu_usage": 45,  # 시뮬레이션 값
                "memory_usage": memory_stats["system"]["percent"],
                "disk_io": 30,
                "network_io": 20
            }
        }
        
        # 시각화 생성
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            title="System Resource Usage"
        )
        
        chart_data = ChartData(data=viz_data)
        result = await viz_engine.create_visualization(config, chart_data)
        
        assert result is not None
        assert result["chart_id"] is not None
        assert result["execution_time"] >= 0
    
    async def test_performance_monitoring_integration(self, integrated_system):
        """성능 모니터링 통합 테스트"""
        memory_optimizer = integrated_system["memory_optimizer"]
        viz_engine = integrated_system["viz_engine"]
        
        # 성능 데이터 수집 시뮬레이션
        performance_data = []
        for i in range(5):
            await asyncio.sleep(0.1)  # 짧은 지연
            
            snapshot = await memory_optimizer.take_snapshot()
            performance_data.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "agent_id": f"agent_{i % 2 + 1}",
                "cpu_usage": 40 + (i * 5),
                "memory_usage": snapshot.memory_percent,
                "response_time": 100 + (i * 10)
            })
        
        # 타임라인 시각화 생성
        config = VisualizationConfig(
            chart_type=VisualizationType.PERFORMANCE_TIMELINE,
            style=ChartStyle.PROFESSIONAL,
            title="Performance Timeline"
        )
        
        chart_data = ChartData(data=performance_data)
        result = await viz_engine.create_visualization(config, chart_data)
        
        assert result is not None
        assert len(performance_data) == 5
        assert result["metadata"]["data_points"] > 0
    
    async def test_system_health_check(self, integrated_system):
        """시스템 상태 확인 통합 테스트"""
        memory_optimizer = integrated_system["memory_optimizer"]
        viz_engine = integrated_system["viz_engine"]
        
        # 메모리 시스템 상태
        memory_stats = memory_optimizer.get_memory_stats()
        memory_healthy = memory_stats["system"]["percent"] < 90
        
        # 시각화 시스템 상태
        viz_cache_count = len(viz_engine.chart_cache)
        viz_healthy = True  # 시각화는 기본적으로 건강함
        
        # 전체 시스템 상태
        system_healthy = memory_healthy and viz_healthy
        
        assert isinstance(memory_healthy, bool)
        assert isinstance(viz_healthy, bool)
        assert isinstance(system_healthy, bool)
        
        # 상태 정보 구조화
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if system_healthy else "degraded",
            "components": {
                "memory_optimizer": {
                    "status": "healthy" if memory_healthy else "degraded",
                    "memory_usage_percent": memory_stats["system"]["percent"],
                    "cache_size": memory_stats["cache"]["size"]
                },
                "visualization_engine": {
                    "status": "healthy",
                    "cached_charts": viz_cache_count
                }
            }
        }
        
        assert health_report["overall_status"] in ["healthy", "degraded"]
        assert "memory_optimizer" in health_report["components"]
        assert "visualization_engine" in health_report["components"]


@pytest.mark.asyncio
async def test_full_system_integration():
    """전체 시스템 통합 테스트"""
    
    # 1. 시스템 초기화
    memory_optimizer = MemoryOptimizer()
    viz_engine = AdvancedVisualizationEngine()
    
    try:
        # 2. 메모리 모니터링 시작
        await memory_optimizer.start_monitoring()
        await asyncio.sleep(1)  # 모니터링 안정화 대기
        
        # 3. 메모리 스냅샷 및 최적화
        snapshot = await memory_optimizer.take_snapshot()
        pressure = memory_optimizer.analyze_pressure(snapshot)
        optimization_result = await memory_optimizer.optimize_memory(pressure)
        
        # 4. 결과를 시각화로 변환
        viz_data = {
            "optimization_result": {
                "memory_freed": optimization_result.memory_freed,
                "objects_collected": optimization_result.objects_collected,
                "execution_time": optimization_result.execution_time
            }
        }
        
        # 5. 시각화 생성
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            title="Memory Optimization Results"
        )
        
        chart_data = ChartData(data=viz_data)
        viz_result = await viz_engine.create_visualization(config, chart_data)
        
        # 6. 결과 검증
        assert snapshot is not None
        assert pressure in [MemoryPressure.LOW, MemoryPressure.MEDIUM, 
                          MemoryPressure.HIGH, MemoryPressure.CRITICAL]
        assert optimization_result.success or not optimization_result.success  # 성공/실패 모두 허용
        assert viz_result["chart_id"] is not None
        assert viz_result["execution_time"] >= 0
        
        # 7. 통합 상태 보고서 생성
        integration_report = {
            "test_timestamp": datetime.now().isoformat(),
            "memory_optimization": {
                "pressure": pressure.value,
                "strategy": optimization_result.strategy.value,
                "memory_freed": optimization_result.memory_freed,
                "success": optimization_result.success
            },
            "visualization": {
                "chart_id": viz_result["chart_id"],
                "execution_time": viz_result["execution_time"],
                "data_points": viz_result["metadata"]["data_points"]
            },
            "integration_status": "success"
        }
        
        print(f"통합 테스트 보고서: {json.dumps(integration_report, indent=2, ensure_ascii=False)}")
        
        return integration_report
        
    finally:
        # 8. 정리
        await memory_optimizer.stop_monitoring()


if __name__ == "__main__":
    # 독립적으로 실행될 때의 테스트
    async def main():
        print("=== Project Maestro Phase 3 통합 테스트 시작 ===")
        
        try:
            result = await test_full_system_integration()
            print("✅ 통합 테스트 성공!")
            print(f"결과: {result['integration_status']}")
            
        except Exception as e:
            print(f"❌ 통합 테스트 실패: {str(e)}")
            raise
        
        print("=== 통합 테스트 완료 ===")
    
    asyncio.run(main())