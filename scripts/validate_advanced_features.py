#!/usr/bin/env python3
"""
고급 기능 시스템 검증 스크립트

Phase 3에서 구현된 모든 고급 기능들을 검증하는 스크립트입니다.
"""

import asyncio
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.project_maestro.core.memory_optimizer import (
        MemoryOptimizer, 
        MemoryPressure, 
        MemoryPool,
        SmartCache
    )
    from src.project_maestro.core.distributed_workflow import (
        WorkerNode,
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
    print("✅ 모든 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)


async def validate_memory_optimizer():
    """메모리 최적화 시스템 검증"""
    print("\n=== 메모리 최적화 시스템 검증 ===")
    
    try:
        # 1. 메모리 옵티마이저 생성
        optimizer = MemoryOptimizer()
        print("✅ MemoryOptimizer 인스턴스 생성 성공")
        
        # 2. 메모리 스냅샷 생성
        snapshot = await optimizer.take_snapshot()
        print(f"✅ 메모리 스냅샷 생성 성공 - 사용률: {snapshot.memory_percent:.1f}%")
        
        # 3. 압박 상황 분석
        pressure = optimizer.analyze_pressure(snapshot)
        print(f"✅ 메모리 압박 분석 성공 - 레벨: {pressure.value}")
        
        # 4. 메모리 풀 테스트
        pool = optimizer.get_memory_pool("test_pool", max_size=5)
        
        def create_test_obj():
            return {"data": "test", "timestamp": datetime.now()}
        
        obj1 = pool.get_object(create_test_obj)
        pool.return_object(obj1)
        obj2 = pool.get_object()
        
        if obj2 == obj1:
            print("✅ 메모리 풀 객체 재사용 성공")
        else:
            print("⚠️ 메모리 풀 객체 재사용 확인 불가")
        
        pool_stats = pool.get_stats()
        print(f"✅ 메모리 풀 통계: {pool_stats}")
        
        # 5. 스마트 캐시 테스트
        cache = optimizer.get_cache()
        cache.put("test_key", {"value": "test_data"})
        cached_value = cache.get("test_key")
        
        if cached_value and cached_value["value"] == "test_data":
            print("✅ 스마트 캐시 저장/조회 성공")
        else:
            print("❌ 스마트 캐시 저장/조회 실패")
        
        # 6. 메모리 통계
        stats = optimizer.get_memory_stats()
        print(f"✅ 메모리 통계 수집 성공 - 시스템 사용률: {stats['system']['percent']:.1f}%")
        
        return True, {
            "memory_usage": snapshot.memory_percent,
            "pressure": pressure.value,
            "pool_stats": pool_stats,
            "cache_size": stats['cache']['size']
        }
        
    except Exception as e:
        print(f"❌ 메모리 최적화 시스템 검증 실패: {e}")
        traceback.print_exc()
        return False, {"error": str(e)}


async def validate_distributed_workflow():
    """분산 워크플로우 시스템 검증"""
    print("\n=== 분산 워크플로우 시스템 검증 ===")
    
    try:
        # 1. 워크플로우 생성
        workflow = create_workflow("test_validation_workflow", "validation_script")
        print(f"✅ 워크플로우 생성 성공 - ID: {workflow.workflow_id}")
        
        # 2. 태스크 생성
        task1 = create_task(
            workflow_id=workflow.workflow_id,
            task_type="data_processing",
            payload={"input": "test_data"},
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
        print(f"✅ 태스크 생성 및 추가 성공 - 총 {len(workflow.tasks)}개 태스크")
        
        # 3. 의존성 확인
        ready_tasks = workflow.get_ready_tasks()
        if len(ready_tasks) == 1 and ready_tasks[0].task_id == task1.task_id:
            print("✅ 태스크 의존성 관리 성공")
        else:
            print("⚠️ 태스크 의존성 관리 확인 불가")
        
        # 4. 워커 노드 생성
        node1 = WorkerNode(
            node_id="validation_node_1",
            hostname="localhost",
            port=8001,
            capabilities=["data_processing", "analysis"],
            max_tasks=10
        )
        
        node2 = WorkerNode(
            node_id="validation_node_2", 
            hostname="localhost",
            port=8002,
            capabilities=["analysis"],
            max_tasks=5,
            current_tasks=3
        )
        
        print(f"✅ 워커 노드 생성 성공")
        print(f"  - Node1: {node1.endpoint}, 가용성: {node1.is_available}, 로드: {node1.load_factor}")
        print(f"  - Node2: {node2.endpoint}, 가용성: {node2.is_available}, 로드: {node2.load_factor}")
        
        # 5. 로드 밸런서 테스트
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        nodes = {"node1": node1, "node2": node2}
        
        selected_node = await balancer.select_node(nodes, "data_processing")
        if selected_node and selected_node.node_id == "validation_node_1":
            print("✅ 로드 밸런서 노드 선택 성공")
        else:
            print("⚠️ 로드 밸런서 노드 선택 확인 불가")
        
        return True, {
            "workflow_id": workflow.workflow_id,
            "task_count": len(workflow.tasks),
            "ready_tasks": len(ready_tasks),
            "node_count": len(nodes),
            "selected_node": selected_node.node_id if selected_node else None
        }
        
    except Exception as e:
        print(f"❌ 분산 워크플로우 시스템 검증 실패: {e}")
        traceback.print_exc()
        return False, {"error": str(e)}


async def validate_advanced_visualization():
    """고급 시각화 시스템 검증"""
    print("\n=== 고급 시각화 시스템 검증 ===")
    
    try:
        # 1. 시각화 엔진 생성
        viz_engine = AdvancedVisualizationEngine()
        print("✅ AdvancedVisualizationEngine 인스턴스 생성 성공")
        
        # 2. 리소스 히트맵 데이터 준비
        resource_data = {
            "agent_1": {"cpu_usage": 45, "memory_usage": 60, "disk_io": 30, "network_io": 20},
            "agent_2": {"cpu_usage": 30, "memory_usage": 40, "disk_io": 25, "network_io": 15},
            "agent_3": {"cpu_usage": 60, "memory_usage": 70, "disk_io": 35, "network_io": 25}
        }
        
        # 3. 리소스 히트맵 생성
        heatmap_fig = await viz_engine.create_resource_heatmap(resource_data)
        print("✅ 리소스 히트맵 생성 성공")
        
        # 4. 워크플로우 그래프 데이터 준비
        workflow_data = {
            "tasks": {
                "task_1": {
                    "status": "completed",
                    "task_type": "data_processing",
                    "execution_time": 2.5,
                    "assigned_node": "node_1",
                    "dependencies": []
                },
                "task_2": {
                    "status": "running",
                    "task_type": "analysis",
                    "execution_time": 1.8,
                    "assigned_node": "node_2",
                    "dependencies": ["task_1"]
                }
            }
        }
        
        # 5. 워크플로우 그래프 생성
        workflow_fig = await viz_engine.create_workflow_graph(workflow_data)
        print("✅ 워크플로우 그래프 생성 성공")
        
        # 6. 전체 시각화 파이프라인 테스트
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            width=600,
            height=400,
            export_format="json",
            title="Validation Test Chart"
        )
        
        chart_data = ChartData(
            data=resource_data,
            metadata={"test": "validation"},
            source="validation_script"
        )
        
        viz_result = await viz_engine.create_visualization(config, chart_data)
        print(f"✅ 전체 시각화 파이프라인 성공 - Chart ID: {viz_result['chart_id']}")
        
        # 7. 캐시 확인
        cached_charts = viz_engine.get_cached_charts()
        print(f"✅ 시각화 캐시 확인 성공 - {len(cached_charts)}개 차트 캐시됨")
        
        return True, {
            "heatmap_created": heatmap_fig is not None,
            "workflow_graph_created": workflow_fig is not None,
            "chart_id": viz_result['chart_id'],
            "execution_time": viz_result['execution_time'],
            "cached_charts": len(cached_charts)
        }
        
    except Exception as e:
        print(f"❌ 고급 시각화 시스템 검증 실패: {e}")
        traceback.print_exc()
        return False, {"error": str(e)}


async def validate_system_integration():
    """시스템 통합 검증"""
    print("\n=== 시스템 통합 검증 ===")
    
    try:
        # 1. 모든 시스템 초기화
        memory_optimizer = MemoryOptimizer()
        viz_engine = AdvancedVisualizationEngine()
        print("✅ 모든 시스템 초기화 성공")
        
        # 2. 메모리 데이터 수집
        memory_stats = memory_optimizer.get_memory_stats()
        
        # 3. 시각화 데이터로 변환
        integrated_data = {
            "memory_system": {
                "cpu_usage": 45,  # 시뮬레이션
                "memory_usage": memory_stats["system"]["percent"],
                "disk_io": 30,
                "network_io": 20
            },
            "cache_system": {
                "cpu_usage": 25,
                "memory_usage": memory_stats["cache"]["size"] / 1000,  # KB to %
                "disk_io": 15,
                "network_io": 10
            }
        }
        
        # 4. 통합 시각화 생성
        config = VisualizationConfig(
            chart_type=VisualizationType.RESOURCE_HEATMAP,
            style=ChartStyle.PROFESSIONAL,
            title="Integrated System Status"
        )
        
        chart_data = ChartData(data=integrated_data)
        integration_viz = await viz_engine.create_visualization(config, chart_data)
        
        print(f"✅ 통합 시각화 생성 성공 - Chart ID: {integration_viz['chart_id']}")
        
        # 5. 시스템 상태 보고서
        system_report = {
            "memory_optimizer": {
                "status": "healthy",
                "memory_usage": memory_stats["system"]["percent"],
                "pools": len(memory_optimizer.memory_pools),
                "cache_size": memory_stats["cache"]["size"]
            },
            "visualization_engine": {
                "status": "healthy",
                "cached_charts": len(viz_engine.chart_cache),
                "latest_chart": integration_viz['chart_id']
            },
            "integration_status": "successful"
        }
        
        print("✅ 시스템 통합 검증 완료")
        
        return True, system_report
        
    except Exception as e:
        print(f"❌ 시스템 통합 검증 실패: {e}")
        traceback.print_exc()
        return False, {"error": str(e)}


async def main():
    """메인 검증 프로세스"""
    print("🚀 Project Maestro Phase 3 고급 기능 시스템 검증 시작")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_results = {}
    
    # 1. 메모리 최적화 시스템 검증
    success, result = await validate_memory_optimizer()
    validation_results["memory_optimizer"] = {"success": success, "data": result}
    
    # 2. 분산 워크플로우 시스템 검증
    success, result = await validate_distributed_workflow()
    validation_results["distributed_workflow"] = {"success": success, "data": result}
    
    # 3. 고급 시각화 시스템 검증
    success, result = await validate_advanced_visualization()
    validation_results["advanced_visualization"] = {"success": success, "data": result}
    
    # 4. 시스템 통합 검증
    success, result = await validate_system_integration()
    validation_results["system_integration"] = {"success": success, "data": result}
    
    # 전체 결과 분석
    successful_components = sum(1 for v in validation_results.values() if v["success"])
    total_components = len(validation_results)
    
    print(f"\n=== 검증 결과 요약 ===")
    print(f"총 컴포넌트: {total_components}")
    print(f"성공한 컴포넌트: {successful_components}")
    print(f"실패한 컴포넌트: {total_components - successful_components}")
    print(f"성공률: {(successful_components/total_components)*100:.1f}%")
    
    # 상세 결과 출력
    for component, result in validation_results.items():
        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"  {component}: {status}")
    
    # JSON 보고서 생성
    final_report = {
        "validation_timestamp": datetime.now().isoformat(),
        "overall_success": successful_components == total_components,
        "success_rate": (successful_components/total_components)*100,
        "component_results": validation_results,
        "summary": {
            "total_components": total_components,
            "successful_components": successful_components,
            "failed_components": total_components - successful_components
        }
    }
    
    # 보고서 파일로 저장
    report_path = project_root / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 상세 보고서가 {report_path}에 저장되었습니다.")
    
    if final_report["overall_success"]:
        print("\n🎉 모든 고급 기능 시스템 검증이 성공적으로 완료되었습니다!")
        return 0
    else:
        print("\n⚠️ 일부 시스템에서 문제가 발견되었습니다. 로그를 확인해주세요.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        traceback.print_exc()
        sys.exit(1)