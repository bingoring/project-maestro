"""
고급 기능 API 엔드포인트

Phase 3에서 구현된 고급 기능들을 위한 API 엔드포인트입니다.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime

from ...core.memory_optimizer import get_memory_optimizer, MemoryPressure, OptimizationStrategy
from ...core.distributed_workflow import (
    DistributedWorkflowManager, 
    WorkerNode, 
    DistributedWorkflow, 
    WorkflowTask,
    create_workflow,
    create_task
)
from ...core.advanced_visualization import (
    get_visualization_engine, 
    VisualizationConfig, 
    ChartData, 
    VisualizationType,
    ChartStyle
)

router = APIRouter(prefix="/advanced", tags=["advanced-features"])

# 전역 인스턴스들
memory_optimizer = get_memory_optimizer()
workflow_manager = DistributedWorkflowManager()
viz_engine = get_visualization_engine()


# Memory Optimization Endpoints
@router.get("/memory/stats")
async def get_memory_stats():
    """메모리 통계 조회"""
    try:
        stats = memory_optimizer.get_memory_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/optimize")
async def optimize_memory(
    pressure: Optional[str] = None,
    strategy: Optional[str] = None
):
    """메모리 최적화 수행"""
    try:
        # 현재 스냅샷 생성
        snapshot = await memory_optimizer.take_snapshot()
        
        # 압박 상황 분석 또는 사용자 지정
        if pressure:
            memory_pressure = MemoryPressure(pressure)
        else:
            memory_pressure = memory_optimizer.analyze_pressure(snapshot)
        
        # 최적화 수행
        result = await memory_optimizer.optimize_memory(memory_pressure)
        
        return {
            "status": "success",
            "data": {
                "pressure": memory_pressure.value,
                "strategy": result.strategy.value,
                "memory_freed": result.memory_freed,
                "objects_collected": result.objects_collected,
                "execution_time": result.execution_time,
                "success": result.success
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/monitoring/start")
async def start_memory_monitoring():
    """메모리 모니터링 시작"""
    try:
        await memory_optimizer.start_monitoring()
        return {"status": "success", "message": "메모리 모니터링이 시작되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/monitoring/stop")
async def stop_memory_monitoring():
    """메모리 모니터링 중지"""
    try:
        await memory_optimizer.stop_monitoring()
        return {"status": "success", "message": "메모리 모니터링이 중지되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Distributed Workflow Endpoints
@router.post("/workflow/start")
async def start_workflow_manager():
    """워크플로우 매니저 시작"""
    try:
        await workflow_manager.start()
        return {"status": "success", "message": "워크플로우 매니저가 시작되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/stop")
async def stop_workflow_manager():
    """워크플로우 매니저 중지"""
    try:
        await workflow_manager.stop()
        return {"status": "success", "message": "워크플로우 매니저가 중지되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/node/register")
async def register_worker_node(node_data: Dict[str, Any]):
    """워커 노드 등록"""
    try:
        node = WorkerNode(
            node_id=node_data["node_id"],
            hostname=node_data["hostname"],
            port=node_data["port"],
            capabilities=node_data.get("capabilities", []),
            max_tasks=node_data.get("max_tasks", 10)
        )
        
        success = await workflow_manager.register_node(node)
        
        if success:
            return {"status": "success", "message": f"노드 {node.node_id}가 등록되었습니다"}
        else:
            raise HTTPException(status_code=400, detail="노드 등록에 실패했습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflow/node/{node_id}")
async def unregister_worker_node(node_id: str):
    """워커 노드 등록 해제"""
    try:
        success = await workflow_manager.unregister_node(node_id)
        
        if success:
            return {"status": "success", "message": f"노드 {node_id}가 등록 해제되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="노드를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/submit")
async def submit_workflow(workflow_data: Dict[str, Any]):
    """워크플로우 제출"""
    try:
        # 워크플로우 생성
        workflow = create_workflow(
            name=workflow_data["name"],
            created_by=workflow_data.get("created_by")
        )
        
        # 태스크 추가
        for task_data in workflow_data.get("tasks", []):
            task = create_task(
                workflow_id=workflow.workflow_id,
                task_type=task_data["task_type"],
                payload=task_data["payload"],
                dependencies=task_data.get("dependencies", []),
                priority=task_data.get("priority", 0),
                timeout=task_data.get("timeout", 3600)
            )
            workflow.add_task(task)
        
        # 워크플로우 제출
        workflow_id = await workflow_manager.submit_workflow(workflow)
        
        return {
            "status": "success",
            "data": {
                "workflow_id": workflow_id,
                "task_count": len(workflow.tasks)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """워크플로우 상태 조회"""
    try:
        status = await workflow_manager.get_workflow_status(workflow_id)
        
        if status:
            return {"status": "success", "data": status}
        else:
            raise HTTPException(status_code=404, detail="워크플로우를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """워크플로우 취소"""
    try:
        success = await workflow_manager.cancel_workflow(workflow_id)
        
        if success:
            return {"status": "success", "message": f"워크플로우 {workflow_id}가 취소되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="워크플로우를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Visualization Endpoints
@router.post("/visualization/create")
async def create_visualization(viz_request: Dict[str, Any]):
    """시각화 생성"""
    try:
        # 설정 생성
        config = VisualizationConfig(
            chart_type=VisualizationType(viz_request["chart_type"]),
            style=ChartStyle(viz_request.get("style", "professional")),
            width=viz_request.get("width", 800),
            height=viz_request.get("height", 600),
            interactive=viz_request.get("interactive", True),
            export_format=viz_request.get("export_format", "html"),
            title=viz_request.get("title")
        )
        
        # 데이터 생성
        chart_data = ChartData(
            data=viz_request["data"],
            metadata=viz_request.get("metadata", {}),
            source=viz_request.get("source")
        )
        
        # 시각화 생성
        result = await viz_engine.create_visualization(config, chart_data)
        
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/charts")
async def get_cached_charts():
    """캐시된 차트 목록 조회"""
    try:
        charts = viz_engine.get_cached_charts()
        return {"status": "success", "data": charts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/visualization/cache")
async def clear_visualization_cache():
    """시각화 캐시 정리"""
    try:
        await viz_engine.clear_cache()
        return {"status": "success", "message": "시각화 캐시가 정리되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """실시간 대시보드 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Project Maestro - Advanced Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .chart-container { 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header { text-align: center; margin-bottom: 30px; }
            .metrics-bar {
                display: flex;
                justify-content: space-around;
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric {
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #2196F3;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Project Maestro - Advanced Features Dashboard</h1>
            <p>Memory Optimization | Distributed Workflows | Advanced Visualization</p>
        </div>
        
        <div class="metrics-bar">
            <div class="metric">
                <div class="metric-value" id="memory-usage">--</div>
                <div class="metric-label">Memory Usage (%)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="active-workflows">--</div>
                <div class="metric-label">Active Workflows</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="worker-nodes">--</div>
                <div class="metric-label">Worker Nodes</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="visualizations">--</div>
                <div class="metric-label">Visualizations</div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <div class="chart-container">
                <h3>Memory Usage Timeline</h3>
                <div id="memory-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Workflow Status</h3>
                <div id="workflow-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>System Performance</h3>
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Resource Utilization</h3>
                <div id="resource-chart"></div>
            </div>
        </div>
        
        <script>
            // 실시간 데이터 업데이트를 위한 WebSocket 연결
            const ws = new WebSocket(`ws://${window.location.host}/advanced/ws/dashboard`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            function updateDashboard(data) {
                // 메트릭 업데이트
                document.getElementById('memory-usage').textContent = data.memory_usage + '%';
                document.getElementById('active-workflows').textContent = data.active_workflows;
                document.getElementById('worker-nodes').textContent = data.worker_nodes;
                document.getElementById('visualizations').textContent = data.visualizations;
                
                // 차트 업데이트
                updateCharts(data);
            }
            
            function updateCharts(data) {
                // 메모리 사용량 차트
                const memoryTrace = {
                    x: data.timestamps,
                    y: data.memory_timeline,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Memory Usage',
                    line: { color: '#2196F3' }
                };
                
                Plotly.newPlot('memory-chart', [memoryTrace], {
                    title: 'Memory Usage Over Time',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Memory (%)' }
                });
                
                // 워크플로우 상태 차트
                const workflowTrace = {
                    x: ['Pending', 'Running', 'Completed', 'Failed'],
                    y: [data.pending_workflows, data.running_workflows, data.completed_workflows, data.failed_workflows],
                    type: 'bar',
                    marker: { color: ['#FFA726', '#42A5F5', '#66BB6A', '#EF5350'] }
                };
                
                Plotly.newPlot('workflow-chart', [workflowTrace], {
                    title: 'Workflow Status Distribution'
                });
            }
            
            // 초기 데이터 로드
            fetch('/advanced/dashboard/data')
                .then(response => response.json())
                .then(data => updateDashboard(data.data))
                .catch(console.error);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """대시보드 실시간 데이터 WebSocket"""
    await websocket.accept()
    
    try:
        while True:
            # 실시간 데이터 수집
            dashboard_data = {
                "memory_usage": memory_optimizer.get_memory_stats()["system"]["percent"],
                "active_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "running"]),
                "worker_nodes": len(workflow_manager.nodes),
                "visualizations": len(viz_engine.chart_cache),
                "timestamps": [datetime.now().isoformat()],
                "memory_timeline": [memory_optimizer.get_memory_stats()["system"]["percent"]],
                "pending_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "pending"]),
                "running_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "running"]),
                "completed_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "completed"]),
                "failed_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "failed"])
            }
            
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(5)  # 5초마다 업데이트
            
    except WebSocketDisconnect:
        pass


@router.get("/dashboard/data")
async def get_dashboard_data():
    """대시보드 데이터 조회"""
    try:
        memory_stats = memory_optimizer.get_memory_stats()
        
        data = {
            "memory_usage": memory_stats["system"]["percent"],
            "active_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "running"]),
            "worker_nodes": len(workflow_manager.nodes),
            "visualizations": len(viz_engine.chart_cache),
            "timestamps": [datetime.now().isoformat()],
            "memory_timeline": [memory_stats["system"]["percent"]],
            "pending_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "pending"]),
            "running_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "running"]),
            "completed_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "completed"]),
            "failed_workflows": len([w for w in workflow_manager.workflows.values() if w.status.value == "failed"])
        }
        
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Integration Endpoints
@router.get("/health")
async def advanced_health_check():
    """고급 기능 시스템 상태 확인"""
    try:
        memory_stats = memory_optimizer.get_memory_stats()
        
        health_data = {
            "memory_optimizer": {
                "status": "active",
                "memory_usage": memory_stats["system"]["percent"],
                "cache_size": memory_stats["cache"]["size"]
            },
            "workflow_manager": {
                "status": "active" if workflow_manager.scheduler_task else "inactive",
                "active_workflows": len(workflow_manager.workflows),
                "worker_nodes": len(workflow_manager.nodes)
            },
            "visualization_engine": {
                "status": "active",
                "cached_charts": len(viz_engine.chart_cache)
            },
            "overall_status": "healthy"
        }
        
        return {"status": "success", "data": health_data}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/integration/test")
async def run_integration_test():
    """통합 테스트 실행"""
    try:
        test_results = []
        
        # 1. 메모리 최적화 테스트
        try:
            snapshot = await memory_optimizer.take_snapshot()
            pressure = memory_optimizer.analyze_pressure(snapshot)
            test_results.append({
                "component": "memory_optimizer",
                "test": "snapshot_and_analysis",
                "status": "pass",
                "details": f"Memory pressure: {pressure.value}"
            })
        except Exception as e:
            test_results.append({
                "component": "memory_optimizer",
                "test": "snapshot_and_analysis",
                "status": "fail",
                "error": str(e)
            })
        
        # 2. 시각화 엔진 테스트
        try:
            test_config = VisualizationConfig(
                chart_type=VisualizationType.RESOURCE_HEATMAP,
                style=ChartStyle.PROFESSIONAL
            )
            test_data = ChartData(data={
                "agent1": {"cpu_usage": 45, "memory_usage": 60},
                "agent2": {"cpu_usage": 30, "memory_usage": 40}
            })
            
            viz_result = await viz_engine.create_visualization(test_config, test_data)
            test_results.append({
                "component": "visualization_engine",
                "test": "chart_creation",
                "status": "pass",
                "details": f"Chart created with ID: {viz_result['chart_id']}"
            })
        except Exception as e:
            test_results.append({
                "component": "visualization_engine",
                "test": "chart_creation",
                "status": "fail",
                "error": str(e)
            })
        
        # 3. 워크플로우 매니저 테스트 (기본 기능만)
        try:
            # 워커 노드 생성 테스트
            test_node = WorkerNode(
                node_id="test_node_1",
                hostname="localhost",
                port=8001,
                capabilities=["test_task"]
            )
            
            test_results.append({
                "component": "workflow_manager",
                "test": "node_creation",
                "status": "pass",
                "details": f"Test node created: {test_node.node_id}"
            })
        except Exception as e:
            test_results.append({
                "component": "workflow_manager",
                "test": "node_creation",
                "status": "fail",
                "error": str(e)
            })
        
        # 전체 결과 분석
        passed_tests = len([r for r in test_results if r["status"] == "pass"])
        total_tests = len(test_results)
        
        return {
            "status": "success",
            "data": {
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": total_tests - passed_tests,
                    "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
                },
                "detailed_results": test_results,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))