"""
고급 시각화 시스템

AI 에이전트 워크플로우와 성능 데이터를 다양한 방식으로 시각화하는 시스템입니다.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
from pathlib import Path

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.colors import qualitative
import networkx as nx

from ..utils.metrics import PrometheusMetrics


class VisualizationType(Enum):
    """시각화 타입"""
    WORKFLOW_GRAPH = "workflow_graph"
    PERFORMANCE_TIMELINE = "performance_timeline"
    RESOURCE_HEATMAP = "resource_heatmap"
    AGENT_NETWORK = "agent_network"
    EXECUTION_GANTT = "execution_gantt"
    DEPENDENCY_TREE = "dependency_tree"
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    COMPARISON_CHART = "comparison_chart"
    DISTRIBUTION_PLOT = "distribution_plot"
    CORRELATION_MATRIX = "correlation_matrix"


class ChartStyle(Enum):
    """차트 스타일"""
    MINIMAL = "minimal"
    PROFESSIONAL = "professional"
    DARK = "dark"
    COLORFUL = "colorful"
    SCIENTIFIC = "scientific"


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    chart_type: VisualizationType
    style: ChartStyle = ChartStyle.PROFESSIONAL
    width: int = 800
    height: int = 600
    interactive: bool = True
    export_format: str = "html"  # html, png, svg, pdf
    color_scheme: str = "viridis"
    animation: bool = False
    show_legend: bool = True
    title: Optional[str] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartData:
    """차트 데이터"""
    data: Union[Dict, List, pd.DataFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None


class AdvancedVisualizationEngine:
    """고급 시각화 엔진"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.metrics = PrometheusMetrics()
        
        # 차트 캐시
        self.chart_cache: Dict[str, Any] = {}
        
        # 스타일 설정
        self.styles = {
            ChartStyle.MINIMAL: {
                'template': 'simple_white',
                'color_palette': px.colors.qualitative.Pastel,
                'font_family': 'Arial',
                'grid_color': '#E5E5E5'
            },
            ChartStyle.PROFESSIONAL: {
                'template': 'plotly_white',
                'color_palette': px.colors.qualitative.Set3,
                'font_family': 'Segoe UI',
                'grid_color': '#F0F0F0'
            },
            ChartStyle.DARK: {
                'template': 'plotly_dark',
                'color_palette': px.colors.qualitative.Dark2,
                'font_family': 'Roboto',
                'grid_color': '#3E3E3E'
            },
            ChartStyle.COLORFUL: {
                'template': 'plotly_white',
                'color_palette': px.colors.qualitative.Vivid,
                'font_family': 'Comic Sans MS',
                'grid_color': '#EAEAEA'
            },
            ChartStyle.SCIENTIFIC: {
                'template': 'plotly_white',
                'color_palette': px.colors.sequential.Viridis,
                'font_family': 'Times New Roman',
                'grid_color': '#D3D3D3'
            }
        }
        
        # 실시간 업데이트를 위한 WebSocket 관리
        self.websocket_connections: Dict[str, Any] = {}
        
    async def create_visualization(
        self,
        config: VisualizationConfig,
        data: ChartData
    ) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            start_time = datetime.now()
            
            # 데이터 전처리
            processed_data = await self._preprocess_data(data, config)
            
            # 차트 생성
            chart = await self._create_chart(config, processed_data)
            
            # 스타일 적용
            styled_chart = await self._apply_style(chart, config)
            
            # 내보내기
            output = await self._export_chart(styled_chart, config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'chart_id': f"{config.chart_type.value}_{data.timestamp.timestamp()}",
                'config': asdict(config),
                'output': output,
                'execution_time': execution_time,
                'metadata': {
                    'data_points': self._count_data_points(processed_data),
                    'chart_size': f"{config.width}x{config.height}",
                    'created_at': datetime.now().isoformat()
                }
            }
            
            # 캐시에 저장
            self.chart_cache[result['chart_id']] = result
            
            self.logger.info(
                "시각화 생성 완료",
                chart_type=config.chart_type.value,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error("시각화 생성 실패", error=str(e))
            raise
    
    async def create_workflow_graph(self, workflow_data: Dict[str, Any]) -> go.Figure:
        """워크플로우 그래프 생성"""
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        # 노드 및 엣지 추가
        for task_id, task_info in workflow_data.get('tasks', {}).items():
            G.add_node(task_id, **task_info)
            
            for dependency in task_info.get('dependencies', []):
                G.add_edge(dependency, task_id)
        
        # 레이아웃 계산
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 엣지 그리기
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 그리기
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_info = [G.nodes[node] for node in G.nodes()]
        
        node_colors = []
        node_sizes = []
        hover_texts = []
        
        for i, (node_id, info) in enumerate(zip(G.nodes(), node_info)):
            status = info.get('status', 'pending')
            
            # 상태별 색상
            color_map = {
                'pending': '#FFA500',
                'running': '#1E90FF',
                'completed': '#32CD32',
                'failed': '#FF4500',
                'cancelled': '#808080'
            }
            node_colors.append(color_map.get(status, '#808080'))
            
            # 실행 시간에 따른 크기
            exec_time = info.get('execution_time', 1)
            node_sizes.append(max(20, min(60, exec_time * 10)))
            
            # 호버 텍스트
            hover_text = f"""
            Task: {node_id}<br>
            Status: {status}<br>
            Type: {info.get('task_type', 'N/A')}<br>
            Execution Time: {exec_time:.2f}s<br>
            Node: {info.get('assigned_node', 'N/A')}
            """
            hover_texts.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            text=[node[:8] + '...' if len(node) > 8 else node for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # 그래프 구성
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Workflow Execution Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size reflects execution time",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='#888', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    async def create_performance_timeline(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """성능 타임라인 생성"""
        df = pd.DataFrame(performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Response Time (ms)')
        )
        
        # CPU 사용률
        for agent in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent]
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['cpu_usage'],
                    mode='lines+markers',
                    name=f'{agent} CPU',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 메모리 사용률
        for agent in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent]
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['memory_usage'],
                    mode='lines+markers',
                    name=f'{agent} Memory',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 응답 시간
        for agent in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent]
            fig.add_trace(
                go.Scatter(
                    x=agent_data['timestamp'],
                    y=agent_data['response_time'],
                    mode='lines+markers',
                    name=f'{agent} Response',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='Agent Performance Timeline',
            height=800,
            hovermode='x unified'
        )
        
        return fig
    
    async def create_resource_heatmap(self, resource_data: Dict[str, Any]) -> go.Figure:
        """리소스 히트맵 생성"""
        # 데이터를 행렬로 변환
        agents = list(resource_data.keys())
        metrics = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']
        
        z_data = []
        for agent in agents:
            row = []
            for metric in metrics:
                value = resource_data[agent].get(metric, 0)
                row.append(value)
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=metrics,
            y=agents,
            colorscale='RdYlBu_r',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Agent: %{y}<br>Metric: %{x}<br>Value: %{z:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Resource Usage Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Agents',
            width=600,
            height=400
        )
        
        return fig
    
    async def create_execution_gantt(self, execution_data: List[Dict[str, Any]]) -> go.Figure:
        """실행 간트 차트 생성"""
        df = pd.DataFrame(execution_data)
        df['Start'] = pd.to_datetime(df['start_time'])
        df['Finish'] = pd.to_datetime(df['end_time'])
        df['Duration'] = (df['Finish'] - df['Start']).dt.total_seconds()
        
        # 상태별 색상 매핑
        colors = {
            'completed': 'green',
            'running': 'blue',
            'failed': 'red',
            'pending': 'orange',
            'cancelled': 'gray'
        }
        
        fig = ff.create_gantt(
            df,
            colors=colors,
            index_col='status',
            title='Task Execution Timeline',
            show_colorbar=True,
            bar_width=0.3,
            showgrid_x=True,
            showgrid_y=True
        )
        
        return fig
    
    async def create_agent_network(self, collaboration_data: Dict[str, Any]) -> go.Figure:
        """에이전트 네트워크 생성"""
        # 네트워크 그래프 생성
        G = nx.Graph()
        
        # 에이전트들을 노드로 추가
        for agent_id, agent_info in collaboration_data.get('agents', {}).items():
            G.add_node(agent_id, **agent_info)
        
        # 협업 관계를 엣지로 추가
        for collab in collaboration_data.get('collaborations', []):
            source = collab['source']
            target = collab['target']
            weight = collab.get('interaction_count', 1)
            G.add_edge(source, target, weight=weight)
        
        # 중심성 계산
        centrality = nx.betweenness_centrality(G)
        
        # 레이아웃 계산
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 엣지 그리기
        edge_x, edge_y = [], []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 그리기
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_sizes = [centrality[node] * 1000 + 20 for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=[centrality[node] for node in G.nodes()],
                colorscale='Viridis',
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.1,
                    title="Centrality"
                ),
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Agent Collaboration Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size and color represent betweenness centrality",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='#888', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    async def create_real_time_dashboard(self, dashboard_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """실시간 대시보드 생성"""
        dashboard = {}
        
        # 시스템 상태 게이지
        system_health = dashboard_data.get('system_health', 75)
        dashboard['system_health'] = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=system_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health (%)"},
            delta={'reference': 80},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "green"}
                   ],
                   'threshold': {
                       'line': {'color': "red", 'width': 4},
                       'thickness': 0.75,
                       'value': 90
                   }}
        ))
        
        # 실시간 메트릭
        timestamps = dashboard_data.get('timestamps', [])
        cpu_data = dashboard_data.get('cpu_usage', [])
        memory_data = dashboard_data.get('memory_usage', [])
        
        dashboard['real_time_metrics'] = go.Figure()
        dashboard['real_time_metrics'].add_trace(go.Scatter(
            x=timestamps,
            y=cpu_data,
            mode='lines',
            name='CPU Usage (%)',
            line=dict(color='blue', width=2)
        ))
        dashboard['real_time_metrics'].add_trace(go.Scatter(
            x=timestamps,
            y=memory_data,
            mode='lines',
            name='Memory Usage (%)',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        dashboard['real_time_metrics'].update_layout(
            title='Real-time System Metrics',
            xaxis=dict(title='Time'),
            yaxis=dict(title='CPU Usage (%)', side='left'),
            yaxis2=dict(title='Memory Usage (%)', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        return dashboard
    
    async def _preprocess_data(self, data: ChartData, config: VisualizationConfig) -> Any:
        """데이터 전처리"""
        if isinstance(data.data, pd.DataFrame):
            return data.data
        elif isinstance(data.data, dict):
            # 딕셔너리를 DataFrame으로 변환 시도
            try:
                return pd.DataFrame(data.data)
            except:
                return data.data
        else:
            return data.data
    
    async def _create_chart(self, config: VisualizationConfig, data: Any) -> go.Figure:
        """차트 생성"""
        chart_type = config.chart_type
        
        if chart_type == VisualizationType.WORKFLOW_GRAPH:
            return await self.create_workflow_graph(data)
        elif chart_type == VisualizationType.PERFORMANCE_TIMELINE:
            return await self.create_performance_timeline(data)
        elif chart_type == VisualizationType.RESOURCE_HEATMAP:
            return await self.create_resource_heatmap(data)
        elif chart_type == VisualizationType.AGENT_NETWORK:
            return await self.create_agent_network(data)
        elif chart_type == VisualizationType.EXECUTION_GANTT:
            return await self.create_execution_gantt(data)
        else:
            # 기본 차트
            return go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
    
    async def _apply_style(self, chart: go.Figure, config: VisualizationConfig) -> go.Figure:
        """스타일 적용"""
        style_config = self.styles.get(config.style, self.styles[ChartStyle.PROFESSIONAL])
        
        chart.update_layout(
            template=style_config['template'],
            font_family=style_config['font_family'],
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            title=config.title
        )
        
        return chart
    
    async def _export_chart(self, chart: go.Figure, config: VisualizationConfig) -> str:
        """차트 내보내기"""
        if config.export_format.lower() == 'html':
            return chart.to_html(
                include_plotlyjs='cdn',
                config={'displayModeBar': config.interactive}
            )
        elif config.export_format.lower() == 'json':
            return chart.to_json()
        else:
            # 다른 형식들은 파일로 저장
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.export_format}"
            
            if config.export_format.lower() == 'png':
                chart.write_image(filename, format='png')
            elif config.export_format.lower() == 'svg':
                chart.write_image(filename, format='svg')
            elif config.export_format.lower() == 'pdf':
                chart.write_image(filename, format='pdf')
            
            return filename
    
    def _count_data_points(self, data: Any) -> int:
        """데이터 포인트 개수 계산"""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, (list, dict)):
            return len(data)
        else:
            return 1
    
    async def update_real_time_chart(self, chart_id: str, new_data: ChartData):
        """실시간 차트 업데이트"""
        if chart_id in self.chart_cache:
            # 캐시된 차트 업데이트
            cached_chart = self.chart_cache[chart_id]
            
            # WebSocket으로 업데이트 전송
            for connection_id, connection in self.websocket_connections.items():
                try:
                    await connection.send_json({
                        'type': 'chart_update',
                        'chart_id': chart_id,
                        'data': asdict(new_data)
                    })
                except Exception as e:
                    self.logger.warning("WebSocket 업데이트 실패", connection_id=connection_id, error=str(e))
    
    async def create_comparison_chart(self, comparison_data: Dict[str, List[float]]) -> go.Figure:
        """비교 차트 생성"""
        categories = list(comparison_data.keys())
        
        fig = go.Figure()
        
        for i, (category, values) in enumerate(comparison_data.items()):
            fig.add_trace(go.Box(
                y=values,
                name=category,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title='Performance Comparison',
            yaxis_title='Values',
            showlegend=True
        )
        
        return fig
    
    async def create_distribution_plot(self, data: List[float], title: str = "Distribution") -> go.Figure:
        """분포 플롯 생성"""
        fig = go.Figure()
        
        # 히스토그램
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            name='Histogram',
            opacity=0.7
        ))
        
        # 밀도 곡선
        from scipy import stats
        x_range = np.linspace(min(data), max(data), 100)
        kde = stats.gaussian_kde(data)
        density = kde(x_range)
        
        # 밀도를 히스토그램 스케일에 맞춤
        scale_factor = len(data) * (max(data) - min(data)) / 30
        density_scaled = density * scale_factor
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density_scaled,
            mode='lines',
            name='Density',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        
        return fig
    
    async def create_correlation_matrix(self, correlation_data: pd.DataFrame) -> go.Figure:
        """상관관계 매트릭스 생성"""
        corr_matrix = correlation_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=600,
            height=600
        )
        
        return fig
    
    def get_cached_charts(self) -> Dict[str, Any]:
        """캐시된 차트들 반환"""
        return {
            chart_id: {
                'config': chart_info['config'],
                'metadata': chart_info['metadata']
            }
            for chart_id, chart_info in self.chart_cache.items()
        }
    
    async def clear_cache(self):
        """캐시 정리"""
        self.chart_cache.clear()
        self.logger.info("차트 캐시가 정리되었습니다")


# 전역 시각화 엔진 인스턴스
_visualization_engine = None

def get_visualization_engine() -> AdvancedVisualizationEngine:
    """글로벌 시각화 엔진 획득"""
    global _visualization_engine
    if _visualization_engine is None:
        _visualization_engine = AdvancedVisualizationEngine()
    return _visualization_engine