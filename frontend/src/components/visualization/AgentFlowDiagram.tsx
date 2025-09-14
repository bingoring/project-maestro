import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { sankey, sankeyLinkHorizontal, SankeyGraph, SankeyNode, SankeyLink } from 'd3-sankey'
import { Tile, Tag, Tooltip } from '@carbon/react'
import { maestroCustomColors, agentTypeColors } from '@/setup/carbon-theme'
import type { Agent, WorkflowState, WorkflowConnection } from '@types'
import './AgentFlowDiagram.scss'

interface FlowNode extends SankeyNode<{}, {}> {
  id: string
  name: string
  type: Agent['type']
  status: Agent['status']
  metrics?: Agent['metrics']
}

interface FlowLink extends SankeyLink<FlowNode, {}> {
  id: string
  type: WorkflowConnection['type']
  status: WorkflowConnection['status']
  dataFlow?: number
}

interface AgentFlowDiagramProps {
  workflowData: WorkflowState
  interactive?: boolean
  width?: number
  height?: number
  onNodeClick?: (agent: Agent) => void
  onNodeHover?: (agent: Agent | null) => void
  className?: string
}

export const AgentFlowDiagram: React.FC<AgentFlowDiagramProps> = ({
  workflowData,
  interactive = true,
  width = 1200,
  height = 600,
  onNodeClick,
  onNodeHover,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const { nodes, links } = useMemo(() => {
    const nodes: FlowNode[] = workflowData.agents.map(agent => ({
      id: agent.id,
      name: agent.name,
      type: agent.type,
      status: agent.status,
      metrics: agent.metrics,
    }))

    const links: FlowLink[] = workflowData.connections.map(conn => ({
      id: conn.id,
      source: nodes.find(n => n.id === conn.from)!,
      target: nodes.find(n => n.id === conn.to)!,
      type: conn.type,
      status: conn.status,
      dataFlow: conn.dataFlow || 1,
      value: conn.dataFlow || 1,
    }))

    return { nodes, links }
  }, [workflowData])

  const getNodeColor = (status: Agent['status'], type: Agent['type']) => {
    if (status === 'error') return maestroCustomColors.agent.error
    if (status === 'complete') return maestroCustomColors.agent.complete
    if (status === 'executing') return maestroCustomColors.agent.executing
    if (status === 'waiting') return maestroCustomColors.agent.waiting
    if (status === 'planning') return maestroCustomColors.agent.planning
    return agentTypeColors[type] || maestroCustomColors.agent.idle
  }

  const getLinkColor = (type: WorkflowConnection['type'], status: WorkflowConnection['status']) => {
    if (status === 'inactive') return '#525252'
    
    switch (type) {
      case 'data': return '#0f62fe'
      case 'control': return '#8a3ffc'
      case 'feedback': return '#f1c21b'
      default: return '#393939'
    }
  }

  const showTooltip = (event: MouseEvent, node: FlowNode) => {
    if (!tooltipRef.current) return

    const tooltip = tooltipRef.current
    tooltip.innerHTML = `
      <div class="tooltip-content">
        <h4>${node.name}</h4>
        <p><strong>Type:</strong> ${node.type}</p>
        <p><strong>Status:</strong> ${node.status}</p>
        ${node.metrics ? `
          <div class="metrics">
            <p><strong>Tokens:</strong> ${node.metrics.tokensUsed.toLocaleString()}</p>
            <p><strong>Time:</strong> ${node.metrics.executionTime.toFixed(2)}s</p>
            <p><strong>Memory:</strong> ${node.metrics.memoryUsage.toFixed(1)}MB</p>
            <p><strong>Success Rate:</strong> ${(node.metrics.successRate * 100).toFixed(1)}%</p>
          </div>
        ` : ''}
      </div>
    `
    
    tooltip.style.display = 'block'
    tooltip.style.left = `${event.pageX + 10}px`
    tooltip.style.top = `${event.pageY - 10}px`
  }

  const hideTooltip = () => {
    if (tooltipRef.current) {
      tooltipRef.current.style.display = 'none'
    }
  }

  useEffect(() => {
    if (!svgRef.current || !nodes.length) return

    const svg = d3.select(svgRef.current)
    const margin = { top: 20, right: 200, bottom: 20, left: 20 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Clear previous content
    svg.selectAll('*').remove()

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    // Create sankey layout
    const sankeyGenerator = sankey<FlowNode, FlowLink>()
      .nodeWidth(40)
      .nodePadding(20)
      .extent([[0, 0], [innerWidth, innerHeight]])
      .nodeId(d => d.id)

    // Generate sankey data
    const graph: SankeyGraph<FlowNode, FlowLink> = {
      nodes: [...nodes],
      links: [...links],
    }

    sankeyGenerator(graph)

    // Create gradient definitions
    const defs = svg.append('defs')
    
    // Gradient for links
    const gradient = defs.append('linearGradient')
      .attr('id', 'link-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#0f62fe')
      .attr('stop-opacity', 0.8)
    
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#42be65')
      .attr('stop-opacity', 0.8)

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('.link')
      .data(graph.links)
      .enter().append('path')
      .attr('class', 'link')
      .attr('d', sankeyLinkHorizontal())
      .style('stroke', d => getLinkColor(d.type, d.status))
      .style('stroke-width', d => Math.max(2, d.width || 2))
      .style('fill', 'none')
      .style('opacity', 0.6)

    // Add link labels
    g.append('g')
      .attr('class', 'link-labels')
      .selectAll('.link-label')
      .data(graph.links)
      .enter().append('text')
      .attr('class', 'link-label')
      .attr('x', d => {
        const source = d.source as FlowNode
        const target = d.target as FlowNode
        return (source.x1! + target.x0!) / 2
      })
      .attr('y', d => {
        const source = d.source as FlowNode
        const target = d.target as FlowNode
        return (source.y0! + source.y1! + target.y0! + target.y1!) / 4
      })
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('font-size', '10px')
      .style('fill', '#c6c6c6')
      .text(d => d.type)

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('.node')
      .data(graph.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x0}, ${d.y0})`)

    // Node rectangles
    node.append('rect')
      .attr('height', d => d.y1! - d.y0!)
      .attr('width', sankeyGenerator.nodeWidth())
      .style('fill', d => getNodeColor(d.status, d.type))
      .style('stroke', '#393939')
      .style('stroke-width', 2)
      .style('rx', 4)
      .style('cursor', interactive ? 'pointer' : 'default')

    // Node labels
    node.append('text')
      .attr('x', -6)
      .attr('y', d => (d.y1! - d.y0!) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .style('font-size', '12px')
      .style('font-weight', '500')
      .style('fill', '#f4f4f4')
      .text(d => d.name)

    // Status indicators
    node.append('circle')
      .attr('cx', sankeyGenerator.nodeWidth() + 10)
      .attr('cy', d => (d.y1! - d.y0!) / 2)
      .attr('r', 4)
      .style('fill', d => getNodeColor(d.status, d.type))
      .style('stroke', '#f4f4f4')
      .style('stroke-width', 1)

    // Progress bars for executing agents
    node.filter(d => d.status === 'executing')
      .append('rect')
      .attr('x', 2)
      .attr('y', d => d.y1! - d.y0! - 6)
      .attr('width', sankeyGenerator.nodeWidth() - 4)
      .attr('height', 2)
      .style('fill', '#262626')
      .style('rx', 1)

    node.filter(d => d.status === 'executing')
      .append('rect')
      .attr('x', 2)
      .attr('y', d => d.y1! - d.y0! - 6)
      .attr('width', d => {
        const progress = workflowData.agents.find(a => a.id === d.id)?.progress || 0
        return ((sankeyGenerator.nodeWidth() - 4) * progress) / 100
      })
      .attr('height', 2)
      .style('fill', maestroCustomColors.agent.executing)
      .style('rx', 1)

    // Interactive features
    if (interactive) {
      node.on('mouseover', function(event, d) {
        // Highlight connected links
        link.style('opacity', l => 
          l.source === d || l.target === d ? 0.9 : 0.2
        )
        
        // Show tooltip
        showTooltip(event, d)
        onNodeHover?.(workflowData.agents.find(a => a.id === d.id) || null)
      })
      .on('mouseout', function() {
        // Reset link opacity
        link.style('opacity', 0.6)
        
        // Hide tooltip
        hideTooltip()
        onNodeHover?.(null)
      })
      .on('click', function(event, d) {
        const agent = workflowData.agents.find(a => a.id === d.id)
        if (agent) {
          onNodeClick?.(agent)
        }
      })
    }

    // Animation for workflow status
    if (workflowData.status === 'running') {
      // Animate data flow
      const animateFlow = () => {
        link
          .style('stroke-dasharray', '8, 4')
          .style('stroke-dashoffset', 0)
          .transition()
          .duration(2000)
          .ease(d3.easeLinear)
          .style('stroke-dashoffset', -12)
          .on('end', animateFlow)
      }
      
      // Start animation after a short delay
      setTimeout(animateFlow, 500)
    }

  }, [nodes, links, workflowData, width, height, interactive, onNodeClick, onNodeHover])

  return (
    <div className={`agent-flow-diagram ${className}`}>
      <div className="diagram-container">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="flow-svg"
        />
        
        {/* Tooltip */}
        <div
          ref={tooltipRef}
          className="flow-tooltip"
          style={{ display: 'none' }}
        />

        {/* Legend */}
        <div className="diagram-legend">
          <div className="legend-section">
            <h4>Agent Status</h4>
            <div className="legend-items">
              <div className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: maestroCustomColors.agent.idle }}
                />
                <span>Idle</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: maestroCustomColors.agent.planning }}
                />
                <span>Planning</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: maestroCustomColors.agent.executing }}
                />
                <span>Executing</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: maestroCustomColors.agent.complete }}
                />
                <span>Complete</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: maestroCustomColors.agent.error }}
                />
                <span>Error</span>
              </div>
            </div>
          </div>

          <div className="legend-section">
            <h4>Connection Type</h4>
            <div className="legend-items">
              <div className="legend-item">
                <div 
                  className="legend-line" 
                  style={{ backgroundColor: '#0f62fe' }}
                />
                <span>Data Flow</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-line" 
                  style={{ backgroundColor: '#8a3ffc' }}
                />
                <span>Control</span>
              </div>
              <div className="legend-item">
                <div 
                  className="legend-line" 
                  style={{ backgroundColor: '#f1c21b' }}
                />
                <span>Feedback</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}