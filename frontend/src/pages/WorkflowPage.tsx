import React from 'react'
import { WorkflowMonitor } from '@/components/dashboard/WorkflowMonitor'
import './WorkflowPage.scss'

export const WorkflowPage: React.FC = () => {
  return (
    <div className="workflow-page">
      <div className="page-header">
        <h1>Workflow Monitor</h1>
        <p className="page-subtitle">
          Monitor and manage AI agent workflows in real-time
        </p>
      </div>
      
      <WorkflowMonitor className="full-page-monitor" />
    </div>
  )
}