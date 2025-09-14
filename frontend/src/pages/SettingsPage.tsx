import React from 'react'
import { Grid, Column, Tile } from '@carbon/react'
import './SettingsPage.scss'

export const SettingsPage: React.FC = () => {
  return (
    <div className="settings-page">
      <div className="page-header">
        <h1>Settings</h1>
        <p className="page-subtitle">
          Configure your Project Maestro experience
        </p>
      </div>
      
      <Grid>
        <Column span={16}>
          <Tile className="coming-soon-tile">
            <div className="coming-soon-content">
              <h2>⚙️ Settings Panel</h2>
              <p>
                Configuration options and user preferences coming soon.
                This will include API settings, notification preferences,
                theme customization, and workflow defaults.
              </p>
              
              <div className="planned-features">
                <h3>Planned Settings:</h3>
                <ul>
                  <li>API Configuration & Keys</li>
                  <li>Notification Preferences</li>
                  <li>Theme & Display Options</li>
                  <li>Workflow Default Settings</li>
                  <li>Agent Behavior Configuration</li>
                  <li>Performance Optimization</li>
                </ul>
              </div>
            </div>
          </Tile>
        </Column>
      </Grid>
    </div>
  )
}