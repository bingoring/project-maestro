import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Header,
  HeaderContainer,
  HeaderName,
  HeaderNavigation,
  HeaderMenuItem,
  HeaderGlobalBar,
  HeaderGlobalAction,
  SideNav,
  SideNavItems,
  SideNavMenuItem,
  SideNavMenu,
  SideNavLink,
  Content,
  Theme,
  SkipToContent,
} from '@carbon/react'
import {
  Notification,
  UserAvatar,
  Settings,
  Dashboard,
  Flow,
  Analytics,
  Menu,
} from '@carbon/icons-react'
import type { NavigationItem } from '@types'
import { maestroCustomColors } from '@/setup/carbon-theme'
import './AppLayout.scss'

interface AppLayoutProps {
  children: React.ReactNode
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: Dashboard,
    path: '/dashboard',
  },
  {
    id: 'workflow',
    label: 'Workflow Monitor',
    icon: Flow,
    path: '/workflow',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: Analytics,
    path: '/analytics',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    path: '/settings',
  },
]

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const [isSideNavExpanded, setIsSideNavExpanded] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()

  const isActive = (path: string) => {
    return location.pathname === path
  }

  const handleNavigation = (path: string) => {
    navigate(path)
    setIsSideNavExpanded(false) // Close side nav on mobile after navigation
  }

  return (
    <div className="app-layout">
      <HeaderContainer
        render={() => (
          <>
            <Header aria-label="Project Maestro">
              <SkipToContent />
              
              <HeaderName prefix="IBM">
                Project Maestro
              </HeaderName>

              {/* Desktop navigation */}
              <HeaderNavigation aria-label="Project Maestro" className="hide-mobile">
                {navigationItems.map((item) => (
                  <HeaderMenuItem
                    key={item.id}
                    isActive={isActive(item.path)}
                    onClick={() => handleNavigation(item.path)}
                  >
                    {item.label}
                  </HeaderMenuItem>
                ))}
              </HeaderNavigation>

              <HeaderGlobalBar>
                <HeaderGlobalAction
                  aria-label="Notifications"
                  tooltipAlignment="end"
                  onClick={() => {
                    console.log('Notifications clicked')
                  }}
                >
                  <Notification size={20} />
                </HeaderGlobalAction>
                
                <HeaderGlobalAction
                  aria-label="User Settings"
                  tooltipAlignment="end"
                  onClick={() => handleNavigation('/settings')}
                >
                  <UserAvatar size={20} />
                </HeaderGlobalAction>
                
                {/* Mobile menu toggle */}
                <HeaderGlobalAction
                  aria-label="Open menu"
                  tooltipAlignment="end"
                  className="mobile-menu-toggle"
                  onClick={() => setIsSideNavExpanded(!isSideNavExpanded)}
                >
                  <Menu size={20} />
                </HeaderGlobalAction>
              </HeaderGlobalBar>
            </Header>

            {/* Mobile side navigation */}
            <SideNav
              aria-label="Side navigation"
              expanded={isSideNavExpanded}
              onOverlayClick={() => setIsSideNavExpanded(false)}
              className="mobile-side-nav"
            >
              <SideNavItems>
                {navigationItems.map((item) => {
                  const IconComponent = item.icon
                  return (
                    <SideNavMenuItem
                      key={item.id}
                      isActive={isActive(item.path)}
                      onClick={() => handleNavigation(item.path)}
                    >
                      <div className="side-nav-item">
                        <IconComponent size={16} />
                        {item.label}
                      </div>
                    </SideNavMenuItem>
                  )
                })}
              </SideNavItems>
            </SideNav>
          </>
        )}
      />

      <Content className="app-content">
        <div className="content-wrapper">
          {children}
        </div>
      </Content>

      {/* Connection status indicator */}
      <div className="connection-indicator" id="connection-status">
        {/* This will be updated by WebSocket connection status */}
      </div>
    </div>
  )
}