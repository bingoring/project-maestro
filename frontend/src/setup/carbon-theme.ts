import { g100 } from '@carbon/themes'
import type { MaestroTheme } from '@types'

// Custom theme extending IBM Carbon g100 (dark theme)
export const maestroTheme = {
  ...g100,
}

// Custom color tokens for AI workflow visualization
export const maestroCustomColors: MaestroTheme = {
  agent: {
    idle: '#525252',       // Gray 70
    planning: '#0f62fe',   // Blue 60
    executing: '#42be65',  // Green 40  
    waiting: '#f1c21b',    // Yellow 30
    error: '#da1e28',      // Red 50
    complete: '#24a148',   // Green 50
  },
  workflow: {
    background: '#161616', // Gray 100 (darkest)
    border: '#393939',     // Gray 80
    hover: '#4c4c4c',      // Gray 70
  },
}

// Agent type colors for visual distinction
export const agentTypeColors = {
  orchestrator: '#8a3ffc', // Purple 60
  codex: '#0f62fe',        // Blue 60
  canvas: '#42be65',       // Green 40
  sonata: '#f1c21b',       // Yellow 30
  labyrinth: '#ff832b',    // Orange 40
  builder: '#da1e28',      // Red 50
}

// Status colors for various UI elements
export const statusColors = {
  success: '#24a148',      // Green 50
  warning: '#f1c21b',      // Yellow 30
  error: '#da1e28',        // Red 50
  info: '#0f62fe',         // Blue 60
  neutral: '#525252',      // Gray 70
}

// Chart colors for metrics and analytics
export const chartColors = [
  '#0f62fe', // Blue 60
  '#42be65', // Green 40
  '#f1c21b', // Yellow 30
  '#8a3ffc', // Purple 60
  '#ff832b', // Orange 40
  '#da1e28', // Red 50
  '#33b1ff', // Blue 40
  '#6fdc8c', // Green 30
  '#ffb000', // Yellow 20
  '#a56eff', // Purple 40
]

// Semantic color mappings
export const semanticColors = {
  primary: '#0f62fe',
  secondary: '#393939',
  accent: '#8a3ffc',
  background: {
    primary: '#161616',
    secondary: '#262626',
    tertiary: '#393939',
  },
  text: {
    primary: '#f4f4f4',
    secondary: '#c6c6c6',
    tertiary: '#8d8d8d',
    inverse: '#161616',
  },
  border: {
    subtle: '#393939',
    strong: '#6f6f6f',
    inverse: '#e0e0e0',
  },
}

// Animation and transition values
export const animations = {
  duration: {
    fast: '100ms',
    moderate: '240ms',
    slow: '400ms',
    slower: '700ms',
  },
  easing: {
    standard: 'cubic-bezier(0.2, 0, 0.38, 0.9)',
    entrance: 'cubic-bezier(0, 0, 0.38, 0.9)',
    exit: 'cubic-bezier(0.2, 0, 1, 0.9)',
    productive: 'cubic-bezier(0.2, 0, 0.38, 0.9)',
    expressive: 'cubic-bezier(0.4, 0.14, 0.3, 1)',
  },
}

// Spacing values
export const spacing = {
  xs: '0.25rem',   // 4px
  sm: '0.5rem',    // 8px
  md: '1rem',      // 16px
  lg: '1.5rem',    // 24px
  xl: '2rem',      // 32px
  xxl: '3rem',     // 48px
  xxxl: '4rem',    // 64px
}

// Breakpoints for responsive design
export const breakpoints = {
  sm: '320px',
  md: '672px',
  lg: '1056px',
  xlg: '1312px',
  max: '1584px',
}

export default maestroTheme