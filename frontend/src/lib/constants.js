// Application constants
export const APP_NAME = 'AVIATOR';
export const APP_SUBTITLE = 'Aviation Intelligence Platform';
export const APP_VERSION = '2.0.0';

export const COLORS = {
  blue: { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa', hex: '#3b82f6' },
  violet: { bg: 'rgba(139, 92, 246, 0.15)', text: '#a78bfa', hex: '#8b5cf6' },
  cyan: { bg: 'rgba(6, 182, 212, 0.15)', text: '#22d3ee', hex: '#06b6d4' },
  emerald: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', hex: '#10b981' },
  rose: { bg: 'rgba(244, 63, 94, 0.15)', text: '#fb7185', hex: '#f43f5e' },
  amber: { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24', hex: '#f59e0b' },
  indigo: { bg: 'rgba(99, 102, 241, 0.15)', text: '#818cf8', hex: '#6366f1' },
};

export const CHART_COLORS = [
  '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f43f5e',
  '#f59e0b', '#6366f1', '#ec4899', '#14b8a6', '#a855f7',
];

export const NAV_SECTIONS = [
  {
    title: 'Overview',
    items: [
      { path: '/', label: 'Command Center', icon: 'LayoutDashboard' },
      { path: '/dashboard', label: 'Dashboard', icon: 'BarChart3' },
    ]
  },
  {
    title: 'Analytics',
    items: [
      { path: '/analytics', label: 'Analytics', icon: 'TrendingUp' },
      { path: '/globe', label: 'Globe Explorer', icon: 'Globe2' },
      { path: '/network', label: 'Network Graph', icon: 'Network' },
    ]
  },
  {
    title: 'Intelligence',
    items: [
      { path: '/recommendations', label: 'Route Finder', icon: 'Route' },
      { path: '/airports', label: 'Airport Intel', icon: 'Building2' },
      { path: '/airlines', label: 'Airline Intel', icon: 'Plane' },
      { path: '/countries', label: 'Country Index', icon: 'Flag' },
    ]
  },
  {
    title: 'AI Lab',
    items: [
      { path: '/ml-lab', label: 'ML Lab', icon: 'Brain' },
    ]
  }
];
