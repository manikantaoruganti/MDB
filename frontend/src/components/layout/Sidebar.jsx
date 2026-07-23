import { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, BarChart3, TrendingUp, Globe2, Network,
  Route, Building2, Plane, Flag, Brain, ChevronDown,
  Menu, X, Activity, Zap, Search
} from 'lucide-react';
import { APP_NAME, APP_SUBTITLE, NAV_SECTIONS } from '@/lib/constants';

const ICON_MAP = {
  LayoutDashboard, BarChart3, TrendingUp, Globe2, Network,
  Route, Building2, Plane, Flag, Brain,
};

const Sidebar = ({ isOpen, onToggle }) => {
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState(
    NAV_SECTIONS.map(() => true)
  );

  const toggleSection = (idx) => {
    setExpandedSections(prev => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 lg:hidden"
          onClick={onToggle}
        />
      )}

      <aside className={`av-sidebar ${isOpen ? 'open' : ''}`}>
        {/* Logo */}
        <div className="p-5 border-b border-[var(--av-glass-border)]">
          <Link to="/" className="flex items-center gap-3 no-underline">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'var(--av-gradient-primary)' }}>
              <Plane className="w-5 h-5 text-white" style={{ transform: 'rotate(-45deg)' }} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight" style={{ fontFamily: 'Space Grotesk' }}>
                {APP_NAME}
              </h1>
              <p className="text-[10px] text-[var(--av-text-muted)] uppercase tracking-widest">
                {APP_SUBTITLE}
              </p>
            </div>
          </Link>
        </div>

        {/* Search */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg"
            style={{ background: 'rgba(99, 102, 241, 0.06)', border: '1px solid var(--av-glass-border)' }}>
            <Search className="w-4 h-4 text-[var(--av-text-muted)]" />
            <span className="text-xs text-[var(--av-text-muted)]">Quick search...</span>
            <span className="ml-auto text-[10px] text-[var(--av-text-muted)] font-mono bg-[var(--av-surface)] px-1.5 py-0.5 rounded">⌘K</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-2 space-y-1">
          {NAV_SECTIONS.map((section, sIdx) => (
            <div key={section.title} className="mb-2">
              <button
                onClick={() => toggleSection(sIdx)}
                className="flex items-center justify-between w-full px-3 py-1.5 text-[10px] uppercase tracking-widest text-[var(--av-text-muted)] hover:text-[var(--av-text-secondary)] transition-colors"
                style={{ background: 'none', border: 'none', cursor: 'pointer' }}
              >
                {section.title}
                <ChevronDown
                  className={`w-3 h-3 transition-transform ${expandedSections[sIdx] ? '' : '-rotate-90'}`}
                />
              </button>

              {expandedSections[sIdx] && (
                <div className="space-y-0.5 mt-1">
                  {section.items.map((item) => {
                    const Icon = ICON_MAP[item.icon] || LayoutDashboard;
                    const isActive = location.pathname === item.path;

                    return (
                      <Link
                        key={item.path}
                        to={item.path}
                        className={`nav-item ${isActive ? 'active' : ''}`}
                        onClick={() => window.innerWidth < 1024 && onToggle?.()}
                      >
                        <Icon className="w-4 h-4" />
                        <span>{item.label}</span>
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </nav>

        {/* Status Footer */}
        <div className="p-4 border-t border-[var(--av-glass-border)]">
          <div className="glass-static rounded-xl p-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="status-dot online" />
              <span className="text-xs font-medium text-[var(--av-text-secondary)]">System Online</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] text-[var(--av-text-muted)]">
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3" /> API Active
              </span>
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" /> ML Ready
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
