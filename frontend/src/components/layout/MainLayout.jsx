import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import { Menu } from 'lucide-react';

const MainLayout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  // Landing page gets no sidebar
  const isLanding = location.pathname === '/landing';
  if (isLanding) return <>{children}</>;

  return (
    <div className="av-layout">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      <main className="av-main bg-grid">
        {/* Mobile top bar */}
        <div className="lg:hidden fixed top-0 left-0 right-0 z-20 glass-static px-4 py-3 flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--av-text-primary)' }}
            aria-label="Open menu"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="text-sm font-semibold" style={{ fontFamily: 'Space Grotesk' }}>
            AVIATOR
          </span>
        </div>

        {/* Background glow effect */}
        <div className="bg-radial-glow pointer-events-none fixed inset-0 z-0" />

        {/* Page content */}
        <div className="relative z-10 av-content pt-4 lg:pt-0">
          {children}
        </div>
      </main>
    </div>
  );
};

export default MainLayout;
