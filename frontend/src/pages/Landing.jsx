import { Link } from 'react-router-dom';
import { Plane, ChevronRight, Activity, Globe2, ShieldCheck } from 'lucide-react';
import { APP_NAME, APP_SUBTITLE } from '@/lib/constants';

const Landing = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden bg-[var(--av-void)]">
      {/* Background elements */}
      <div className="absolute inset-0 bg-grid opacity-20 pointer-events-none" />
      <div className="bg-radial-glow absolute inset-0 pointer-events-none" />
      
      {/* Particle effect simulation */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {Array.from({ length: 20 }).map((_, i) => (
          <div 
            key={i}
            className="absolute rounded-full"
            style={{
              width: Math.random() * 4 + 1 + 'px',
              height: Math.random() * 4 + 1 + 'px',
              background: 'rgba(99, 102, 241, 0.4)',
              top: Math.random() * 100 + '%',
              left: Math.random() * 100 + '%',
              animation: `float ${Math.random() * 10 + 5}s linear infinite`,
              opacity: Math.random() * 0.5 + 0.1
            }}
          />
        ))}
      </div>

      <div className="relative z-10 text-center max-w-4xl px-4 fade-in-up">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 glass-static border border-indigo-500/30">
          <span className="status-dot online" />
          <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#a78bfa' }}>
            System V2.0 Online
          </span>
        </div>

        <h1 className="text-6xl md:text-8xl font-bold mb-6 tracking-tighter" style={{ fontFamily: 'Space Grotesk' }}>
          Welcome to <br />
          <span className="gradient-text">{APP_NAME}</span>
        </h1>
        
        <p className="text-xl md:text-2xl mb-12 max-w-2xl mx-auto leading-relaxed" style={{ color: 'var(--av-text-secondary)' }}>
          The enterprise-grade {APP_SUBTITLE.toLowerCase()} powered by vector embeddings and global data.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <Link to="/" className="w-full sm:w-auto">
            <button className="btn-primary h-14 px-8 text-lg w-full flex items-center justify-center gap-2 group">
              Access Command Center
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </Link>
          <Link to="/globe" className="w-full sm:w-auto">
            <button className="btn-ghost h-14 px-8 text-lg w-full border border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/10 backdrop-blur-md rounded-xl">
              Explore 3D Globe
            </button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left stagger-children border-t border-white/10 pt-12">
          <div>
            <Activity className="w-6 h-6 mb-3 text-blue-400" />
            <h3 className="font-semibold mb-2">Real-time Analytics</h3>
            <p className="text-sm text-slate-400">Process and analyze thousands of routes and aviation metrics instantly.</p>
          </div>
          <div>
            <Globe2 className="w-6 h-6 mb-3 text-violet-400" />
            <h3 className="font-semibold mb-2">Global Network</h3>
            <p className="text-sm text-slate-400">Comprehensive coverage of over 7,600 airports and 67,000 routes worldwide.</p>
          </div>
          <div>
            <ShieldCheck className="w-6 h-6 mb-3 text-emerald-400" />
            <h3 className="font-semibold mb-2">AI Embeddings</h3>
            <p className="text-sm text-slate-400">Advanced TF-IDF vectorization for intelligent similarity search.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
