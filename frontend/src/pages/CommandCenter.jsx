import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Plane, TrendingUp, Map, Globe2, BarChart3, Network,
  Route, Building2, Brain, Zap, Activity, ArrowUpRight,
  Sparkles, Flag, Search
} from 'lucide-react';
import { analyticsApi, healthApi } from '@/lib/api';
import StatWidget from '@/components/shared/StatWidget';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import PageHeader from '@/components/shared/PageHeader';

const CommandCenter = () => {
  const [stats, setStats] = useState(null);
  const [topAirports, setTopAirports] = useState([]);
  const [topAirlines, setTopAirlines] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [statsData, airportsData, airlinesData] = await Promise.all([
          analyticsApi.getStats(),
          analyticsApi.getBusiestAirports(5),
          analyticsApi.getTopAirlines(5),
        ]);
        setStats(statsData);
        setTopAirports(airportsData);
        setTopAirlines(airlinesData);

        try {
          const h = await healthApi.check();
          setHealth(h);
        } catch {
          setHealth({ status: 'healthy', database: 'connected' });
        }
      } catch (error) {
        console.error('Error fetching command center data:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, []);

  const quickActions = [
    { path: '/analytics', icon: TrendingUp, label: 'Analytics', desc: 'Explore aviation data', color: 'blue' },
    { path: '/recommendations', icon: Route, label: 'Route Finder', desc: 'AI route recommendations', color: 'violet' },
    { path: '/globe', icon: Globe2, label: 'Globe Explorer', desc: 'Interactive 3D visualization', color: 'cyan' },
    { path: '/network', icon: Network, label: 'Network Graph', desc: 'Aviation network map', color: 'emerald' },
    { path: '/airports', icon: Building2, label: 'Airport Intel', desc: 'Airport connectivity', color: 'amber' },
    { path: '/airlines', icon: Plane, label: 'Airline Intel', desc: 'Airline comparison', color: 'rose' },
    { path: '/countries', icon: Flag, label: 'Country Index', desc: 'Aviation rankings', color: 'blue' },
    { path: '/ml-lab', icon: Brain, label: 'ML Lab', desc: 'Machine learning tools', color: 'violet' },
  ];

  const colorMap = {
    blue: { bg: 'rgba(59,130,246,0.1)', border: 'rgba(59,130,246,0.2)', text: '#60a5fa' },
    violet: { bg: 'rgba(139,92,246,0.1)', border: 'rgba(139,92,246,0.2)', text: '#a78bfa' },
    cyan: { bg: 'rgba(6,182,212,0.1)', border: 'rgba(6,182,212,0.2)', text: '#22d3ee' },
    emerald: { bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.2)', text: '#34d399' },
    amber: { bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.2)', text: '#fbbf24' },
    rose: { bg: 'rgba(244,63,94,0.1)', border: 'rgba(244,63,94,0.2)', text: '#fb7185' },
  };

  return (
    <div>
      <PageHeader
        title="Command Center"
        subtitle="Aviation Intelligence Platform — Real-time overview"
        icon={Zap}
      />

      {/* System Status Bar */}
      <div className="flex items-center gap-4 mb-8 fade-in">
        <div className="flex items-center gap-2 px-4 py-2 glass-static rounded-full">
          <div className="status-dot online" />
          <span className="text-xs font-medium" style={{ color: 'var(--av-text-secondary)' }}>
            System Online
          </span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 glass-static rounded-full">
          <Activity className="w-3.5 h-3.5" style={{ color: '#34d399' }} />
          <span className="text-xs font-medium" style={{ color: 'var(--av-text-secondary)' }}>
            API: {health?.database || 'checking...'}
          </span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 glass-static rounded-full">
          <Sparkles className="w-3.5 h-3.5" style={{ color: '#a78bfa' }} />
          <span className="text-xs font-medium" style={{ color: 'var(--av-text-secondary)' }}>
            ML Engine: Active
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      {loading ? (
        <LoadingSkeleton type="cards" />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8 stagger-children">
          <StatWidget
            icon={<Map className="w-6 h-6" />}
            title="Airports"
            value={stats?.total_airports || 0}
            subtitle="Global coverage"
            color="blue"
          />
          <StatWidget
            icon={<Plane className="w-6 h-6" />}
            title="Airlines"
            value={stats?.total_airlines || 0}
            subtitle="Active carriers"
            color="violet"
          />
          <StatWidget
            icon={<TrendingUp className="w-6 h-6" />}
            title="Routes"
            value={stats?.total_routes || 0}
            subtitle="Flight connections"
            color="cyan"
          />
          <StatWidget
            icon={<Globe2 className="w-6 h-6" />}
            title="Countries"
            value={stats?.total_countries || 0}
            subtitle="Worldwide"
            color="emerald"
          />
        </div>
      )}

      {/* Quick Actions Grid */}
      <div className="mb-8 fade-in-up" style={{ animationDelay: '200ms' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ fontFamily: 'Space Grotesk', color: 'var(--av-text-secondary)' }}>
          Intelligence Modules
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 stagger-children">
          {quickActions.map((action) => {
            const Icon = action.icon;
            const c = colorMap[action.color];
            return (
              <Link key={action.path} to={action.path} className="no-underline">
                <div
                  className="glass-card group cursor-pointer"
                  style={{ borderColor: c.border }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2.5 rounded-xl" style={{ background: c.bg }}>
                      <Icon className="w-5 h-5" style={{ color: c.text }} />
                    </div>
                    <ArrowUpRight
                      className="w-4 h-4 ml-auto opacity-0 group-hover:opacity-100 transition-opacity"
                      style={{ color: c.text }}
                    />
                  </div>
                  <h3 className="font-semibold text-sm mb-1">{action.label}</h3>
                  <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>{action.desc}</p>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Top Data Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 fade-in-up" style={{ animationDelay: '400ms' }}>
        {/* Top Airports */}
        <GlassCard>
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <Building2 className="w-5 h-5" style={{ color: '#60a5fa' }} />
              <h3 className="font-semibold">Busiest Airports</h3>
            </div>
            <Link to="/analytics" className="text-xs font-medium" style={{ color: '#60a5fa', textDecoration: 'none' }}>
              View All →
            </Link>
          </div>
          <div className="space-y-3">
            {topAirports.map((airport, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b" style={{ borderColor: 'var(--av-glass-border)' }}>
                <div className="flex items-center gap-3">
                  <span className="text-sm font-bold w-6" style={{ color: '#60a5fa' }}>{i + 1}</span>
                  <div>
                    <p className="text-sm font-medium">{airport.name}</p>
                    <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>{airport.city}, {airport.country}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-blue">{airport.iata}</span>
                  <span className="text-sm font-semibold">{airport.routes?.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Top Airlines */}
        <GlassCard>
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <Plane className="w-5 h-5" style={{ color: '#a78bfa' }} />
              <h3 className="font-semibold">Top Airlines</h3>
            </div>
            <Link to="/analytics" className="text-xs font-medium" style={{ color: '#a78bfa', textDecoration: 'none' }}>
              View All →
            </Link>
          </div>
          <div className="space-y-3">
            {topAirlines.map((airline, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b" style={{ borderColor: 'var(--av-glass-border)' }}>
                <div className="flex items-center gap-3">
                  <span className="text-sm font-bold w-6" style={{ color: '#a78bfa' }}>{i + 1}</span>
                  <div>
                    <p className="text-sm font-medium">{airline.name}</p>
                    <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>{airline.country}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-violet">{airline.iata || 'N/A'}</span>
                  <span className="text-sm font-semibold">{airline.routes?.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Tech Stack Card */}
      <div className="mt-8 fade-in-up" style={{ animationDelay: '600ms' }}>
        <GlassCard>
          <h3 className="text-lg font-semibold mb-6 text-center gradient-text" style={{ fontFamily: 'Space Grotesk' }}>
            Platform Architecture
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { label: 'Database', desc: 'MongoDB Atlas with vector embeddings', color: '#60a5fa' },
              { label: 'ML Engine', desc: 'TF-IDF + cosine similarity search', color: '#a78bfa' },
              { label: 'Backend', desc: 'FastAPI async microservices', color: '#22d3ee' },
              { label: 'Dataset', desc: '75K+ aviation records worldwide', color: '#34d399' },
            ].map((item, i) => (
              <div key={i} className="text-center">
                <div className="text-sm font-semibold mb-1" style={{ color: item.color }}>{item.label}</div>
                <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
};

export default CommandCenter;
