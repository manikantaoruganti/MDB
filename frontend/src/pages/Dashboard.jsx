import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Plane, TrendingUp, Map, Globe } from 'lucide-react';
import StatWidget from '@/components/shared/StatWidget';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import PageHeader from '@/components/shared/PageHeader';
import { BarChart3 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/analytics/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Flight analytics overview and quick navigation"
        icon={BarChart3}
      />

      {/* Stats Grid */}
      {loading ? (
        <LoadingSkeleton type="cards" />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10 stagger-children">
          <StatWidget
            icon={<Map className="w-6 h-6" />}
            title="Airports"
            value={stats?.total_airports || 0}
            color="blue"
            delay={0}
          />
          <StatWidget
            icon={<Plane className="w-6 h-6" />}
            title="Airlines"
            value={stats?.total_airlines || 0}
            color="violet"
            delay={100}
          />
          <StatWidget
            icon={<TrendingUp className="w-6 h-6" />}
            title="Routes"
            value={stats?.total_routes || 0}
            color="cyan"
            delay={200}
          />
          <StatWidget
            icon={<Globe className="w-6 h-6" />}
            title="Countries"
            value={stats?.total_countries || 0}
            color="emerald"
            delay={300}
          />
        </div>
      )}

      {/* Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 fade-in-up" style={{ animationDelay: '300ms' }}>
        <Link to="/analytics" className="no-underline">
          <GlassCard className="group cursor-pointer h-full" data-testid="analytics-card">
            <div className="p-3 rounded-xl w-fit mb-4" style={{ background: 'rgba(59,130,246,0.12)' }}>
              <TrendingUp className="w-8 h-8" style={{ color: '#60a5fa' }} />
            </div>
            <h2 className="text-xl font-bold mb-2" style={{ fontFamily: 'Space Grotesk' }}>
              Analytics Dashboard
            </h2>
            <p className="text-sm mb-5" style={{ color: 'var(--av-text-muted)' }}>
              Explore busiest airports, top airlines, popular routes, and more
              with interactive visualizations
            </p>
            <span className="btn-primary inline-flex items-center gap-2 text-sm" data-testid="view-analytics-btn">
              View Analytics
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </span>
          </GlassCard>
        </Link>

        <Link to="/recommendations" className="no-underline">
          <GlassCard className="group cursor-pointer h-full" data-testid="recommendations-card">
            <div className="p-3 rounded-xl w-fit mb-4" style={{ background: 'rgba(139,92,246,0.12)' }}>
              <Plane className="w-8 h-8" style={{ color: '#a78bfa' }} />
            </div>
            <h2 className="text-xl font-bold mb-2" style={{ fontFamily: 'Space Grotesk' }}>
              Route Recommendations
            </h2>
            <p className="text-sm mb-5" style={{ color: 'var(--av-text-muted)' }}>
              Get AI-powered route suggestions using TF-IDF vector embeddings
              and cosine similarity
            </p>
            <span className="btn-primary inline-flex items-center gap-2 text-sm" data-testid="find-routes-btn">
              Find Routes
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </span>
          </GlassCard>
        </Link>
      </div>

      {/* Tech Info */}
      <div className="mt-10 fade-in-up" style={{ animationDelay: '500ms' }}>
        <GlassCard>
          <h3 className="text-lg font-semibold mb-6 text-center gradient-text" style={{ fontFamily: 'Space Grotesk' }}>
            Technology Stack
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-sm font-semibold mb-2" style={{ color: '#60a5fa' }}>Database</div>
              <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>
                MongoDB with vector embeddings for similarity search
              </p>
            </div>
            <div className="text-center">
              <div className="text-sm font-semibold mb-2" style={{ color: '#a78bfa' }}>Machine Learning</div>
              <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>
                TF-IDF vectorization with cosine similarity ranking
              </p>
            </div>
            <div className="text-center">
              <div className="text-sm font-semibold mb-2" style={{ color: '#fb7185' }}>Data Source</div>
              <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>
                OpenFlights dataset: 7,698 airports, 67,240 routes
              </p>
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  );
};

export default Dashboard;
