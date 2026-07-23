import { useState } from 'react';
import { Building2, Search, Network, Plane, Globe2, ArrowRightLeft } from 'lucide-react';
import { intelligenceApi } from '@/lib/api';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import StatWidget from '@/components/shared/StatWidget';

const AirportIntelligence = () => {
  const [iata, setIata] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!iata) return;
    setLoading(true);
    setSearched(true);
    try {
      const result = await intelligenceApi.getAirportConnectivity(iata);
      setData(result);
    } catch (err) {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const quickPicks = ['JFK', 'LAX', 'LHR', 'CDG', 'DXB', 'SIN', 'HND', 'ATL'];

  return (
    <div>
      <PageHeader title="Airport Intelligence" subtitle="Analyze airport connectivity and network position" icon={Building2} />

      <GlassCard className="mb-8 fade-in">
        <form onSubmit={handleSearch} className="flex gap-4 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <Input type="text" placeholder="Enter IATA code (e.g., JFK)" value={iata}
              onChange={(e) => setIata(e.target.value.toUpperCase())} maxLength={4}
              className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 h-12 font-mono text-lg" />
          </div>
          <Button type="submit" className="btn-primary h-12 px-8" disabled={!iata || loading}>
            <Search className="w-4 h-4 mr-2" />{loading ? 'Analyzing...' : 'Analyze'}
          </Button>
        </form>
        <div className="flex flex-wrap gap-2 mt-4">
          <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Quick:</span>
          {quickPicks.map(c => (
            <button key={c} onClick={() => setIata(c)} className="badge badge-blue" style={{ border: 'none', cursor: 'pointer' }}>{c}</button>
          ))}
        </div>
      </GlassCard>

      {loading && <LoadingSkeleton type="chart" />}

      {!loading && searched && data && (
        <div className="space-y-6 fade-in-up">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 stagger-children">
            <StatWidget icon={<Network className="w-5 h-5" />} title="Connections" value={data.total_connections} color="blue" />
            <StatWidget icon={<Plane className="w-5 h-5" />} title="Outbound" value={data.outbound_routes} color="emerald" />
            <StatWidget icon={<ArrowRightLeft className="w-5 h-5" />} title="Inbound" value={data.inbound_routes} color="violet" />
            <StatWidget icon={<Globe2 className="w-5 h-5" />} title="Countries" value={new Set(data.nodes?.map(n => n.country)).size} color="cyan" />
          </div>

          <GlassCard>
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Network className="w-4 h-4" style={{ color: '#60a5fa' }} /> Connectivity — {data.hub}
            </h3>
            <svg viewBox="-200 -200 400 400" className="w-full" style={{ height: '400px' }}>
              <circle cx="0" cy="0" r="8" fill="#3b82f6" opacity="0.9">
                <animate attributeName="r" values="8;10;8" dur="2s" repeatCount="indefinite" />
              </circle>
              <text x="0" y="-15" textAnchor="middle" fill="#60a5fa" fontSize="8" fontWeight="bold">{data.hub}</text>
              {data.nodes?.filter(n => !n.hub).slice(0, 60).map((node, i, arr) => {
                const angle = (i / arr.length) * 2 * Math.PI;
                const r = 120 + (i % 3) * 30;
                const x = Math.cos(angle) * r, y = Math.sin(angle) * r;
                return (
                  <g key={i}>
                    <line x1="0" y1="0" x2={x} y2={y} stroke="rgba(99,102,241,0.15)" strokeWidth="0.5" />
                    <circle cx={x} cy={y} r="3" fill="#6366f1" opacity="0.6" />
                    <text x={x} y={y - 6} textAnchor="middle" fill="#94a3b8" fontSize="4">{node.iata}</text>
                  </g>
                );
              })}
            </svg>
          </GlassCard>

          <GlassCard>
            <h3 className="font-semibold mb-4">Connected Airports ({(data.nodes?.length || 1) - 1})</h3>
            <div className="overflow-x-auto" style={{ maxHeight: '350px', overflowY: 'auto' }}>
              <table className="data-table">
                <thead><tr><th>IATA</th><th>Airport</th><th>Country</th></tr></thead>
                <tbody>
                  {data.nodes?.filter(n => !n.hub).slice(0, 50).map((n, i) => (
                    <tr key={i}>
                      <td><span className="badge badge-blue">{n.iata}</span></td>
                      <td className="font-medium text-sm">{n.name}</td>
                      <td className="text-sm" style={{ color: 'var(--av-text-muted)' }}>{n.country}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>
      )}

      {!loading && searched && !data && (
        <GlassCard className="text-center py-12">
          <Building2 className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--av-text-muted)' }} />
          <h3 className="text-lg font-semibold mb-2">Airport Not Found</h3>
          <p className="text-sm" style={{ color: 'var(--av-text-muted)' }}>Check the IATA code and try again</p>
        </GlassCard>
      )}
    </div>
  );
};

export default AirportIntelligence;
