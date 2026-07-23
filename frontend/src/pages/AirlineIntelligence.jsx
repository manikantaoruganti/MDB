import { useState } from 'react';
import { Plane, Search, TrendingUp, Map, Navigation, ArrowRight, Globe2 } from 'lucide-react';
import { intelligenceApi } from '@/lib/api';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const AirlineIntelligence = () => {
  const [airlines, setAirlines] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!airlines) return;
    setLoading(true);
    setSearched(true);
    try {
      const result = await intelligenceApi.compareAirlines(airlines);
      setData(result);
    } catch (err) {
      console.error('Error:', err);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const quickPicks = ['DL,AA,UA', 'EK,QR,EY', 'FR,U2,W6', 'BA,LH,AF'];

  return (
    <div>
      <PageHeader
        title="Airline Intelligence"
        subtitle="Compare airline networks, coverage, and connectivity performance"
        icon={Plane}
      />

      <GlassCard className="mb-8 fade-in">
        <form onSubmit={handleSearch} className="flex gap-4 flex-wrap">
          <div className="flex-1 min-w-[300px]">
            <Input
              type="text"
              placeholder="Enter comma-separated IATA codes (e.g., DL, AA, UA)"
              value={airlines}
              onChange={(e) => setAirlines(e.target.value.toUpperCase())}
              className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 h-12 font-mono text-lg"
            />
          </div>
          <Button type="submit" className="btn-primary h-12 px-8" disabled={!airlines || loading}>
            <Search className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Compare Airlines'}
          </Button>
        </form>

        <div className="flex flex-wrap gap-2 mt-4">
          <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Comparisons:</span>
          {quickPicks.map(pick => (
            <button
              key={pick}
              onClick={() => setAirlines(pick)}
              className="badge badge-violet cursor-pointer hover:opacity-80 transition-opacity"
              style={{ border: 'none' }}
            >
              {pick}
            </button>
          ))}
        </div>
      </GlassCard>

      {loading && <LoadingSkeleton type="table" rows={6} />}

      {!loading && searched && data && data.length > 0 && (
        <div className="space-y-6 fade-in-up">
          <div className="grid gap-6">
            {data.map((airline, idx) => (
              <GlassCard key={idx} className="relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full" style={{ background: idx === 0 ? '#fbbf24' : '#6366f1' }} />
                
                <div className="flex flex-col md:flex-row gap-6 justify-between items-start md:items-center">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="badge badge-blue text-lg">{airline.code}</span>
                      <h3 className="text-xl font-bold" style={{ fontFamily: 'Space Grotesk' }}>{airline.name}</h3>
                      {idx === 0 && <span className="badge badge-amber text-xs">Top Performer</span>}
                    </div>
                    <p className="text-sm" style={{ color: 'var(--av-text-muted)' }}>
                      Based in {airline.country} · {airline.active ? 'Active Carrier' : 'Inactive'}
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-4 text-center">
                    <div>
                      <p className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--av-text-muted)' }}>Score</p>
                      <p className="text-3xl font-bold" style={{ color: idx === 0 ? '#fbbf24' : '#6366f1', fontFamily: 'JetBrains Mono' }}>
                        {airline.connectivity_score}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t" style={{ borderColor: 'var(--av-glass-border)' }}>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-4 h-4" style={{ color: '#60a5fa' }} />
                      <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Total Routes</span>
                    </div>
                    <p className="text-lg font-semibold">{airline.total_routes.toLocaleString()}</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Navigation className="w-4 h-4" style={{ color: '#a78bfa' }} />
                      <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Unique Origins</span>
                    </div>
                    <p className="text-lg font-semibold">{airline.unique_sources.toLocaleString()}</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Map className="w-4 h-4" style={{ color: '#34d399' }} />
                      <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Destinations</span>
                    </div>
                    <p className="text-lg font-semibold">{airline.unique_destinations.toLocaleString()}</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Globe2 className="w-4 h-4" style={{ color: '#f43f5e' }} />
                      <span className="text-xs" style={{ color: 'var(--av-text-muted)' }}>Countries Served</span>
                    </div>
                    <p className="text-lg font-semibold">{airline.countries_served}</p>
                  </div>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {!loading && searched && data?.length === 0 && (
        <GlassCard className="text-center py-12">
          <Plane className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--av-text-muted)' }} />
          <h3 className="text-lg font-semibold mb-2">No Airlines Found</h3>
          <p className="text-sm" style={{ color: 'var(--av-text-muted)' }}>
            Check the IATA codes and try again
          </p>
        </GlassCard>
      )}
    </div>
  );
};

export default AirlineIntelligence;
