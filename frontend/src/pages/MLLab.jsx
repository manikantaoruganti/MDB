import { useState, useEffect } from 'react';
import { Brain, Sparkles, Database, Code, ShieldAlert, Activity, Plane } from 'lucide-react';
import { mlApi } from '@/lib/api';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const MLLab = () => {
  const [loading, setLoading] = useState(true);
  const [anomalies, setAnomalies] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [forecastInput, setForecastInput] = useState('JFK');
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    const fetchMLData = async () => {
      try {
        const [anomaliesData, clustersData] = await Promise.all([
          mlApi.getAnomalies(500),
          mlApi.getClusters(5, 300)
        ]);
        setAnomalies(anomaliesData);
        setClusters(clustersData);
        await loadForecast('JFK');
      } catch (err) {
        console.error('Failed to load ML data', err);
      } finally {
        setLoading(false);
      }
    };
    fetchMLData();
  }, []);

  const loadForecast = async (iata) => {
    setIsPredicting(true);
    try {
      const data = await mlApi.getForecast(iata);
      setForecast(data);
    } catch (err) {
      console.error(err);
      setForecast({ error: err?.response?.data?.detail || "No data available for this airport" });
    } finally {
      setIsPredicting(false);
    }
  };

  const handleForecastSearch = (e) => {
    e.preventDefault();
    if (forecastInput) {
      loadForecast(forecastInput);
    }
  };

  if (loading) {
    return (
      <div>
        <PageHeader title="Machine Learning Lab" subtitle="Experimental AI models and predictive analytics" icon={Brain} />
        <LoadingSkeleton type="card" count={3} />
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Machine Learning Lab"
        subtitle="Experimental AI models and predictive analytics"
        icon={Brain}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 fade-in mb-8">
        <GlassCard>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
              <Database className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-lg">Demand Forecasting (ARIMA/Prophet Sim)</h3>
          </div>
          
          <form onSubmit={handleForecastSearch} className="flex gap-2 mb-4">
            <input 
              type="text" 
              value={forecastInput}
              onChange={(e) => setForecastInput(e.target.value.toUpperCase())}
              placeholder="Airport IATA (e.g. JFK)"
              className="bg-[var(--av-surface)] border border-[var(--av-glass-border)] rounded-lg px-3 py-1.5 text-sm w-full outline-none focus:border-blue-500/50"
            />
            <button type="submit" disabled={isPredicting} className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm transition-colors disabled:opacity-50">
              {isPredicting ? 'Predicting...' : 'Predict'}
            </button>
          </form>

          {forecast && !forecast.error && (
            <div>
              <div className="flex justify-between text-xs text-[var(--av-text-muted)] mb-2">
                <span>Base Demand Index: {forecast.base_route_count}</span>
              </div>
              <div className="h-40 flex items-end gap-1 mt-4">
                {forecast.forecast?.map((f, i) => {
                  const max = Math.max(...forecast.forecast.map(x => x.passengers));
                  const height = `${(f.passengers / max) * 100}%`;
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center gap-2 group">
                      <div className="w-full bg-blue-500/20 rounded-t-sm relative hover:bg-blue-500/40 transition-colors" style={{ height }}>
                        <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2 bg-[var(--av-surface)] border border-[var(--av-glass-border)] px-2 py-1 rounded text-xs whitespace-nowrap z-10 transition-opacity">
                          {f.passengers.toLocaleString()}
                        </div>
                      </div>
                      <span className="text-[10px] text-[var(--av-text-muted)]">{f.month}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {forecast?.error && <p className="text-red-400 text-sm">{forecast.error}</p>}
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-lg">Anomaly Detection (Isolation Forest)</h3>
          </div>
          <p className="text-sm text-[var(--av-text-muted)] mb-4">
            Top anomalous routes based on connectivity structure and flight density.
          </p>
          <div className="space-y-2 max-h-[220px] overflow-y-auto custom-scrollbar pr-2">
            {anomalies.map((anom, i) => (
              <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[var(--av-surface)] border border-[var(--av-glass-border)]">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1 font-mono text-sm">
                    <span className="text-rose-400">{anom.source}</span>
                    <Plane className="w-3 h-3 text-[var(--av-text-muted)]" />
                    <span className="text-blue-400">{anom.dest}</span>
                  </div>
                  <span className="text-xs text-[var(--av-text-muted)] border-l border-[var(--av-glass-border)] pl-3">
                    Airlines: {anom.airline_count}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] uppercase text-[var(--av-text-muted)] block">Anomaly Score</span>
                  <span className="text-sm font-mono text-rose-400 font-semibold">{anom.anomaly_score.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      <GlassCard>
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
            <Code className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-lg">Route Clustering (K-Means on TF-IDF)</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {clusters.map((cluster) => (
            <div key={cluster.id} className="p-4 rounded-xl bg-[var(--av-surface)] border border-[var(--av-glass-border)]">
              <div className="flex justify-between items-center mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-[var(--av-text-muted)]">Cluster {cluster.id}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-mono">
                  n={cluster.count}
                </span>
              </div>
              <div className="space-y-2">
                {cluster.routes.map((r, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm font-mono">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400/50" />
                    <span className="text-[var(--av-text-secondary)]">{r.source}</span>
                    <span className="text-[var(--av-text-muted)]">→</span>
                    <span className="text-[var(--av-text-secondary)]">{r.dest}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </GlassCard>

    </div>
  );
};

export default MLLab;
