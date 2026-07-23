import { useState, useEffect, useRef, useMemo } from 'react';
import { Globe2, Plane, MapPin, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { intelligenceApi } from '@/lib/api';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import StatWidget from '@/components/shared/StatWidget';
import Globe from 'react-globe.gl';

const GlobeExplorer = () => {
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const globeEl = useRef();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await intelligenceApi.getGlobalHeatmap();
        setHeatmapData(data);
      } catch (err) {
        console.error('Error fetching heatmap:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const getIntensityColor = (intensity) => {
    if (intensity > 200) return '#f43f5e';
    if (intensity > 100) return '#f59e0b';
    if (intensity > 50) return '#3b82f6';
    if (intensity > 10) return '#06b6d4';
    return '#6366f1';
  };

  const globeData = useMemo(() => {
    if (!heatmapData?.data) return [];
    return heatmapData.data.filter(d => d.lat !== 0 && d.lng !== 0).map(point => ({
      lat: point.lat,
      lng: point.lng,
      size: Math.max(0.1, point.intensity / 50),
      color: getIntensityColor(point.intensity),
      name: point.name,
      iata: point.iata,
      routes: point.intensity
    }));
  }, [heatmapData]);

  const topAirports = heatmapData?.data
    ?.filter(d => d.intensity > 0)
    .sort((a, b) => b.intensity - a.intensity)
    .slice(0, 10) || [];

  return (
    <div>
      <PageHeader
        title="Globe Explorer"
        subtitle="Interactive 3D WebGL global airport visualization"
        icon={Globe2}
      />

      {loading ? (
        <LoadingSkeleton type="chart" />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6 stagger-children">
            <StatWidget icon={<MapPin />} title="Airports Mapped" value={heatmapData?.total_points || 0} color="blue" />
            <StatWidget icon={<Plane />} title="Active Hubs" value={heatmapData?.data?.filter(d => d.intensity > 50).length || 0} color="rose" />
            <StatWidget icon={<Globe2 />} title="Countries" value={new Set(heatmapData?.data?.map(d => d.country)).size || 0} color="cyan" />
          </div>

          <GlassCard className="mb-6 relative overflow-hidden" padding={false}>
            <div style={{ height: '600px', cursor: 'grab' }}>
              <Globe
                ref={globeEl}
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
                backgroundColor="rgba(0,0,0,0)"
                pointsData={globeData}
                pointAltitude="size"
                pointColor="color"
                pointLabel={(d) => `
                  <div style="background: rgba(15,23,42,0.9); padding: 8px; border-radius: 4px; border: 1px solid #334155; font-family: sans-serif; color: white;">
                    <b>${d.name} (${d.iata})</b><br/>
                    Routes: <span style="color: #60a5fa">${d.routes}</span>
                  </div>
                `}
                pointRadius={0.5}
                pointsMerge={true}
              />
            </div>
          </GlassCard>

          <GlassCard>
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <MapPin className="w-4 h-4" style={{ color: '#f43f5e' }} /> Busiest Airport Hubs
            </h3>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr><th>Rank</th><th>IATA</th><th>Airport</th><th>Country</th><th>Lat</th><th>Lng</th><th style={{ textAlign: 'right' }}>Routes</th></tr>
                </thead>
                <tbody>
                  {topAirports.map((a, i) => (
                    <tr key={i}>
                      <td style={{ color: '#f43f5e', fontWeight: 700 }}>{i + 1}</td>
                      <td><span className="badge badge-rose">{a.iata}</span></td>
                      <td className="font-medium">{a.name}</td>
                      <td style={{ color: 'var(--av-text-muted)' }}>{a.country}</td>
                      <td className="font-mono text-xs">{a.lat?.toFixed(2)}</td>
                      <td className="font-mono text-xs">{a.lng?.toFixed(2)}</td>
                      <td style={{ textAlign: 'right', fontWeight: 600 }}>{a.intensity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </>
      )}
    </div>
  );
};

export default GlobeExplorer;
