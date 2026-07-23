import { useState, useEffect } from 'react';
import { Network, Activity, ZoomIn, ZoomOut, Move } from 'lucide-react';
import { intelligenceApi } from '@/lib/api';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const NetworkVisualizer = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await intelligenceApi.getNetworkGraph(100);
        setData(result);
      } catch (err) {
        console.error('Error fetching network:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div>
      <PageHeader
        title="Global Network Visualizer"
        subtitle="Aviation network topology and hub connections"
        icon={Network}
      />

      {loading ? (
        <LoadingSkeleton type="chart" />
      ) : (
        <div className="space-y-6 fade-in">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
             <GlassCard padding={false} className="p-4 text-center">
               <p className="text-xs text-[var(--av-text-muted)] uppercase tracking-wider mb-1">Nodes (Airports)</p>
               <p className="text-2xl font-bold text-blue-400 font-mono">{data?.total_nodes || 0}</p>
             </GlassCard>
             <GlassCard padding={false} className="p-4 text-center">
               <p className="text-xs text-[var(--av-text-muted)] uppercase tracking-wider mb-1">Edges (Routes)</p>
               <p className="text-2xl font-bold text-violet-400 font-mono">{data?.total_edges || 0}</p>
             </GlassCard>
          </div>

          <GlassCard padding={false} className="relative overflow-hidden h-[600px] flex items-center justify-center">
            <div className="absolute inset-0 opacity-20 bg-grid" />
            
            {/* Placeholder for complex D3/Canvas graph - rendered as stylized SVG abstract for now */}
            <svg width="100%" height="100%" viewBox="-500 -500 1000 1000">
              <defs>
                <radialGradient id="glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="rgba(99,102,241,0.4)" />
                  <stop offset="100%" stopColor="transparent" />
                </radialGradient>
              </defs>
              
              <circle cx="0" cy="0" r="400" fill="url(#glow)" />
              
              {data?.edges?.slice(0, 500).map((edge, i) => {
                 // Simulated layout
                 const sourceNode = data.nodes.find(n => n.iata === edge.source);
                 const targetNode = data.nodes.find(n => n.iata === edge.target);
                 
                 if (!sourceNode || !targetNode) return null;
                 
                 // Map lat/lng to roughly circular layout for visual effect if real force layout isn't running
                 const sLng = sourceNode.lng * 2;
                 const sLat = -sourceNode.lat * 2;
                 const tLng = targetNode.lng * 2;
                 const tLat = -targetNode.lat * 2;
                 
                 return (
                   <line 
                     key={i} 
                     x1={sLng} y1={sLat} 
                     x2={tLng} y2={tLat} 
                     stroke="rgba(99,102,241,0.15)" 
                     strokeWidth={Math.max(0.5, edge.weight / 10)} 
                   />
                 )
              })}

              {data?.nodes?.map((node, i) => {
                 const x = node.lng * 2;
                 const y = -node.lat * 2;
                 const r = Math.max(2, Math.min(10, Math.sqrt(node.outbound) / 3));
                 
                 return (
                   <g key={i}>
                     <circle cx={x} cy={y} r={r} fill="#60a5fa" opacity="0.8" />
                     {r > 5 && (
                       <text x={x} y={y - r - 2} fill="var(--av-text-muted)" fontSize="8" textAnchor="middle">
                         {node.iata}
                       </text>
                     )}
                   </g>
                 )
              })}
            </svg>
            
            <div className="absolute bottom-4 right-4 glass-static rounded-lg p-3 max-w-xs text-xs" style={{ color: 'var(--av-text-muted)' }}>
              <Activity className="w-4 h-4 mb-2" style={{ color: '#6366f1' }} />
              <p>Network visualization displays topological relationships between major aviation hubs. Node size indicates outbound volume. Edge thickness indicates route frequency.</p>
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  );
};

export default NetworkVisualizer;
