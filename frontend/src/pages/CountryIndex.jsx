import { useState, useEffect } from 'react';
import { Flag, TrendingUp, Search } from 'lucide-react';
import { intelligenceApi } from '@/lib/api';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const CountryIndex = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await intelligenceApi.getCountryIndex(50);
        setData(result);
      } catch (err) {
        console.error('Error fetching country index:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div>
      <PageHeader
        title="Global Aviation Index"
        subtitle="Rankings of countries by overall aviation infrastructure and connectivity"
        icon={Flag}
      />

      {loading ? (
        <LoadingSkeleton type="table" rows={10} />
      ) : (
        <GlassCard className="fade-in">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-16">Rank</th>
                  <th>Country</th>
                  <th style={{ textAlign: 'right' }}>Aviation Index</th>
                  <th style={{ textAlign: 'right' }}>Airports</th>
                  <th style={{ textAlign: 'right' }}>Airlines</th>
                  <th style={{ textAlign: 'right' }}>Total Routes</th>
                </tr>
              </thead>
              <tbody>
                {data?.map((item, i) => (
                  <tr key={i}>
                    <td style={{ color: i < 3 ? '#fbbf24' : '#60a5fa', fontWeight: 700 }}>
                      #{i + 1}
                    </td>
                    <td className="font-bold">{item.country}</td>
                    <td style={{ textAlign: 'right' }}>
                      <span className="badge badge-emerald text-sm">
                        {item.aviation_index.toFixed(2)}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right', color: 'var(--av-text-muted)' }}>
                      {item.airports}
                    </td>
                    <td style={{ textAlign: 'right', color: 'var(--av-text-muted)' }}>
                      {item.airlines}
                    </td>
                    <td style={{ textAlign: 'right', color: 'var(--av-text-muted)' }}>
                      {item.routes.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}
    </div>
  );
};

export default CountryIndex;
