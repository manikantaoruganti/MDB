import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import {
  TrendingUp, ArrowLeft, Map, Plane, Globe, Building2, BarChart3
} from 'lucide-react';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Analytics = () => {
  const [activeTab, setActiveTab] = useState('airports');
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData(activeTab);
  }, [activeTab]);

  const fetchData = async (tab) => {
    setLoading(true);
    try {
      let endpoint = '';
      switch (tab) {
        case 'airports':
          endpoint = '/analytics/busiest-airports?limit=15';
          break;
        case 'airlines':
          endpoint = '/analytics/top-airlines?limit=15';
          break;
        case 'routes':
          endpoint = '/analytics/popular-routes?limit=15';
          break;
        case 'countries':
          endpoint = '/analytics/airports-by-country?limit=20';
          break;
        default:
          endpoint = '/analytics/busiest-airports?limit=15';
      }
      const response = await axios.get(`${API}${endpoint}`);
      setData(response.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'airports', icon: Map, label: 'Busiest Airports', color: '#60a5fa' },
    { id: 'airlines', icon: Plane, label: 'Top Airlines', color: '#a78bfa' },
    { id: 'routes', icon: TrendingUp, label: 'Popular Routes', color: '#fb7185' },
    { id: 'countries', icon: Globe, label: 'By Country', color: '#22d3ee' },
  ];

  return (
    <div>
      <PageHeader
        title="Analytics Dashboard"
        subtitle="Explore aviation data insights and trends"
        icon={BarChart3}
      />

      {/* Tabs */}
      <div className="flex flex-wrap gap-3 mb-8 fade-in" data-testid="analytics-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              data-testid={`tab-${tab.id}`}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all"
              style={{
                background: isActive
                  ? 'var(--av-gradient-primary)'
                  : 'var(--av-glass)',
                color: isActive ? 'white' : 'var(--av-text-secondary)',
                border: `1px solid ${isActive ? 'transparent' : 'var(--av-glass-border)'}`,
                cursor: 'pointer',
                backdropFilter: 'blur(12px)',
              }}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="fade-in">
        {loading ? (
          <LoadingSkeleton type="table" rows={10} />
        ) : (
          <GlassCard className="overflow-hidden" data-testid="analytics-content">
            {activeTab === 'airports' && <AirportsTable data={data} />}
            {activeTab === 'airlines' && <AirlinesTable data={data} />}
            {activeTab === 'routes' && <RoutesTable data={data} />}
            {activeTab === 'countries' && <CountriesTable data={data} />}
          </GlassCard>
        )}
      </div>
    </div>
  );
};

/* ========== TABLE COMPONENTS (preserved logic, enhanced styling) ========== */

const AirportsTable = ({ data }) => (
  <div className="overflow-x-auto" data-testid="airports-table">
    <table className="data-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Airport</th>
          <th>City</th>
          <th>Country</th>
          <th>IATA</th>
          <th style={{ textAlign: 'right' }}>Routes</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={index} data-testid={`airport-row-${index}`}>
            <td style={{ color: '#60a5fa', fontWeight: 700 }}>{index + 1}</td>
            <td className="font-medium">{item.name}</td>
            <td style={{ color: 'var(--av-text-muted)' }}>{item.city}</td>
            <td style={{ color: 'var(--av-text-muted)' }}>{item.country}</td>
            <td><span className="badge badge-blue">{item.iata}</span></td>
            <td style={{ textAlign: 'right', fontWeight: 600 }}>{item.routes?.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const AirlinesTable = ({ data }) => (
  <div className="overflow-x-auto" data-testid="airlines-table">
    <table className="data-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Airline</th>
          <th>IATA</th>
          <th>Country</th>
          <th style={{ textAlign: 'right' }}>Routes</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={index} data-testid={`airline-row-${index}`}>
            <td style={{ color: '#a78bfa', fontWeight: 700 }}>{index + 1}</td>
            <td className="font-medium">{item.name}</td>
            <td><span className="badge badge-violet">{item.iata || 'N/A'}</span></td>
            <td style={{ color: 'var(--av-text-muted)' }}>{item.country}</td>
            <td style={{ textAlign: 'right', fontWeight: 600 }}>{item.routes?.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const RoutesTable = ({ data }) => (
  <div className="overflow-x-auto" data-testid="routes-table">
    <table className="data-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Route</th>
          <th>From</th>
          <th>To</th>
          <th style={{ textAlign: 'right' }}>Airlines</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={index} data-testid={`route-row-${index}`}>
            <td style={{ color: '#fb7185', fontWeight: 700 }}>{index + 1}</td>
            <td>
              <span className="badge badge-rose">{item.source} → {item.dest}</span>
            </td>
            <td style={{ color: 'var(--av-text-muted)' }}>{item.source_name}</td>
            <td style={{ color: 'var(--av-text-muted)' }}>{item.dest_name}</td>
            <td style={{ textAlign: 'right', fontWeight: 600 }}>{item.airlines?.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const CountriesTable = ({ data }) => (
  <div className="overflow-x-auto" data-testid="countries-table">
    <table className="data-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Country</th>
          <th style={{ textAlign: 'right' }}>Airports</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={index} data-testid={`country-row-${index}`}>
            <td style={{ color: '#22d3ee', fontWeight: 700 }}>{index + 1}</td>
            <td className="font-medium">{item.country}</td>
            <td style={{ textAlign: 'right', fontWeight: 600 }}>{item.airports?.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default Analytics;
