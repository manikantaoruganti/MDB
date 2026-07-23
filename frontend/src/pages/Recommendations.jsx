import { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Search, Plane, Sparkles, TrendingUp, Route, Zap } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import PageHeader from '@/components/shared/PageHeader';
import GlassCard from '@/components/shared/GlassCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Recommendations = () => {
  const [source, setSource] = useState('');
  const [destination, setDestination] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!source || !destination) return;

    setLoading(true);
    setSearched(true);
    try {
      const response = await axios.get(
        `${API}/recommendations/similar-routes?source=${source}&destination=${destination}&top_k=15`
      );
      setRecommendations(response.data);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  };

  const getSimilarityColor = (sim) => {
    if (sim >= 0.8) return '#34d399';
    if (sim >= 0.5) return '#fbbf24';
    return '#60a5fa';
  };

  return (
    <div>
      <PageHeader
        title="Route Recommendations"
        subtitle="Find similar routes using AI-powered vector similarity"
        icon={Route}
      />

      {/* Search Form */}
      <GlassCard className="mb-8 fade-in" data-testid="search-form">
        <form onSubmit={handleSearch} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--av-text-muted)' }}>
                Source Airport (IATA)
              </label>
              <Input
                type="text"
                placeholder="e.g., JFK, LAX, LHR"
                value={source}
                onChange={(e) => setSource(e.target.value.toUpperCase())}
                maxLength={3}
                data-testid="source-input"
                className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 h-12 text-lg font-mono"
              />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--av-text-muted)' }}>
                Destination Airport (IATA)
              </label>
              <Input
                type="text"
                placeholder="e.g., SFO, ORD, CDG"
                value={destination}
                onChange={(e) => setDestination(e.target.value.toUpperCase())}
                maxLength={3}
                data-testid="destination-input"
                className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 h-12 text-lg font-mono"
              />
            </div>
          </div>
          <Button
            type="submit"
            className="w-full btn-primary h-12 text-sm font-semibold"
            disabled={!source || !destination || loading}
            data-testid="search-btn"
          >
            <Search className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing Routes...' : 'Find Similar Routes'}
          </Button>
        </form>
      </GlassCard>

      {/* How it works */}
      <GlassCard hover={false} className="mb-8 fade-in" style={{ animationDelay: '100ms' }}>
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-xl" style={{ background: 'rgba(245,158,11,0.12)' }}>
            <Sparkles className="w-5 h-5" style={{ color: '#fbbf24' }} />
          </div>
          <div>
            <h3 className="font-semibold text-sm mb-1">How the AI Engine Works</h3>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--av-text-muted)' }}>
              Our recommendation engine uses TF-IDF (Term Frequency-Inverse Document Frequency) 
              vectorization with character n-grams to create high-dimensional embeddings for each route. 
              Cosine similarity is then computed between your query and all 67,000+ routes to find 
              the most similar connections in the aviation network.
            </p>
          </div>
        </div>
      </GlassCard>

      {/* Results */}
      {loading && <LoadingSkeleton type="table" rows={8} />}

      {!loading && searched && recommendations.length > 0 && (
        <div className="fade-in-up" data-testid="recommendations-results">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl" style={{ background: 'rgba(59,130,246,0.12)' }}>
              <Zap className="w-5 h-5" style={{ color: '#60a5fa' }} />
            </div>
            <h2 className="text-xl font-bold" style={{ fontFamily: 'Space Grotesk' }}>
              Similar Routes
            </h2>
            <span className="badge badge-blue">{recommendations.length} results</span>
          </div>

          <div className="grid gap-3">
            {recommendations.map((rec, index) => (
              <GlassCard
                key={index}
                className="group"
                data-testid={`recommendation-${index}`}
              >
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-5">
                    <div className="text-2xl font-bold" style={{ color: '#60a5fa', fontFamily: 'Space Grotesk', minWidth: '3rem' }}>
                      #{index + 1}
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="badge badge-blue text-sm">{rec.source}</span>
                        <Plane className="w-4 h-4" style={{ color: 'var(--av-text-muted)', transform: 'rotate(-45deg)' }} />
                        <span className="badge badge-violet text-sm">{rec.dest}</span>
                      </div>
                      <p className="text-xs" style={{ color: 'var(--av-text-muted)' }}>
                        Airline: {rec.airline || 'Various'}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--av-text-muted)' }}>
                      Similarity
                    </p>
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${(rec.similarity * 100)}%`,
                            background: getSimilarityColor(rec.similarity),
                          }}
                        />
                      </div>
                      <span className="text-lg font-bold" style={{ color: getSimilarityColor(rec.similarity), fontFamily: 'JetBrains Mono' }}>
                        {(rec.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {!loading && searched && recommendations.length === 0 && (
        <GlassCard className="text-center py-12 fade-in" data-testid="no-results">
          <Plane className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--av-text-muted)' }} />
          <h3 className="text-lg font-semibold mb-2">No routes found</h3>
          <p className="text-sm" style={{ color: 'var(--av-text-muted)' }}>
            Try different airport codes or check if they exist in our database
          </p>
        </GlassCard>
      )}
    </div>
  );
};

export default Recommendations;
