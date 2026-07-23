import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const api = axios.create({
  baseURL: `${BACKEND_URL}/api`,
  timeout: 150000,
  headers: { 'Content-Type': 'application/json' },
});

const apiV2 = axios.create({
  baseURL: `${BACKEND_URL}/api/v2`,
  timeout: 150000,
  headers: { 'Content-Type': 'application/json' },
});


const handleResponse = (response) => response.data;
const handleError = (error) => {
  console.error('API Error:', error?.response?.data || error.message);
  throw error;
};

api.interceptors.response.use(handleResponse, handleError);
apiV2.interceptors.response.use(handleResponse, handleError);

// ===== V1 API (existing endpoints) =====
export const analyticsApi = {
  getStats: () => api.get('/analytics/stats'),
  getBusiestAirports: (limit = 15) => api.get(`/analytics/busiest-airports?limit=${limit}`),
  getTopAirlines: (limit = 15) => api.get(`/analytics/top-airlines?limit=${limit}`),
  getPopularRoutes: (limit = 15) => api.get(`/analytics/popular-routes?limit=${limit}`),
  getAirportsByCountry: (limit = 20) => api.get(`/analytics/airports-by-country?limit=${limit}`),
};

export const searchApi = {
  searchAirports: (q, limit = 20) => api.get(`/search/airports?q=${q}&limit=${limit}`),
  getRoutesByAirport: (id, limit = 50) => api.get(`/search/routes/${id}?limit=${limit}`),
};

export const recommendApi = {
  getDirectRoutes: (src, dest) => api.get(`/recommendations/direct-routes?source=${src}&destination=${dest}`),
  getSimilarRoutes: (src, dest, topK = 15) =>
    api.get(`/recommendations/similar-routes?source=${src}&destination=${dest}&top_k=${topK}`),
};

// ===== V2 API (new intelligence endpoints) =====
export const intelligenceApi = {
  getAirportConnectivity: (iata) => apiV2.get(`/intelligence/airport-connectivity/${iata}`),
  compareAirlines: (codes) => apiV2.get(`/intelligence/airline-comparison?airlines=${codes}`),
  getCountryIndex: (limit = 20) => apiV2.get(`/intelligence/country-index?limit=${limit}`),
  getNetworkGraph: (limit = 50) => apiV2.get(`/intelligence/network-graph?limit=${limit}`),
  getRouteScoring: (src, dest) => apiV2.get(`/intelligence/route-scoring?source=${src}&destination=${dest}`),
  getGlobalHeatmap: () => apiV2.get('/intelligence/global-heatmap'),
};

export const mlApi = {
  getClusters: (clusters = 5, limit = 500) => apiV2.get(`/ml/clusters?n_clusters=${clusters}&limit=${limit}`),
  getAnomalies: (limit = 1000) => apiV2.get(`/ml/anomalies?limit=${limit}`),
  getForecast: (iata) => apiV2.get(`/ml/forecast/${iata}`),
};

export const healthApi = {
  check: () => axios.get(`${BACKEND_URL}/health`).then(r => r.data),
};

export { api, apiV2 };
export default api;
