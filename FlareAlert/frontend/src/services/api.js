import axios from 'axios';
import config from './config';

const api = axios.create({
  baseURL: config.API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API endpoints
export const apiService = {
  // Health check
  getHealth: () => api.get('/health'),
  
  // Predictions
  getPrediction: (timestamp) => api.post('/predict', { timestamp }),
  
  // Model information
  getModelInfo: () => api.get('/model-info'),
  
  // Solar activity
  getSolarActivity: () => api.get('/solar-activity'),
  
  // Historical data
  getHistoricalData: (days = 7) => api.get(`/historical-data?days=${days}`),
  
  // Data ingestion
  triggerDataIngestion: () => api.post('/ingest-data'),
  
  // Model retraining
  retrainModel: (options = {}) => api.post('/retrain-model', options),
  
  // Alert configuration
  configureAlerts: (config) => api.post('/alerts', config),
};

export default api;
