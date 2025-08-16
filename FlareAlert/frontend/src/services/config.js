const config = {
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_BASE_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  
  REFRESH_INTERVALS: {
    PREDICTION: process.env.REACT_APP_PREDICTION_REFRESH || 600000,
    SOLAR_ACTIVITY: process.env.REACT_APP_SOLAR_ACTIVITY_REFRESH || 300000,
    MODEL_INFO: process.env.REACT_APP_MODEL_INFO_REFRESH || 900000,
  },
  
  ALERTS: {
    DEFAULT_THRESHOLD: process.env.REACT_APP_DEFAULT_ALERT_THRESHOLD || 0.4,
    TOAST_DURATION: process.env.REACT_APP_TOAST_DURATION || 3000,
    HIGH_RISK_DURATION: process.env.REACT_APP_HIGH_RISK_TOAST_DURATION || 6000,
  },
  
  RISK_LEVELS: {
    MINIMAL: { min: 0, max: 0.2, color: '#4ade80' },
    LOW: { min: 0.2, max: 0.4, color: '#fbbf24' },
    MEDIUM: { min: 0.4, max: 0.7, color: '#ff8c42' },
    HIGH: { min: 0.7, max: 1.0, color: '#ef4444' },
  },
  
  WEBSOCKET: {
    RECONNECT_DELAY: process.env.REACT_APP_WS_RECONNECT_DELAY || 5000,
    MAX_RECONNECT_ATTEMPTS: process.env.REACT_APP_WS_MAX_RECONNECT || 10,
    THROTTLE_INTERVAL: 1000,
  },
  
  UI: {
    ANIMATION_DURATION: 300,
    GAUGE_SIZES: {
      sm: 'w-24 h-24',
      md: 'w-32 h-32',
      lg: 'w-48 h-48'
    },
    REDUCED_MOTION: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  },
  
  DEFAULTS: {
    FEATURES_COUNT: 51,
    MODEL_VERSION: 'hazard_ensemble_v1.0',
    CONFIDENCE: 0.85,
  }
};

export default config;
