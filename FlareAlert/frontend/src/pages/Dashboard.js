import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';
import { 
  Activity, 
  Zap, 
  AlertTriangle, 
  TrendingUp, 
  RefreshCw
} from 'lucide-react';
import { apiService } from '../services/api';
import { useWebSocket } from '../services/WebSocketContext';
import { useCurrentPrediction } from '../hooks/usePrediction';
import RiskGauge from '../components/RiskGauge';
import StatCard from '../components/StatCard';
import config from '../services/config';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const { latestData, alerts } = useWebSocket();
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const prefersReducedMotion = useMemo(() => 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches, 
    []
  );

  const { refetch: refetchPrediction, data: predictionData } = useCurrentPrediction();

  const { data: solarActivity, refetch: refetchSolarActivity } = useQuery(
    'solar-activity',
    apiService.getSolarActivity,
    {
      refetchInterval: config.REFRESH_INTERVALS.SOLAR_ACTIVITY,
      staleTime: 240000,
      cacheTime: 600000,
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: 1,
    }
  );

  const { data: modelInfo } = useQuery(
    'model-info',
    apiService.getModelInfo,
    {
      refetchInterval: config.REFRESH_INTERVALS.MODEL_INFO,
      staleTime: 600000,
      cacheTime: 900000,
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: 1,
    }
  );

  useEffect(() => {
    if (predictionData && !currentPrediction) {
      setCurrentPrediction(predictionData.data);
      setLastUpdated(new Date());
    }
  }, [predictionData, currentPrediction]);

  useEffect(() => {
    if (latestData && latestData.type === 'status_update') {
      setCurrentPrediction({
        prediction: latestData.prediction,
        confidence: latestData.confidence,
        risk_level: latestData.risk_level,
        timestamp: latestData.timestamp,
      });
      setLastUpdated(new Date());
    }
  }, [latestData]);

  const handleRefresh = useCallback(async () => {
    try {
      await Promise.all([refetchPrediction(), refetchSolarActivity()]);
      toast.success('Data refreshed successfully');
    } catch (error) {
      toast.error('Failed to refresh data');
    }
  }, [refetchPrediction, refetchSolarActivity]);

  const getRiskColor = useCallback((level) => {
    switch (level) {
      case 'MINIMAL':
        return 'text-solar-safe';
      case 'LOW':
        return 'text-solar-warning';
      case 'MEDIUM':
        return 'text-solar-cme';
      case 'HIGH':
        return 'text-solar-danger';
      default:
        return 'text-gray-400';
    }
  }, []);

  // Memoize solar activity data to prevent unnecessary re-renders
  const solarActivityData = useMemo(() => ({
    flares_30d: solarActivity?.data?.flares_30d || 0,
    cmes_30d: solarActivity?.data?.cmes_30d || 0,
    storms_30d: solarActivity?.data?.storms_30d || 0,
    current_kp: solarActivity?.data?.current_kp || 'N/A',
  }), [solarActivity]);

  // Memoize recent alerts to prevent unnecessary re-renders
  const recentAlerts = useMemo(() => 
    alerts.slice(0, 5), 
    [alerts]
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Solar Flare Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Real-time monitoring and prediction of solar flare activity
          </p>
        </div>
        <button
          onClick={handleRefresh}
          className="btn-primary flex items-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Current Risk Level */}
      <motion.div
        initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, scale: 0.95 }}
        animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, scale: 1 }}
        transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
        className="card"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-white">Current Risk Level</h2>
            <p className="text-gray-400">Solar flare probability for next 6 hours</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-400">Last Updated</p>
            <p className="text-sm text-white">
              {lastUpdated.toLocaleTimeString()}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-center">
          {currentPrediction ? (
            <RiskGauge
              value={currentPrediction.prediction}
              riskLevel={currentPrediction.risk_level}
              size="lg"
            />
          ) : (
            <div className="text-center">
              <div className="loading-spinner mx-auto mb-4"></div>
              <p className="text-gray-400">Loading prediction...</p>
            </div>
          )}
        </div>

        {currentPrediction && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-sm text-gray-400">Confidence</p>
              <p className="text-lg font-semibold text-white">
                {Math.round(currentPrediction.confidence * 100)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-400">Risk Level</p>
              <p className={`text-lg font-semibold ${getRiskColor(currentPrediction.risk_level)}`}>
                {currentPrediction.risk_level}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-400">Features Used</p>
              <p className="text-lg font-semibold text-white">
                {currentPrediction.features_used || config.DEFAULTS.FEATURES_COUNT}
              </p>
            </div>
          </div>
        )}
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Solar Flares (30d)"
          value={solarActivityData.flares_30d}
          icon={Zap}
          color="text-solar-flare"
          subtitle="Last 30 days"
          delay={prefersReducedMotion ? 0 : 0}
        />
        <StatCard
          title="CMEs (30d)"
          value={solarActivityData.cmes_30d}
          icon={Activity}
          color="text-solar-cme"
          subtitle="Last 30 days"
          delay={prefersReducedMotion ? 0 : 1}
        />
        <StatCard
          title="Geomagnetic Storms"
          value={solarActivityData.storms_30d}
          icon={AlertTriangle}
          color="text-solar-storm"
          subtitle="Last 30 days"
          delay={prefersReducedMotion ? 0 : 2}
        />
        <StatCard
          title="Current Kp Index"
          value={solarActivityData.current_kp}
          icon={TrendingUp}
          color="text-blue-400"
          subtitle="Geomagnetic activity"
          delay={prefersReducedMotion ? 0 : 3}
        />
      </div>

      {/* Recent Alerts */}
      {recentAlerts.length > 0 && (
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 10 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Recent Alerts</h2>
          <div className="space-y-3">
            {recentAlerts.map((alert) => (
              <div
                key={alert.id}
                className="flex items-center justify-between p-3 bg-gray-700 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <AlertTriangle className={`w-5 h-5 ${getRiskColor(alert.risk_level)}`} />
                  <div>
                    <p className="text-white font-medium">{alert.message}</p>
                    <p className="text-sm text-gray-400">
                      {new Date(alert.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskColor(alert.risk_level)}`}>
                  {alert.risk_level}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Model Status */}
      {modelInfo && (
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 10 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Model Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-400">Model Type</p>
              <p className="text-white font-medium">{modelInfo.data.model_type}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Features</p>
              <p className="text-white font-medium">{modelInfo.data.features}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Performance (AUC)</p>
              <p className="text-white font-medium">
                {modelInfo.data.performance?.auc?.toFixed(3) || 'N/A'}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Dashboard;
