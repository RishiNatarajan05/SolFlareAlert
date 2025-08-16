import React, { useState, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Calendar, Clock, TrendingUp, CalendarDays } from 'lucide-react';
import { useHistoricalPrediction, useCurrentPrediction, usePredictionHistory } from '../hooks/usePrediction';
import RiskGauge from '../components/RiskGauge';
import PredictionHistory from '../components/PredictionHistory';
import toast from 'react-hot-toast';

const Predictions = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [showHistorical, setShowHistorical] = useState(false);

  // Check for reduced motion preference
  const prefersReducedMotion = useMemo(() => 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches, 
    []
  );

  // Get current prediction (same as Dashboard)
  const { data: currentPrediction, isLoading: currentLoading, refetch: refetchCurrent } = useCurrentPrediction();
  
  // Get historical prediction (only when needed)
  const { data: historicalPrediction, isLoading: historicalLoading, refetch: refetchHistorical } = useHistoricalPrediction(
    showHistorical ? selectedDate : null
  );

  // Get prediction history
  const { data: historyData, isLoading: historyLoading } = usePredictionHistory(7);

  const handleDateChange = useCallback((date) => {
    setSelectedDate(date);
  }, []);

  const handleRefresh = useCallback(() => {
    if (showHistorical) {
      refetchHistorical();
    } else {
      refetchCurrent();
    }
    toast.success('Prediction refreshed');
  }, [showHistorical, refetchHistorical, refetchCurrent]);

  const getRiskColor = useCallback((level) => {
    switch (level) {
      case 'MINIMAL': return 'text-solar-safe';
      case 'LOW': return 'text-solar-warning';
      case 'MEDIUM': return 'text-solar-cme';
      case 'HIGH': return 'text-solar-danger';
      default: return 'text-gray-400';
    }
  }, []);

  // Use current prediction by default, historical when selected
  const activePrediction = showHistorical ? historicalPrediction : currentPrediction;
  const isLoading = showHistorical ? historicalLoading : currentLoading;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Solar Flare Predictions</h1>
          <p className="text-gray-400 mt-1">Detailed prediction analysis and forecasting</p>
        </div>
        <button onClick={handleRefresh} className="btn-primary">
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Controls */}
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, x: -20 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, x: 0 }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Prediction Settings</h2>
          
          <div className="space-y-4">
            {/* Prediction Type Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Prediction Type
              </label>
              <div className="flex space-x-2">
                <button
                  onClick={() => setShowHistorical(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    !showHistorical 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <TrendingUp className="w-4 h-4 inline mr-2" />
                  Current
                </button>
                <button
                  onClick={() => setShowHistorical(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    showHistorical 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <CalendarDays className="w-4 h-4 inline mr-2" />
                  Historical
                </button>
              </div>
            </div>

            {/* Date Selector (only for historical) */}
            {showHistorical && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Prediction Date
                </label>
                <div className="flex items-center space-x-2">
                  <Calendar className="w-5 h-5 text-gray-400" />
                  <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => handleDateChange(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
              </div>
            )}

            {/* Info about current prediction */}
            {!showHistorical && (
              <div className="p-3 bg-blue-900/20 border border-blue-700/30 rounded-lg">
                <p className="text-sm text-blue-300">
                  <Clock className="w-4 h-4 inline mr-2" />
                  Showing current prediction for the next 6 hours
                </p>
              </div>
            )}

            <div className="pt-4 border-t border-gray-700">
              <h3 className="text-lg font-medium text-white mb-2">Risk Level Guide</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-solar-safe">MINIMAL</span>
                  <span className="text-sm text-gray-400">0-20% probability</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-solar-warning">LOW</span>
                  <span className="text-sm text-gray-400">20-40% probability</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-solar-cme">MEDIUM</span>
                  <span className="text-sm text-gray-400">40-70% probability</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-solar-danger">HIGH</span>
                  <span className="text-sm text-gray-400">70-100% probability</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Prediction Display */}
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, x: 20 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, x: 0 }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">
            {showHistorical ? 'Historical Prediction' : 'Current Prediction'}
          </h2>
          
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="loading-spinner"></div>
            </div>
          ) : activePrediction?.data ? (
            <div className="space-y-6">
              <div className="flex justify-center">
                <RiskGauge
                  value={activePrediction.data.prediction}
                  riskLevel={activePrediction.data.risk_level}
                  size="lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Confidence</p>
                  <p className="text-2xl font-bold text-white">
                    {Math.round(activePrediction.data.confidence * 100)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Risk Level</p>
                  <p className={`text-2xl font-bold ${getRiskColor(activePrediction.data.risk_level)}`}>
                    {activePrediction.data.risk_level}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Prediction Time</span>
                  <span className="text-white">
                    {new Date(activePrediction.data.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Features Used</span>
                  <span className="text-white">{activePrediction.data.features_used}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Model Version</span>
                  <span className="text-white">{activePrediction.data.model_version}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              No prediction data available
            </div>
          )}
        </motion.div>
      </div>

      {/* Prediction History */}
      <motion.div
        initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
        animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
        transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
        className="card"
      >
        <h2 className="text-xl font-semibold text-white mb-4">Prediction History</h2>
        <PredictionHistory data={historyData} isLoading={historyLoading} />
      </motion.div>
    </div>
  );
};

export default Predictions;
