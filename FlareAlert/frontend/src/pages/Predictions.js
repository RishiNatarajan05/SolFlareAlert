import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';
import { Calendar, Clock, TrendingUp } from 'lucide-react';
import { apiService } from '../services/api';
import RiskGauge from '../components/RiskGauge';
import toast from 'react-hot-toast';

const Predictions = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const { data: prediction, isLoading, refetch } = useQuery(
    ['prediction', selectedDate],
    () => apiService.getPrediction(selectedDate),
    {
      enabled: !!selectedDate,
    }
  );

  const handleDateChange = (date) => {
    setSelectedDate(date);
  };

  const handleRefresh = () => {
    refetch();
    toast.success('Prediction refreshed');
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'MINIMAL': return 'text-solar-safe';
      case 'LOW': return 'text-solar-warning';
      case 'MEDIUM': return 'text-solar-cme';
      case 'HIGH': return 'text-solar-danger';
      default: return 'text-gray-400';
    }
  };

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
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Prediction Settings</h2>
          
          <div className="space-y-4">
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

        {/* Current Prediction */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Current Prediction</h2>
          
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="loading-spinner"></div>
            </div>
          ) : prediction?.data ? (
            <div className="space-y-6">
              <div className="flex justify-center">
                <RiskGauge
                  value={prediction.data.prediction}
                  riskLevel={prediction.data.risk_level}
                  size="lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Confidence</p>
                  <p className="text-2xl font-bold text-white">
                    {Math.round(prediction.data.confidence * 100)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400">Risk Level</p>
                  <p className={`text-2xl font-bold ${getRiskColor(prediction.data.risk_level)}`}>
                    {prediction.data.risk_level}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Prediction Time</span>
                  <span className="text-white">
                    {new Date(prediction.data.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Features Used</span>
                  <span className="text-white">{prediction.data.features_used}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Model Version</span>
                  <span className="text-white">{prediction.data.model_version}</span>
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
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <h2 className="text-xl font-semibold text-white mb-4">Prediction History</h2>
        <div className="text-center text-gray-400 py-8">
          Historical prediction data will be displayed here
        </div>
      </motion.div>
    </div>
  );
};

export default Predictions;
