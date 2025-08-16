import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity, BarChart3 } from 'lucide-react';

const PredictionHistory = ({ data, isLoading }) => {
  const prefersReducedMotion = useMemo(() => 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches, 
    []
  );

  const getRiskColor = (level) => {
    switch (level) {
      case 'MINIMAL': return 'text-solar-safe';
      case 'LOW': return 'text-solar-warning';
      case 'MEDIUM': return 'text-solar-cme';
      case 'HIGH': return 'text-solar-danger';
      default: return 'text-gray-400';
    }
  };

  const getRiskBgColor = (level) => {
    switch (level) {
      case 'MINIMAL': return 'bg-green-500/20 border-green-500/30';
      case 'LOW': return 'bg-yellow-500/20 border-yellow-500/30';
      case 'MEDIUM': return 'bg-orange-500/20 border-orange-500/30';
      case 'HIGH': return 'bg-red-500/20 border-red-500/30';
      default: return 'bg-gray-500/20 border-gray-500/30';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!data?.predictions?.length) {
    return (
      <div className="text-center text-gray-400 py-8">
        <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-600" />
        <p>No historical prediction data available</p>
        <p className="text-sm mt-2">Historical data will appear here once predictions are made</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-gray-700 rounded-lg">
          <p className="text-sm text-gray-400">Total</p>
          <p className="text-xl font-bold text-white">{data.predictions.length}</p>
        </div>
        <div className="text-center p-3 bg-gray-700 rounded-lg">
          <p className="text-sm text-gray-400">Average</p>
          <p className="text-xl font-bold text-white">
            {Math.round(data.predictions.reduce((sum, p) => sum + p.prediction, 0) / data.predictions.length * 100)}%
          </p>
        </div>
        <div className="text-center p-3 bg-gray-700 rounded-lg">
          <p className="text-sm text-gray-400">Highest</p>
          <p className="text-xl font-bold text-white">
            {Math.round(Math.max(...data.predictions.map(p => p.prediction)) * 100)}%
          </p>
        </div>
        <div className="text-center p-3 bg-gray-700 rounded-lg">
          <p className="text-sm text-gray-400">Lowest</p>
          <p className="text-xl font-bold text-white">
            {Math.round(Math.min(...data.predictions.map(p => p.prediction)) * 100)}%
          </p>
        </div>
      </div>

      {/* Prediction List */}
      <div>
        <h3 className="text-lg font-medium text-white mb-3">Recent Predictions</h3>
        <div className="space-y-3">
          {data.predictions.slice(0, 5).map((item, index) => (
            <motion.div
              key={index}
              initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 10 }}
              animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.2, delay: index * 0.05 }}
              className={`p-4 rounded-lg border ${getRiskBgColor(item.risk_level)}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Activity className="w-5 h-5 text-gray-400" />
                  <div>
                    <p className="text-white font-medium">
                      {new Date(item.timestamp).toLocaleDateString()}
                    </p>
                    <p className="text-sm text-gray-400">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-white">
                    {Math.round(item.prediction * 100)}%
                  </p>
                  <p className={`text-sm font-medium ${getRiskColor(item.risk_level)}`}>
                    {item.risk_level}
                  </p>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-600/30">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="text-white">{Math.round(item.confidence * 100)}%</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PredictionHistory;
