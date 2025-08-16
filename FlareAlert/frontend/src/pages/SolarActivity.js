import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';
import { Activity, Zap, AlertTriangle } from 'lucide-react';
import { apiService } from '../services/api';

const SolarActivity = () => {
  const { data: solarActivity } = useQuery('solar-activity', apiService.getSolarActivity, {
    refetchInterval: 60000,
  });

  const { data: historicalData } = useQuery('historical-data', () => apiService.getHistoricalData(7));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Solar Activity</h1>
        <p className="text-gray-400 mt-1">Real-time solar activity monitoring and historical data</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex items-center space-x-3">
            <Zap className="w-8 h-8 text-solar-flare" />
            <div>
              <p className="text-sm text-gray-400">Solar Flares (24h)</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.flares_24h || 0}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="flex items-center space-x-3">
            <Activity className="w-8 h-8 text-solar-cme" />
            <div>
              <p className="text-sm text-gray-400">CMEs (24h)</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.cmes_24h || 0}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-8 h-8 text-solar-storm" />
            <div>
              <p className="text-sm text-gray-400">Geomagnetic Storms</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.storms_24h || 0}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <h2 className="text-xl font-semibold text-white mb-4">Historical Data (7 Days)</h2>
        <div className="text-center text-gray-400 py-8">
          Historical solar activity charts will be displayed here
        </div>
      </motion.div>
    </div>
  );
};

export default SolarActivity;
