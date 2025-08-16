import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';
import { Activity, Zap, AlertTriangle } from 'lucide-react';
import { apiService } from '../services/api';
import config from '../services/config';
import SolarActivityChart from '../components/SolarActivityChart';

const SolarActivity = () => {
  const { data: solarActivity } = useQuery('solar-activity', apiService.getSolarActivity, {
    refetchInterval: config.REFRESH_INTERVALS.SOLAR_ACTIVITY,
  });

  // Get historical data for the past 30 days
  const { data: historicalData, isLoading: historicalLoading } = useQuery(
    'historical-data-30',
    async () => {
      const response = await apiService.getHistoricalData(30);
      return response.data; // Extract the data from the Axios response
    },
    {
      staleTime: 600000, // 10 minutes
      cacheTime: 900000, // 15 minutes
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: 1,
    }
  );

  // Check for reduced motion preference
  const prefersReducedMotion = useMemo(() => 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches, 
    []
  );



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
              <p className="text-sm text-gray-400">Solar Flares (30d)</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.flares_30d || 0}
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
              <p className="text-sm text-gray-400">CMEs (30d)</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.cmes_30d || 0}
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
              <p className="text-sm text-gray-400">Geomagnetic Storms (30d)</p>
              <p className="text-2xl font-bold text-white">
                {solarActivity?.data?.storms_30d || 0}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
        animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
        transition={{ duration: prefersReducedMotion ? 0 : 0.5 }}
        className="card"
      >
        <SolarActivityChart data={historicalData} isLoading={historicalLoading} />
      </motion.div>
    </div>
  );
};

export default SolarActivity;
