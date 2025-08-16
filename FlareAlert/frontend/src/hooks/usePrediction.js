import { useQuery } from 'react-query';
import { apiService } from '../services/api';
import config from '../services/config';

// Hook for current prediction (Dashboard)
export const useCurrentPrediction = () => {
  return useQuery(
    'current-prediction',
    () => apiService.getPrediction(),
    {
      refetchInterval: config.REFRESH_INTERVALS.PREDICTION,
      staleTime: 300000, // 5 minutes
      cacheTime: 600000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: 1,
    }
  );
};

// Hook for historical predictions (Predictions page)
export const useHistoricalPrediction = (date) => {
  return useQuery(
    ['historical-prediction', date],
    () => apiService.getPrediction(date),
    {
      enabled: !!date, // Only run query if date is provided
      staleTime: 300000, // 5 minutes
      cacheTime: 600000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      retry: 1,
    }
  );
};

// Hook for prediction with custom options
export const usePrediction = (queryKey, options = {}) => {
  const defaultOptions = {
    staleTime: 300000,
    cacheTime: 600000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
    retry: 1,
    ...options,
  };

  return useQuery(
    queryKey,
    () => apiService.getPrediction(),
    defaultOptions
  );
};

// Hook for prediction history
export const usePredictionHistory = (days = 7) => {
  return useQuery(
    ['prediction-history', days],
    async () => {
      const response = await apiService.getPredictionHistory(days);
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
};
