import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from 'react-query';
import { Database } from 'lucide-react';
import { apiService } from '../services/api';

const ModelInfo = () => {
  const { data: modelInfo } = useQuery('model-info', apiService.getModelInfo);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Model Information</h1>
        <p className="text-gray-400 mt-1">Machine learning model details and performance metrics</p>
      </div>

      {modelInfo?.data ? (
        <>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Model Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-gray-400">Model Type</p>
                <p className="text-lg font-semibold text-white">{modelInfo.data.model_type}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Features</p>
                <p className="text-lg font-semibold text-white">{modelInfo.data.features}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Last Updated</p>
                <p className="text-lg font-semibold text-white">
                  {new Date(modelInfo.data.last_updated).toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Model Path</p>
                <p className="text-sm text-gray-300 break-all">{modelInfo.data.model_details?.model_path}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Performance Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <p className="text-sm text-gray-400">AUC</p>
                <p className="text-2xl font-bold text-white">
                  {modelInfo.data.performance?.auc?.toFixed(3) || 'N/A'}
                </p>
              </div>
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <p className="text-sm text-gray-400">Precision</p>
                <p className="text-2xl font-bold text-white">
                  {modelInfo.data.performance?.precision?.toFixed(3) || 'N/A'}
                </p>
              </div>
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <p className="text-sm text-gray-400">Recall</p>
                <p className="text-2xl font-bold text-white">
                  {modelInfo.data.performance?.recall?.toFixed(3) || 'N/A'}
                </p>
              </div>
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <p className="text-sm text-gray-400">False Positive Rate</p>
                <p className="text-2xl font-bold text-white">
                  {modelInfo.data.performance?.false_positive_rate?.toFixed(3) || 'N/A'}
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
            <h2 className="text-xl font-semibold text-white mb-4">Model Details</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Model Type</span>
                <span className="text-white">{modelInfo.data.model_details?.model_type}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Features Count</span>
                <span className="text-white">{modelInfo.data.model_details?.features_count}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Scaler Available</span>
                <span className={`${modelInfo.data.model_details?.scaler_available ? 'text-green-400' : 'text-red-400'}`}>
                  {modelInfo.data.model_details?.scaler_available ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Last Updated</span>
                <span className="text-white">
                  {modelInfo.data.model_details?.last_updated ? 
                    new Date(modelInfo.data.model_details.last_updated).toLocaleString() : 'N/A'}
                </span>
              </div>
            </div>
          </motion.div>
        </>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="text-center text-gray-400 py-8">
            <Database className="w-12 h-12 mx-auto mb-4 text-gray-600" />
            <p>Loading model information...</p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ModelInfo;
