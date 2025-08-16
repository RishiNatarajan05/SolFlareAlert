import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings as SettingsIcon, Bell, Database, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const Settings = () => {
  const [alertThreshold, setAlertThreshold] = useState(0.4);
  const [email, setEmail] = useState('');
  const [webhookUrl, setWebhookUrl] = useState('');

  const handleSaveAlerts = async () => {
    try {
      await apiService.configureAlerts({
        risk_threshold: alertThreshold,
        email,
        webhook_url: webhookUrl,
      });
      toast.success('Alert settings saved successfully');
    } catch (error) {
      toast.error('Failed to save alert settings');
    }
  };

  const handleDataIngestion = async () => {
    try {
      await apiService.triggerDataIngestion();
      toast.success('Data ingestion started');
    } catch (error) {
      toast.error('Failed to start data ingestion');
    }
  };

  const handleModelRetraining = async () => {
    try {
      await apiService.retrainModel({
        force_retrain: true,
        use_latest_data: true,
      });
      toast.success('Model retraining started');
    } catch (error) {
      toast.error('Failed to start model retraining');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Configure system settings and alerts</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alert Settings */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <Bell className="w-5 h-5" />
            <span>Alert Settings</span>
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Risk Threshold ({Math.round(alertThreshold * 100)}%)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={alertThreshold}
                onChange={(e) => setAlertThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Email Notifications
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Webhook URL
              </label>
              <input
                type="url"
                value={webhookUrl}
                onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://your-webhook-url.com"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <button
              onClick={handleSaveAlerts}
              className="btn-primary w-full"
            >
              Save Alert Settings
            </button>
          </div>
        </motion.div>

        {/* System Actions */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <SettingsIcon className="w-5 h-5" />
            <span>System Actions</span>
          </h2>
          
          <div className="space-y-4">
            <div className="p-4 bg-gray-700 rounded-lg">
              <h3 className="text-lg font-medium text-white mb-2">Data Management</h3>
              <p className="text-sm text-gray-400 mb-3">
                Trigger data ingestion from NASA DONKI API
              </p>
              <button
                onClick={handleDataIngestion}
                className="btn-secondary flex items-center space-x-2"
              >
                <Database className="w-4 h-4" />
                <span>Start Data Ingestion</span>
              </button>
            </div>

            <div className="p-4 bg-gray-700 rounded-lg">
              <h3 className="text-lg font-medium text-white mb-2">Model Management</h3>
              <p className="text-sm text-gray-400 mb-3">
                Retrain the machine learning model with latest data
              </p>
              <button
                onClick={handleModelRetraining}
                className="btn-secondary flex items-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Retrain Model</span>
              </button>
            </div>
          </div>
        </motion.div>
      </div>

      {/* System Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <h2 className="text-xl font-semibold text-white mb-4">System Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-400">Version</p>
            <p className="text-white font-medium">1.0.0</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">API Status</p>
            <p className="text-green-400 font-medium">Connected</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Last Update</p>
            <p className="text-white font-medium">
              {new Date().toLocaleString()}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Settings;
